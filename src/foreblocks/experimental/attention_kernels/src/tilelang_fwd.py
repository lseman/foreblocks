"""custom_att.tilelang_fwd.

TileLang flash-attention forward kernel using TVM-based compilation.

SOTA forward pass matching patterns from TileLang's ``example_mha_fwd_bhsd``.
Supports fp16/bf16, causal and non-causal, head dims 16/32/64/128/256. The
LSE output is in log2-domain format; callers convert to natural-log for
compatibility with the rest of the codebase.

Core API:
- tilelang_flash_fwd: flash attention forward via TileLang
- can_use_tilelang_fwd: check whether the TileLang forward kernel can handle input tensor

"""

import functools

import torch

try:
    import tilelang
    import tilelang.language as T

    _HAVE_TILELANG = True
except Exception:  # pragma: no cover - import guard
    _HAVE_TILELANG = False

_LOG2E = 1.4426950408889634


# ---------------------------------------------------------------------------
# Forward kernel builder
# ---------------------------------------------------------------------------


@functools.cache
def _build_fwd(
    batch,
    heads,
    seq_len,
    dim,
    is_causal,
    block_M,
    block_N,
    num_stages,
    threads,
    dtype_key,
    log2_scale,
):
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    scale = log2_scale  # free variable captured by prim_func closure
    _LOG2E = 1.4426950408889634  # free var for LSE conversion

    @tilelang.jit(
        out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    )
    def build():
        @T.prim_func
        def flash_fwd(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(
                T.ceildiv(seq_len, block_M), heads, batch, threads=threads
            ) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)

                # Load Q block once
                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                # Loop range for K/V tiles
                loop_range = (
                    T.ceildiv((bx + 1) * block_M, block_N)
                    if is_causal
                    else T.ceildiv(seq_len, block_N)
                )

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    # Load K tile
                    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)

                    # Initialize acc_s and apply causal / OOB mask before GEMM
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0,
                                T.cast(-1e30, accum_dtype),
                            )
                    else:
                        T.fill(acc_s, 0)
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len,
                                T.cast(-1e30, accum_dtype),
                                0,
                            )

                    # Q @ K^T → acc_s  (FullRow: each warp gets a full row)
                    T.gemm(
                        Q_shared,
                        K_shared,
                        acc_s,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )

                    # Load V (overlaps with softmax work on next iter)
                    T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)

                    # ---- Online softmax ----
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, T.cast(-1e30, accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])

                    # scores_scale = exp2((scores_max_prev - scores_max) * scale)
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale
                        )

                    # Rescale partial output accumulator
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]

                    # P = exp2(acc_s * scale - scores_max * scale)
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale
                        )

                    # Cast P from fp32 to fp16/bf16 for GEMM
                    T.copy(acc_s, acc_s_cast)

                    # acc_o += P @ V
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

                    # scores_sum = sum(P, dim=1)
                    T.reduce_sum(acc_s, scores_sum, dim=1)

                    # logsum = logsum * scores_scale + scores_sum
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                # Final normalisation
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

                # LSE in natural-log domain (matches Triton forward):
                #   logsum = log2(logsum) + scores_max * scale  (log2-domain)
                #   lse = logsum / log2(e)  (natural-log)
                for i in T.Parallel(block_M):
                    logsum[i] = (T.log2(logsum[i]) + scores_max[i] * scale) / _LOG2E
                T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

        return flash_fwd

    return build()


# ---------------------------------------------------------------------------
# Config selection
# ---------------------------------------------------------------------------


def _select_config(n_ctx, d_head, causal):
    if d_head <= 32:
        return 128, 128, 2, 128
    if d_head == 64:
        return 128, 64, 2, 128
    # D>=96: num_stages=1 to avoid shared memory pressure
    return 64, 64, 1, 128


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def tilelang_flash_fwd(q, k, v, causal=False, softmax_scale=None):
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    softmax_scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    B, H = q.shape[0], q.shape[1]

    block_M, block_N, num_stages, threads = _select_config(n_ctx, d_head, bool(causal))
    dtype_key = "fp16" if q.dtype == torch.float16 else "bf16"
    log2_scale = float(softmax_scale * _LOG2E)

    fwd_kern = _build_fwd(
        B,
        H,
        n_ctx,
        d_head,
        bool(causal),
        block_M,
        block_N,
        num_stages,
        threads,
        dtype_key,
        log2_scale,
    )
    out, lse = fwd_kern(q, k, v)
    return out, lse


def can_use_tilelang_fwd(q):
    if not _HAVE_TILELANG:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        return False
    if q.shape[-2] % 64 != 0:
        return False
    return True
