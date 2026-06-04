"""TileLang flash-attention forward kernel.

SOTA forward pass using TileLang's TVM-based compilation, matching the
patterns from examples/flash_attention/example_mha_fwd_bhsd.py.

Supports fp16 / bf16, causal and non-causal, head dims 16/32/64/128/256.
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


@functools.lru_cache(maxsize=None)
def _build_fwd(batch, heads, seq_len, dim, is_causal, block_M, block_N,
               num_stages, threads, dtype_key):
    """Build and cache the flash-attention forward kernel."""
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]

    @tilelang.jit(out_idx=[3, 4],
                  pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build():
        @T.prim_func
        def flash_fwd(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            # Scale factor: softmax_scale * log2(e), applied in exp2
            scale = T.var("float32")
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                          threads=threads) as (bx, by, bz):
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

                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((bx + 1) * block_M, block_N) \
                    if is_causal else T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len,
                                -T.infinity(acc_s.dtype), 0)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

        return flash_fwd

    return build()


def _select_tilelang_fwd_config(n_ctx, d_head, causal):
    """Select TileLang forward config matching the official examples."""
    # Default from example_mha_fwd_bhsd.py: block_M=64, block_N=64, num_stages=1, threads=128
    return 64, 64, 1, 128


def tilelang_flash_fwd(q, k, v, causal=False, softmax_scale=None):
    """Flash attention forward via TileLang.

    Args:
        q, k, v: tensors of shape [B, H, N, D]
        causal: causal masking
        softmax_scale: defaults to 1/sqrt(D)

    Returns:
        out: [B, H, N, D] attention output
        lse: [B, H, N] same-format LSE as TileLang backward (log2-domain)
    """
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    softmax_scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    B, H = q.shape[0], q.shape[1]

    block_M, block_N, num_stages, threads = _select_tilelang_fwd_config(
        n_ctx, d_head, bool(causal))
    dtype_key = "fp16" if q.dtype == torch.float16 else "bf16"
    scale = float(softmax_scale * _LOG2E)

    fwd_kern = _build_fwd(B, H, n_ctx, d_head, bool(causal),
                          block_M, block_N, num_stages, threads, dtype_key)

    # Inject scale into the kernel closure
    # The prim_func uses `scale` as a free variable; we need to pass it.
    # Looking at the TileLang example, scale is captured from the outer scope.
    # So we need to use a different approach: define the kernel with a scale parameter.

    # Actually, the TileLang JIT compiler captures free variables from the closure.
    # The `scale` variable in the prim_func is a free variable captured from build().
    # But `build()` is a function, so `scale` is a local variable in `tilelang_flash_fwd`.
    # We need to pass it to `build()` and have it captured by the prim_func.

    # Let me restructure: pass scale as a parameter to _build_fwd.
    raise NotImplementedError("See below for fixed version")


def tilelang_flash_fwd_fixed(q, k, v, causal=False, softmax_scale=None):
    """Fixed forward that passes scale to the kernel builder."""
    n_ctx = q.shape[-2]
    d_head = q.shape[-1]
    softmax_scale = softmax_scale if softmax_scale is not None else d_head**-0.5
    B, H = q.shape[0], q.shape[1]

    block_M, block_N, num_stages, threads = _select_tilelang_fwd_config(
        n_ctx, d_head, bool(causal))
    dtype_key = "fp16" if q.dtype == torch.float16 else "bf16"
    log2_scale = float(softmax_scale * _LOG2E)

    fwd_kern = _build_fwd_v2(B, H, n_ctx, d_head, bool(causal),
                             block_M, block_N, num_stages, threads,
                             dtype_key, log2_scale)
    out, lse = fwd_kern(q, k, v)
    return out, lse


@functools.lru_cache(maxsize=None)
def _build_fwd_v2(batch, heads, seq_len, dim, is_causal,
                  block_M, block_N, num_stages, threads,
                  dtype_key, log2_scale):
    """Build forward kernel with scale baked in via closure."""
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    scale = log2_scale  # free variable captured by prim_func

    @tilelang.jit(out_idx=[3, 4],
                  pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build():
        @T.prim_func
        def flash_fwd(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            Output: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch,
                          threads=threads) as (bx, by, bz):
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

                T.copy(Q[bz, by, bx * block_M : (bx + 1) * block_M, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))

                loop_range = T.ceildiv((bx + 1) * block_M, block_N) \
                    if is_causal else T.ceildiv(seq_len, block_N)

                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    T.copy(K[bz, by, k * block_N : (k + 1) * block_N, :], K_shared)

                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                bx * block_M + i >= k * block_N + j,
                                0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(
                                k * block_N + j >= seq_len,
                                -T.infinity(acc_s.dtype), 0)

                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True,
                           policy=T.GemmWarpPolicy.FullRow)

                    T.copy(V[bz, by, k * block_N : (k + 1) * block_N, :], V_shared)

                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_M):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_M):
                        scores_scale[i] = T.exp2(
                            scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_M, dim):
                        acc_o[i, j] *= scores_scale[i]
                    for i, j in T.Parallel(block_M, block_N):
                        acc_s[i, j] = T.exp2(
                            acc_s[i, j] * scale - scores_max[i] * scale)
                    T.copy(acc_s, acc_s_cast)
                    T.gemm(acc_s_cast, V_shared, acc_o,
                           policy=T.GemmWarpPolicy.FullRow)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_M):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] /= logsum[i]
                T.copy(acc_o, Output[bz, by, bx * block_M : (bx + 1) * block_M, :])

                for i in T.Parallel(block_M):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_M : (bx + 1) * block_M])

        return flash_fwd

    return build()


# ---------------------------------------------------------------------------
# Replaced main entry point
# ---------------------------------------------------------------------------

tilelang_flash_fwd = tilelang_flash_fwd_fixed


def can_use_tilelang_fwd(q):
    """Check if TileLang forward kernel can handle this tensor."""
    if not _HAVE_TILELANG:
        return False
    if not q.is_cuda:
        return False
    if q.dtype not in (torch.float16, torch.bfloat16):
        return False
    if q.shape[-1] not in (16, 32, 64, 128, 256):
        return False
    block_M = 64
    if q.shape[-2] % block_M != 0:
        return False
    return True
