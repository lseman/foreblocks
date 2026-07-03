"""foreblocks.experimental.attention_kernels.src.tilelang_bwd.

TileLang flash-attention backward kernel for custom attention dispatch.

On RTX 4090 (Ada) the TileLang backward runs ~1.4-1.5x faster than Triton at
D=128 and is roughly SDPA-parity. Ported from tilelang's ``example_mha_bwd_bhsd``
with dtype-parametric support (fp16/bf16), custom softmax_scale, and natural-log
to log2-domain LSE conversion.

Core API:
- tilelang_flash_bwd: backward pass via TileLang; mirrors triton_flash_bwd signature
- can_use_tilelang_bwd: check whether TileLang backward kernel can handle input tensor

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


def _tl_dtype(torch_dtype):
    return T.float16 if torch_dtype == torch.float16 else T.bfloat16


def make_dq_layout(dQ):
    # atomicAdd cannot be vectorized, so reorder dq to match the 8x8 gemm fragment.
    return T.Layout(
        dQ.shape,
        lambda b, h, l, d: [b, h, l // 8, d // 8, (d % 2), 4 * (l % 8) + (d % 8) // 2],
    )


@functools.lru_cache(maxsize=None)
def _build_prep(batch, heads, seq_len, dim, dtype_key):
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    blk = 32

    @tilelang.jit(
        out_idx=[2], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    )
    def build():
        @T.prim_func
        def flash_bwd_prep(
            O: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
                o = T.alloc_fragment([blk, blk], dtype)
                do = T.alloc_fragment([blk, blk], dtype)
                acc = T.alloc_fragment([blk, blk], accum_dtype)
                delta = T.alloc_fragment([blk], accum_dtype)
                T.clear(acc)
                for k in range(T.ceildiv(dim, blk)):
                    T.copy(
                        O[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk], o
                    )
                    T.copy(
                        dO[bz, bx, by * blk : (by + 1) * blk, k * blk : (k + 1) * blk],
                        do,
                    )
                    for i, j in T.Parallel(blk, blk):
                        acc[i, j] += o[i, j] * do[i, j]
                T.reduce_sum(acc, delta, 1)
                T.copy(delta, Delta[bz, bx, by * blk : (by + 1) * blk])

        return flash_bwd_prep

    return build()


@functools.lru_cache(maxsize=None)
def _build_post(batch, heads, seq_len, dim, dtype_key):
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    blk = 64

    @tilelang.jit(
        out_idx=[1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}
    )
    def build():
        @T.prim_func
        def flash_bwd_post(
            dQ: T.Tensor(shape, accum_dtype),
            dQ_out: T.Tensor(shape, dtype),
        ):
            with T.Kernel(T.ceildiv(seq_len, blk), heads, batch, threads=128) as (
                bx,
                by,
                bz,
            ):
                T.annotate_layout({dQ: make_dq_layout(dQ)})
                T.copy(
                    dQ[bz, by, bx * blk : (bx + 1) * blk, :],
                    dQ_out[bz, by, bx * blk : (bx + 1) * blk, :],
                )

        return flash_bwd_post

    return build()


@functools.lru_cache(maxsize=None)
def _build_bwd(
    batch, heads, seq_len, dim, is_causal, block_M, block_N, scale, dtype_key
):
    """scale is the natural softmax scale (1/sqrt(d) or custom). The kernel uses
    scale*log2(e) internally because LSE is fed in log2 domain."""
    dtype = T.float16 if dtype_key == "fp16" else T.bfloat16
    accum_dtype = T.float32
    shape = [batch, heads, seq_len, dim]
    sm_scale = scale
    log2_scale = scale * _LOG2E

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True})
    def build():
        @T.prim_func
        def flash_bwd(
            Q: T.Tensor(shape, dtype),
            K: T.Tensor(shape, dtype),
            V: T.Tensor(shape, dtype),
            dO: T.Tensor(shape, dtype),
            lse: T.Tensor([batch, heads, seq_len], accum_dtype),
            Delta: T.Tensor([batch, heads, seq_len], accum_dtype),
            dQ: T.Tensor(shape, accum_dtype),
            dK: T.Tensor(shape, dtype),
            dV: T.Tensor(shape, dtype),
        ):
            with T.Kernel(heads, T.ceildiv(seq_len, block_M), batch, threads=128) as (
                bx,
                by,
                bz,
            ):
                K_shared = T.alloc_shared([block_M, dim], dtype)
                dsT_shared = T.alloc_shared([block_M, block_N], dtype)
                q = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_M, dim], dtype)
                qkT = T.alloc_fragment([block_M, block_N], accum_dtype)
                dsT = T.alloc_fragment([block_M, block_N], accum_dtype)
                qkT_cast = T.alloc_fragment([block_M, block_N], dtype)
                dsT_cast = T.alloc_fragment([block_M, block_N], dtype)
                lse_shared = T.alloc_shared([block_N], accum_dtype)
                delta = T.alloc_shared([block_N], accum_dtype)
                do = T.alloc_shared([block_N, dim], dtype)
                dv = T.alloc_fragment([block_M, dim], accum_dtype)
                dk = T.alloc_fragment([block_M, dim], accum_dtype)
                dq = T.alloc_fragment([block_N, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_M, dim], dtype)
                dk_shared = T.alloc_shared([block_M, dim], dtype)

                T.annotate_layout({dQ: make_dq_layout(dQ)})
                T.copy(K[bz, bx, by * block_M : (by + 1) * block_M, :], K_shared)
                T.copy(V[bz, bx, by * block_M : (by + 1) * block_M, :], V_shared)
                T.clear(dv)
                T.clear(dk)
                loop_st = T.floordiv(by * block_M, block_N) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_N)
                for k in T.Pipelined(loop_st, loop_ed, num_stages=2):
                    T.copy(Q[bz, bx, k * block_N : (k + 1) * block_N, :], q)
                    T.clear(qkT)
                    T.gemm(
                        K_shared,
                        q,
                        qkT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(lse[bz, bx, k * block_N : (k + 1) * block_N], lse_shared)
                    for i, j in T.Parallel(block_M, block_N):
                        qkT[i, j] = T.exp2(qkT[i, j] * log2_scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            qkT[i, j] = T.if_then_else(
                                by * block_M + i <= k * block_N + j, qkT[i, j], 0
                            )
                    T.copy(dO[bz, bx, k * block_N : (k + 1) * block_N, :], do)
                    T.clear(dsT)
                    T.gemm(
                        V_shared,
                        do,
                        dsT,
                        transpose_B=True,
                        policy=T.GemmWarpPolicy.FullRow,
                    )
                    T.copy(qkT, qkT_cast)
                    T.gemm(qkT_cast, do, dv, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(Delta[bz, bx, k * block_N : (k + 1) * block_N], delta)
                    for i, j in T.Parallel(block_M, block_N):
                        dsT_cast[i, j] = qkT[i, j] * (dsT[i, j] - delta[j]) * sm_scale
                    T.gemm(dsT_cast, q, dk, policy=T.GemmWarpPolicy.FullRow)

                    T.copy(dsT_cast, dsT_shared)
                    T.clear(dq)
                    T.gemm(dsT_shared, K_shared, dq, transpose_A=True)
                    for i, j in T.Parallel(block_N, dim):
                        T.atomic_add(dQ[bz, bx, k * block_N + i, j], dq[i, j])
                T.copy(dv, dv_shared)
                T.copy(dk, dk_shared)
                T.copy(dv_shared, dV[bz, bx, by * block_M : (by + 1) * block_M, :])
                T.copy(dk_shared, dK[bz, bx, by * block_M : (by + 1) * block_M, :])

        return flash_bwd

    return build()


def can_use_tilelang_bwd(q):
    return (
        _HAVE_TILELANG
        and q.is_cuda
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] in (64, 128)
        and q.shape[-2] % 64 == 0  # kernels tile by 64; require aligned seq_len
    )


def tilelang_flash_bwd(grad_out, q, k, v, out, lse, causal=False, softmax_scale=None):
    """Backward via tilelang. Signature mirrors ``triton_flash_bwd``.

    ``lse`` is the package's natural-log LSE; converted to log2 domain here.
    """
    B, H, N, D = q.shape
    scale = softmax_scale if softmax_scale is not None else D**-0.5
    dtype_key = "fp16" if q.dtype == torch.float16 else "bf16"

    block_M = 64
    block_N = 64 if D <= 64 else 32

    do = grad_out.contiguous()
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    out = out.contiguous()
    lse_log2 = (lse * _LOG2E).contiguous()

    prep = _build_prep(B, H, N, D, dtype_key)
    post = _build_post(B, H, N, D, dtype_key)
    kern = _build_bwd(
        B, H, N, D, bool(causal), block_M, block_N, float(scale), dtype_key
    )

    delta = prep(out, do)
    dq = torch.zeros((B, H, N, D), dtype=torch.float32, device=q.device)
    dk = torch.empty((B, H, N, D), dtype=q.dtype, device=q.device)
    dv = torch.empty((B, H, N, D), dtype=q.dtype, device=q.device)
    kern(q, k, v, do, lse_log2, delta, dq, dk, dv)
    dq = post(dq)
    return dq, dk, dv
