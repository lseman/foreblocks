"""foreblocks.ops.kernels.grouped_gemm.

Grouped GEMM with variable M per expert — Triton-accelerated matrix multiply.

Supports concatenated MoE expert segments where each expert processes a variable
number of tokens. The Triton forward kernel is autotuned; the backward falls back
to PyTorch per-expert matmuls. Use when building Mixture-of-Experts models or
any architecture that needs per-group matrix multiplication with varying group sizes.

Core API:
- grouped_mm_varM: compute concat_e(A_e @ B_e) with variable M_e, full autograd
- TRITON_AVAILABLE: whether the Triton path is usable
- _split_by_offsets: split a packed tensor by offset boundaries

"""

# grouped_gemm.py
# -----------------------------------------------------------------------------
# Grouped GEMM (Triton) for packed MoE segments with variable M per expert.
#
# - Triton kernel is autotuned over BLOCK_M/BLOCK_N/BLOCK_K.
# - DO NOT pass BLOCK_* at launch; autotune provides them.
# - Triton path is inference-only; if grads are needed we use torch.mm.
# -----------------------------------------------------------------------------

from collections.abc import Sequence

import torch

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False


class _GroupedMMVarMFunction(torch.autograd.Function):
    """
    Triton forward for grouped variable-M GEMM with manual backward.
    Backward is computed with torch matmuls per expert segment.
    """

    @staticmethod
    def forward(ctx, A_packed, offsets, B_cat, use_fp16_acc: bool):
        if not TRITON_AVAILABLE:
            raise RuntimeError("_GroupedMMVarMFunction requires Triton.")
        if not A_packed.is_cuda:
            raise RuntimeError("_GroupedMMVarMFunction expects CUDA tensors.")

        if not A_packed.is_contiguous():
            A_packed = A_packed.contiguous()
        if not offsets.is_contiguous():
            offsets = offsets.contiguous()
        if not B_cat.is_contiguous():
            B_cat = B_cat.contiguous()

        E = offsets.numel() - 1
        K = B_cat.shape[1]
        N = B_cat.shape[2]
        C_packed = torch.empty(
            (A_packed.shape[0], N), device=A_packed.device, dtype=A_packed.dtype
        )

        starts = offsets[:-1].to(torch.int32).contiguous()
        M_per = (offsets[1:] - offsets[:-1]).to(torch.int32).contiguous()

        stride_a_m, stride_a_k = A_packed.stride()
        stride_b_g, stride_b_k, stride_b_n = B_cat.stride()
        stride_c_m, stride_c_n = C_packed.stride()
        acc_dtype = tl.float16 if use_fp16_acc else tl.float32

        max_M = int(M_per.max().item()) if E > 0 else 0
        grid = lambda meta: (  # noqa: E731
            E,
            triton.cdiv(max_M, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        _grouped_gemm_varM_kernel[grid](
            A_packed,
            B_cat,
            C_packed,
            starts,
            M_per,
            K=K,
            N=N,
            stride_a_m=stride_a_m,
            stride_a_k=stride_a_k,
            stride_b_g=stride_b_g,
            stride_b_k=stride_b_k,
            stride_b_n=stride_b_n,
            stride_c_m=stride_c_m,
            stride_c_n=stride_c_n,
            ACC_DTYPE=acc_dtype,
        )

        ctx.save_for_backward(A_packed, B_cat, offsets)
        return C_packed

    @staticmethod
    def backward(ctx, grad_out):
        A_packed, B_cat, offsets = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        needs_A = ctx.needs_input_grad[0]
        needs_B = ctx.needs_input_grad[2]

        E = offsets.numel() - 1
        K = B_cat.shape[1]
        N = B_cat.shape[2]

        starts = offsets[:-1].to(torch.int32).contiguous()
        M_per = (offsets[1:] - offsets[:-1]).to(torch.int32).contiguous()
        max_M = int(M_per.max().item()) if E > 0 else 0

        if TRITON_AVAILABLE and A_packed.is_cuda:
            grad_A = torch.zeros_like(A_packed) if needs_A else None
            if needs_A:
                dA_grid = lambda meta: (  # noqa: E731
                    E,
                    triton.cdiv(max_M, meta["BLOCK_M"]),
                    triton.cdiv(K, meta["BLOCK_N"]),  # kernel-N == original K
                )
                _grouped_gemm_varM_kernel[dA_grid](
                    grad_out,
                    B_cat,
                    grad_A,
                    starts,
                    M_per,
                    K=N,
                    N=K,
                    stride_a_m=grad_out.stride(0),
                    stride_a_k=grad_out.stride(1),
                    stride_b_g=B_cat.stride(0),
                    stride_b_k=B_cat.stride(2),
                    stride_b_n=B_cat.stride(1),
                    stride_c_m=grad_A.stride(0),
                    stride_c_n=grad_A.stride(1),
                    ACC_DTYPE=tl.float32,
                )

            grad_B = torch.zeros_like(B_cat) if needs_B else None
            if needs_B:
                dB_grid = lambda meta: (  # noqa: E731
                    E,
                    triton.cdiv(K, meta["BLOCK_K"]),
                    triton.cdiv(N, meta["BLOCK_N"]),
                )
                _grouped_gemm_bwd_dB_kernel[dB_grid](
                    A_packed,
                    grad_out,
                    grad_B,
                    starts,
                    M_per,
                    K=K,
                    N=N,
                    stride_a_m=A_packed.stride(0),
                    stride_a_k=A_packed.stride(1),
                    stride_dc_m=grad_out.stride(0),
                    stride_dc_n=grad_out.stride(1),
                    stride_db_g=grad_B.stride(0),
                    stride_db_k=grad_B.stride(1),
                    stride_db_n=grad_B.stride(2),
                    ACC_DTYPE=tl.float32,
                )
        else:
            grad_A = torch.zeros_like(A_packed) if needs_A else None
            grad_B = torch.zeros_like(B_cat) if needs_B else None
            for e in range(E):
                s = int(offsets[e].item())
                t = int(offsets[e + 1].item())
                if t <= s:
                    continue
                A_e = A_packed[s:t]
                dC_e = grad_out[s:t]
                B_e = B_cat[e]
                if needs_A:
                    grad_A[s:t] = dC_e @ B_e.T
                if needs_B:
                    grad_B[e] = A_e.T @ dC_e

        return grad_A, None, grad_B, None


# =============================== Triton kernels ===============================
if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64},
                num_stages=3,
                num_warps=8,
            ),
        ],
        key=["K", "N"],
    )
    @triton.jit
    def _grouped_gemm_varM_kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        starts_ptr,
        M_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_a_m,
        stride_a_k,
        stride_b_g,
        stride_b_k,
        stride_b_n,
        stride_c_m,
        stride_c_n,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """One program per (group g, m-tile, n-tile). Computes a C_g tile."""
        g = tl.program_id(0)
        pid_m = tl.program_id(1)
        pid_n = tl.program_id(2)

        start_g = tl.load(starts_ptr + g).to(tl.int32)
        M_g = tl.load(M_ptr + g).to(tl.int32)

        m0 = pid_m * BLOCK_M
        if m0 >= M_g:
            return  # this m-tile is past the end of group g

        n0 = pid_n * BLOCK_N
        b_base = B_ptr + g * stride_b_g
        a_tile_base = A_ptr + (start_g + m0) * stride_a_m
        c_tile_base = C_ptr + (start_g + m0) * stride_c_m

        offs_m = tl.arange(0, BLOCK_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)

        mask_m = offs_m < (M_g - m0)
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_DTYPE)

        k0 = 0
        while k0 < K:
            k_ids = k0 + offs_k
            k_mask = k_ids < K

            a_ptrs = (
                a_tile_base + offs_m[:, None] * stride_a_m + k_ids[None, :] * stride_a_k
            )
            a = tl.load(a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)

            b_ptrs = b_base + k_ids[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
            b = tl.load(b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(a, b)
            k0 += BLOCK_K

        c_ptrs = (
            c_tile_base + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
        )
        tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

    @triton.autotune(
        configs=[
            triton.Config(
                {"BLOCK_K": 64, "BLOCK_N": 64, "BLOCK_M": 32}, num_stages=2, num_warps=4
            ),
            triton.Config(
                {"BLOCK_K": 64, "BLOCK_N": 128, "BLOCK_M": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_K": 128, "BLOCK_N": 64, "BLOCK_M": 32},
                num_stages=2,
                num_warps=4,
            ),
            triton.Config(
                {"BLOCK_K": 128, "BLOCK_N": 128, "BLOCK_M": 32},
                num_stages=3,
                num_warps=8,
            ),
        ],
        key=["K", "N"],
    )
    @triton.jit
    def _grouped_gemm_bwd_dB_kernel(
        A_ptr,
        dC_ptr,
        dB_ptr,
        starts_ptr,
        M_ptr,
        K: tl.constexpr,
        N: tl.constexpr,
        stride_a_m,
        stride_a_k,
        stride_dc_m,
        stride_dc_n,
        stride_db_g,
        stride_db_k,
        stride_db_n,
        BLOCK_K: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """One program per (group g, k-tile, n-tile). Tile of grad_B[g]=A_g.T @ dC_g."""
        g = tl.program_id(0)
        pid_k = tl.program_id(1)
        pid_n = tl.program_id(2)

        start_g = tl.load(starts_ptr + g).to(tl.int32)
        M_g = tl.load(M_ptr + g).to(tl.int32)

        db_base = dB_ptr + g * stride_db_g

        offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_k = offs_k < K
        mask_n = offs_n < N

        acc = tl.zeros((BLOCK_K, BLOCK_N), dtype=ACC_DTYPE)

        m0 = 0
        while m0 < M_g:
            offs_m = tl.arange(0, BLOCK_M)
            mask_m = offs_m < (M_g - m0)
            abs_m = start_g + m0

            a_T_ptrs = (
                A_ptr
                + (abs_m + offs_m[None, :]) * stride_a_m
                + offs_k[:, None] * stride_a_k
            )
            a_T = tl.load(a_T_ptrs, mask=mask_k[:, None] & mask_m[None, :], other=0.0)

            dc_ptrs = (
                dC_ptr
                + (abs_m + offs_m[:, None]) * stride_dc_m
                + offs_n[None, :] * stride_dc_n
            )
            dc = tl.load(dc_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)

            acc += tl.dot(a_T, dc)
            m0 += BLOCK_M

        db_ptrs = (
            db_base + offs_k[:, None] * stride_db_k + offs_n[None, :] * stride_db_n
        )
        tl.store(db_ptrs, acc, mask=mask_k[:, None] & mask_n[None, :])


# ============================== Python helpers ===============================


def _split_by_offsets(x: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
    """Return a list of E views: x[start:end] for each expert."""
    out: list[torch.Tensor] = []
    for e in range(offsets.numel() - 1):
        s = int(offsets[e].item())
        t = int(offsets[e + 1].item())
        out.append(x.narrow(0, s, max(0, t - s)))
    return out


def _foreach_mm(
    A_blocks: list[torch.Tensor],
    B_blocks: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Portable fallback (no Triton): per-group matmul with autograd."""
    Y: list[torch.Tensor] = []
    for A, B in zip(A_blocks, B_blocks):
        if A.numel() == 0:
            Y.append(A.new_zeros((0, B.shape[1])))
        else:
            Y.append(A @ B)
    return Y


# ============================== Grouped GEMM API ==============================

__SHARED_B_WARNED = False


def grouped_mm_varM(
    A_packed: torch.Tensor,
    offsets: torch.Tensor,
    B_per_expert: Sequence[torch.Tensor] | torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_fp16_acc: bool = False,
    use_shared_b: bool = False,
    allow_triton_training: bool = True,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> torch.Tensor:
    """
    Compute concatenated C = concat_e (A_e @ B_e) with variable M_e.

    A_packed: concat_e A_e along M, shape [S, K]
    offsets: starts/ends per expert in A_packed/C_packed, shape [E+1]
    B_per_expert: list of [K, N] or prepacked tensor [E, K, N]
    Returns C_packed: [S, N]
    """
    global __SHARED_B_WARNED

    device = A_packed.device
    dtype = A_packed.dtype
    out_dtype = out_dtype or dtype

    E = offsets.numel() - 1
    if E <= 0:
        return A_packed.new_zeros((0, 0), dtype=out_dtype)

    if not A_packed.is_contiguous():
        A_packed = A_packed.contiguous()
    if not offsets.is_contiguous():
        offsets = offsets.contiguous()

    packed_B_input = isinstance(B_per_expert, torch.Tensor)
    requires_grad = torch.is_grad_enabled() and (
        A_packed.requires_grad
        or (
            B_per_expert.requires_grad
            if packed_B_input
            else any(b.requires_grad for b in B_per_expert)
        )
    )

    if (not TRITON_AVAILABLE) or (not A_packed.is_cuda):
        A_blocks = _split_by_offsets(A_packed, offsets)
        B_blocks = (
            list(B_per_expert.unbind(0)) if packed_B_input else list(B_per_expert)
        )
        Y = _foreach_mm(A_blocks, B_blocks)
        N = B_blocks[0].shape[1] if B_blocks else 0
        C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
        for y, s, t in zip(Y, offsets[:-1], offsets[1:]):
            s_i, t_i = int(s.item()), int(t.item())
            if t_i > s_i and y.numel() > 0:
                C_packed[s_i:t_i] = y.to(out_dtype)
        return C_packed

    if packed_B_input:
        if B_per_expert.ndim != 3:
            raise ValueError("Prepacked B_per_expert must have shape [E, K, N].")
        if B_per_expert.shape[0] != E:
            raise ValueError(
                f"Expected {E} expert matrices from offsets, got {B_per_expert.shape[0]}."
            )
        B_cat = B_per_expert.contiguous()
        if B_cat.dtype != A_packed.dtype:
            B_cat = B_cat.to(A_packed.dtype)
    else:
        K = B_per_expert[0].shape[0]
        N = B_per_expert[0].shape[1]
        B_cat = torch.stack([b.contiguous() for b in B_per_expert], dim=0).to(
            A_packed.dtype
        )

    K = B_cat.shape[1]
    N = B_cat.shape[2]

    if requires_grad and allow_triton_training:
        C = _GroupedMMVarMFunction.apply(A_packed, offsets, B_cat, use_fp16_acc)
        return C.to(out_dtype) if C.dtype != out_dtype else C

    if requires_grad:
        A_blocks = _split_by_offsets(A_packed, offsets)
        B_blocks = (
            list(B_per_expert.unbind(0)) if packed_B_input else list(B_per_expert)
        )
        Y = _foreach_mm(A_blocks, B_blocks)
        C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
        for y, s, t in zip(Y, offsets[:-1], offsets[1:]):
            s_i, t_i = int(s.item()), int(t.item())
            if t_i > s_i and y.numel() > 0:
                C_packed[s_i:t_i] = y.to(out_dtype)
        return C_packed

    if use_shared_b and not __SHARED_B_WARNED:
        print(
            "[grouped_mm_varM] use_shared_b=True is currently ignored; "
            "shared-memory optimization was removed for clarity."
        )
        __SHARED_B_WARNED = True

    C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
    starts = offsets[:-1].to(torch.int32).contiguous()
    M_per = (offsets[1:] - offsets[:-1]).to(torch.int32).contiguous()

    stride_a_m, stride_a_k = A_packed.stride()
    stride_b_g, stride_b_k, stride_b_n = B_cat.stride()
    stride_c_m, stride_c_n = C_packed.stride()

    acc_dtype = tl.float16 if use_fp16_acc else tl.float32

    max_M = int(M_per.max().item()) if E > 0 else 0
    grid = lambda meta: (  # noqa: E731
        E,
        triton.cdiv(max_M, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    _grouped_gemm_varM_kernel[grid](
        A_packed,
        B_cat,
        C_packed,
        starts,
        M_per,
        K=K,
        N=N,
        stride_a_m=stride_a_m,
        stride_a_k=stride_a_k,
        stride_b_g=stride_b_g,
        stride_b_k=stride_b_k,
        stride_b_n=stride_b_n,
        stride_c_m=stride_c_m,
        stride_c_n=stride_c_n,
        ACC_DTYPE=acc_dtype,
    )
    return C_packed


__all__ = [
    "TRITON_AVAILABLE",
    "_GroupedMMVarMFunction",
    "_split_by_offsets",
    "_foreach_mm",
    "grouped_mm_varM",
]
