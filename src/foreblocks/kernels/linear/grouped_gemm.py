"""foreblocks.kernels.linear.grouped_gemm.

Grouped GEMM with variable M per expert — Triton-accelerated matrix multiply.

Supports concatenated MoE expert segments where each expert processes a variable
number of tokens. The Triton forward and backward kernels are autotuned, with a persistent
scheduler for sparse expert loads and an eager PyTorch fallback. Use when building Mixture-of-Experts models or
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
# - Forward and dA share a scheduler; dB reduces within each expert.
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
    @staticmethod
    def forward(ctx, A_packed, offsets, B_cat, use_fp16_acc: bool, backend="auto"):
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

        ctx.backend = backend
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

        # Grid over total packed rows instead of max(M_e): avoids a GPU sync
        # (.item()); out-of-range m-tiles early-return in the kernel.
        S_total = A_packed.shape[0]
        grid = lambda meta: (  # noqa: E731
            E,
            triton.cdiv(S_total, meta["BLOCK_M"]),
            triton.cdiv(N, meta["BLOCK_N"]),
        )
        if _use_persistent(backend, E, S_total):
            _launch_persistent_grouped_mm(
                A_packed, B_cat, C_packed, starts, M_per, use_fp16_acc
            )
        else:
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
        S_total = A_packed.shape[0]

        if TRITON_AVAILABLE and A_packed.is_cuda:
            grad_A = torch.zeros_like(A_packed) if needs_A else None
            if needs_A:
                dA_grid = lambda meta: (  # noqa: E731
                    E,
                    triton.cdiv(S_total, meta["BLOCK_M"]),
                    triton.cdiv(K, meta["BLOCK_N"]),  # kernel-N == original K
                )
                if _use_persistent(ctx.backend, E, S_total):
                    _launch_persistent_grouped_mm(
                        grad_out, B_cat.transpose(-1, -2), grad_A, starts, M_per
                    )
                else:
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

        return (grad_A, None, grad_B, None, None)[: len(ctx.needs_input_grad)]


# =============================== Triton kernels ===============================
if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 32, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8),
        ],
        key=["K", "N", "E", "TOTAL_M"],
    )
    @triton.jit
    def _persistent_grouped_gemm_kernel(
        A,
        B,
        C,
        Starts,
        Sizes,
        K: tl.constexpr,
        N: tl.constexpr,
        E: tl.constexpr,
        TOTAL_M: tl.constexpr,
        SAM: tl.constexpr,
        SAK: tl.constexpr,
        SBG: tl.constexpr,
        SBK: tl.constexpr,
        SBN: tl.constexpr,
        SCM: tl.constexpr,
        SCN: tl.constexpr,
        WORKERS: tl.constexpr,
        FP16_ACC: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        tile = tl.program_id(0)
        prefix = 0
        n_tiles = tl.cdiv(N, BLOCK_N)
        for expert in range(E):
            size = tl.load(Sizes + expert)
            start = tl.load(Starts + expert)
            tiles = tl.cdiv(size, BLOCK_M) * n_tiles
            while tile < prefix + tiles:
                local = tile - prefix
                m = (local // n_tiles) * BLOCK_M + tl.arange(0, BLOCK_M)
                n = (local % n_tiles) * BLOCK_N + tl.arange(0, BLOCK_N)
                ks = tl.arange(0, BLOCK_K)
                acc = tl.zeros(
                    (BLOCK_M, BLOCK_N), tl.float16 if FP16_ACC else tl.float32
                )
                for k0 in range(tl.cdiv(K, BLOCK_K)):
                    kk = k0 * BLOCK_K + ks
                    a = tl.load(
                        A + (start + m[:, None]) * SAM + kk[None, :] * SAK,
                        (m[:, None] < size) & (kk[None, :] < K),
                        other=0,
                    )
                    b = tl.load(
                        B + expert * SBG + kk[:, None] * SBK + n[None, :] * SBN,
                        (kk[:, None] < K) & (n[None, :] < N),
                        other=0,
                    )
                    acc = (acc + tl.dot(a, b, input_precision="tf32x3")).to(
                        tl.float16 if FP16_ACC else tl.float32
                    )
                tl.store(
                    C + (start + m[:, None]) * SCM + n[None, :] * SCN,
                    acc,
                    (m[:, None] < size) & (n[None, :] < N),
                )
                tile += WORKERS
            prefix += tiles

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

            acc += tl.dot(a_T, dc, input_precision="tf32x3")
            m0 += BLOCK_M

        db_ptrs = (
            db_base + offs_k[:, None] * stride_db_k + offs_n[None, :] * stride_db_n
        )
        tl.store(db_ptrs, acc, mask=mask_k[:, None] & mask_n[None, :])


# ============================== Python helpers ===============================


def _split_by_offsets(x: torch.Tensor, offsets: torch.Tensor) -> list[torch.Tensor]:
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
    Y: list[torch.Tensor] = []
    for A, B in zip(A_blocks, B_blocks):
        if A.numel() == 0:
            Y.append(A.new_zeros((0, B.shape[1])))
        else:
            Y.append(A @ B)
    return Y


# ============================== Grouped GEMM API ==============================

__SHARED_B_WARNED = False


def _use_persistent(backend, expert_count, total_rows):
    # The 5090 benchmark favors persistent scheduling for many short groups;
    # large dense groups retain the legacy launch. Either can be forced.
    return backend == "persistent" or (
        backend == "auto" and expert_count >= 16 and total_rows <= expert_count * 128
    )


def _launch_persistent_grouped_mm(A, B, C, starts, sizes, use_fp16_acc=False):
    if A.size(0) == 0 or C.size(1) == 0:
        return
    workers = torch.cuda.get_device_properties(A.device).multi_processor_count
    _persistent_grouped_gemm_kernel[(workers,)](
        A,
        B,
        C,
        starts,
        sizes,
        K=A.size(1),
        N=C.size(1),
        E=B.size(0),
        TOTAL_M=A.size(0),
        SAM=A.stride(0),
        SAK=A.stride(1),
        SBG=B.stride(0),
        SBK=B.stride(1),
        SBN=B.stride(2),
        SCM=C.stride(0),
        SCN=C.stride(1),
        WORKERS=workers,
        FP16_ACC=use_fp16_acc,
    )


def grouped_mm_varM(
    A_packed: torch.Tensor,
    offsets: torch.Tensor,
    B_per_expert: Sequence[torch.Tensor] | torch.Tensor,
    out_dtype: torch.dtype | None = None,
    use_fp16_acc: bool = False,
    use_shared_b: bool = False,
    allow_triton_training: bool = True,
    backend: str = "auto",
) -> torch.Tensor:
    global __SHARED_B_WARNED
    if backend not in {"auto", "persistent", "legacy"}:
        raise ValueError("grouped GEMM backend must be auto, persistent, or legacy")

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
        C = _GroupedMMVarMFunction.apply(
            A_packed, offsets, B_cat, use_fp16_acc, backend
        )
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

    S_total = A_packed.shape[0]
    grid = lambda meta: (  # noqa: E731
        E,
        triton.cdiv(S_total, meta["BLOCK_M"]),
        triton.cdiv(N, meta["BLOCK_N"]),
    )
    if _use_persistent(backend, E, S_total):
        _launch_persistent_grouped_mm(
            A_packed, B_cat, C_packed, starts, M_per, use_fp16_acc
        )
    else:
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
