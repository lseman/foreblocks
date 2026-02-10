# grouped_gemm.py / tf/kernels.py (FIXED AUTOTUNE META CONFLICT)
# -----------------------------------------------------------------------------
# From-scratch grouped GEMM (Triton) + SwiGLU MLP for packed MoE segments.
#
# - Triton kernel is autotuned over BLOCK_M/BLOCK_N/BLOCK_K.
# - DO NOT pass BLOCK_* at launch; autotune provides them.
# - Triton path is inference-only; if grads are needed we use torch.mm.
# -----------------------------------------------------------------------------

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    TRITON_AVAILABLE = False

# =============================== Triton kernels ===============================
if TRITON_AVAILABLE:

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=2, num_warps=4),
            triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64}, num_stages=3, num_warps=8),
        ],
        key=["K", "N"],
    )
    @triton.jit
    def _grouped_gemm_varM_kernel(
        A_ptr,  # *elem (points to A_packed [S,K])
        B_ptr,  # *elem (points to B_cat [E,K,N])
        C_ptr,  # *elem (points to C_packed [S,N])
        starts_ptr,  # *int32 (row start per group in A/C)
        M_ptr,  # *int32 (rows per group)
        # Shared sizes (K,N) and B strides for [E,K,N]
        K: tl.constexpr,
        N: tl.constexpr,
        stride_a_m,
        stride_a_k,
        stride_b_g,
        stride_b_k,
        stride_b_n,  # B is 3D: [g, k, n]
        stride_c_m,
        stride_c_n,
        # Tiling params (autotuned) — PROVIDED BY AUTOTUNE CONFIGS
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        ACC_DTYPE: tl.constexpr,
    ):
        """
        One program per group g.

        A_g: [M_g,K] is located at rows [start_g : start_g+M_g) of A_packed
        B_g: [K,N] is slice B_cat[g, :, :]
        C_g: [M_g,N] is written at rows [start_g : start_g+M_g) of C_packed
        """

        g = tl.program_id(0)
        start_g = tl.load(starts_ptr + g).to(tl.int32)
        M_g = tl.load(M_ptr + g).to(tl.int32)

        # Base pointer for B (for this group)
        b_base = B_ptr + g * stride_b_g

        m0 = 0
        while m0 < M_g:
            a_tile_base = A_ptr + (start_g + m0) * stride_a_m
            c_tile_base = C_ptr + (start_g + m0) * stride_c_m

            n0 = 0
            while n0 < N:
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

                    # A chunk: [BLOCK_M, BLOCK_K]
                    a_ptrs = (
                        a_tile_base
                        + offs_m[:, None] * stride_a_m
                        + k_ids[None, :] * stride_a_k
                    )
                    a = tl.load(
                        a_ptrs,
                        mask=mask_m[:, None] & k_mask[None, :],
                        other=0.0,
                    )

                    # B chunk: [BLOCK_K, BLOCK_N] from B_g
                    b_ptrs = (
                        b_base
                        + k_ids[:, None] * stride_b_k
                        + offs_n[None, :] * stride_b_n
                    )
                    b = tl.load(
                        b_ptrs,
                        mask=k_mask[:, None] & mask_n[None, :],
                        other=0.0,
                    )

                    acc += tl.dot(a, b)
                    k0 += BLOCK_K

                # Write C_g
                c_ptrs = (
                    c_tile_base
                    + offs_m[:, None] * stride_c_m
                    + offs_n[None, :] * stride_c_n
                )
                tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])

                n0 += BLOCK_N
            m0 += BLOCK_M

# ============================== Python helpers ===============================


def _split_by_offsets(x: torch.Tensor, offsets: torch.Tensor) -> List[torch.Tensor]:
    """
    Given x of shape [S, D] and offsets [E+1], return a list of E views: x[start:end].
    Only used in the CPU / fallback path.
    """
    out: List[torch.Tensor] = []
    for e in range(offsets.numel() - 1):
        s = int(offsets[e].item())
        t = int(offsets[e + 1].item())
        out.append(x.narrow(0, s, max(0, t - s)))
    return out


def _foreach_mm(
    A_blocks: List[torch.Tensor],
    B_blocks: List[torch.Tensor],
) -> List[torch.Tensor]:
    """Portable fallback (no Triton): per-group matmul with autograd."""
    Y: List[torch.Tensor] = []
    for A, B in zip(A_blocks, B_blocks):
        if A.numel() == 0:
            Y.append(A.new_zeros((0, B.shape[1])))
        else:
            Y.append(A @ B)
    return Y


# ============================== Grouped GEMM API ==============================

__SHARED_B_WARNED = False  # shared-mem flag is a no-op for now


def grouped_mm_varM(
    A_packed: torch.Tensor,  # [S, K]
    offsets: torch.Tensor,  # [E+1]
    B_per_expert: Sequence[torch.Tensor],  # each [K, N], same K,N
    out_dtype: Optional[torch.dtype] = None,
    use_fp16_acc: bool = False,  # Optional tuning toggle for Triton path
    use_shared_b: bool = False,  # Ignored for now; kept for API compat
    block_m: int = 128,          # NOTE: ignored on Triton path (autotune chooses)
    block_n: int = 128,          # NOTE: ignored on Triton path
    block_k: int = 64,           # NOTE: ignored on Triton path
) -> torch.Tensor:
    """
    Compute concatenated C = concat_e (A_e @ B_e) with variable M_e.

    A_packed: concat_e A_e along M, shape [S, K]
    offsets: starts/ends per expert in A_packed/C_packed, shape [E+1]
    B_per_expert: list of [K, N] (one per expert, same K and N)
    Returns C_packed: [S, N]

    - Triton path is *inference-only* (no gradients).
    - Training / autograd path uses pure torch.mm.
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

    # If we need gradients, stay with PyTorch matmul for correctness.
    requires_grad = (
        torch.is_grad_enabled()
        and (A_packed.requires_grad or any(b.requires_grad for b in B_per_expert))
    )

    if (not TRITON_AVAILABLE) or (not A_packed.is_cuda) or requires_grad:
        A_blocks = _split_by_offsets(A_packed, offsets)
        Y = _foreach_mm(A_blocks, list(B_per_expert))
        N = B_per_expert[0].shape[1] if len(B_per_expert) else 0
        C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
        for y, s, t in zip(Y, offsets[:-1], offsets[1:]):
            s_i, t_i = int(s.item()), int(t.item())
            if t_i > s_i and y.numel() > 0:
                C_packed[s_i:t_i] = y.to(out_dtype)
        return C_packed

    # Triton path (inference-only) -------------------------------------------
    K = B_per_expert[0].shape[0]
    N = B_per_expert[0].shape[1]
    B_cat = torch.stack([b.contiguous() for b in B_per_expert], dim=0).to(A_packed.dtype)  # [E,K,N]

    if use_shared_b and not __SHARED_B_WARNED:
        print(
            "[grouped_mm_varM] use_shared_b=True is currently ignored; "
            "shared-memory optimization was removed for clarity. "
            "If you need it, implement it in a dedicated kernel."
        )
        __SHARED_B_WARNED = True

    C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
    starts = offsets[:-1].to(torch.int32).contiguous()  # [E]
    M_per = (offsets[1:] - offsets[:-1]).to(torch.int32).contiguous()  # [E]

    stride_a_m, stride_a_k = A_packed.stride()
    stride_b_g, stride_b_k, stride_b_n = B_cat.stride()
    stride_c_m, stride_c_n = C_packed.stride()

    acc_dtype = tl.float16 if use_fp16_acc else tl.float32

    grid = (E,)

    # IMPORTANT: DO NOT pass BLOCK_M/N/K here; autotune sets them.
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


# ============================ SwiGLU grouped MLP ==============================

@torch.no_grad()
def _weights_from_swiglu_experts(
    experts: Sequence[torch.nn.Module],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
    """
    Extract per-expert weights for SwiGLU experts.
    Returns ([w12_e], [w3_e], H), with shapes [D,2H] and [H,D].
    """
    w12_list: List[torch.Tensor] = []
    w3_list: List[torch.Tensor] = []
    H: Optional[int] = None

    for e in experts:
        if hasattr(e, "w12") and hasattr(e, "w3"):
            w12, w3 = e.w12.weight.t().contiguous(), e.w3.weight.t().contiguous()
        elif hasattr(e, "gate_up_proj") and hasattr(e, "down_proj"):
            w12, w3 = (
                e.gate_up_proj.weight.t().contiguous(),
                e.down_proj.weight.t().contiguous(),
            )
        else:
            raise ValueError("Unsupported expert layout for SwiGLU extraction.")

        if H is None:
            H = w3.shape[0]
        w12_list.append(w12)
        w3_list.append(w3)

    assert H is not None
    return w12_list, w3_list, H


def grouped_mlp_swiglu(
    packed_x: torch.Tensor,  # [S, D]
    offsets: torch.Tensor,  # [E+1]
    experts: Sequence[torch.nn.Module],
    dropout_p: float = 0.0,
    training: bool = False,
    out_dtype: Optional[torch.dtype] = None,
    use_fp16_acc: bool = False,  # Passed through for optional tuning
    use_shared_b: bool = False,  # Currently ignored in Triton path (see grouped_mm_varM)
) -> torch.Tensor:
    """
    SwiGLU MLP in grouped mode over packed slices:

        GU = grouped_mm_varM(packed_x, W12[e])  # D -> 2H
        H  = SiLU(G) * U
        Y  = grouped_mm_varM(H, W3[e])         # H -> D

    Returns Y packed in the same order ([S, D]).
    """
    if offsets.numel() <= 1:
        return packed_x.new_zeros((0, packed_x.shape[1]))

    w12_list, w3_list, H = _weights_from_swiglu_experts(experts)
    D = packed_x.shape[1]

    assert all(w.shape == (D, 2 * H) for w in w12_list), "All w12 must be [D, 2H]"
    assert all(w.shape == (H, D) for w in w3_list), "All w3 must be [H, D]"

    # Phase A: up-projection (D -> 2H)
    GU = grouped_mm_varM(
        A_packed=packed_x,
        offsets=offsets,
        B_per_expert=w12_list,  # [D, 2H]
        out_dtype=packed_x.dtype,
        use_fp16_acc=use_fp16_acc,
        use_shared_b=use_shared_b,
        # block_* args kept for API compat but ignored on Triton path
        block_m=128,
        block_n=128,
        block_k=64,
    )

    # SwiGLU
    G, U = GU.split(H, dim=-1)
    H_act = F.silu(G) * U
    if training and dropout_p > 0:
        H_act = F.dropout(H_act, p=dropout_p, training=True)

    # Phase B: down-projection (H -> D)
    Y = grouped_mm_varM(
        A_packed=H_act,
        offsets=offsets,
        B_per_expert=w3_list,  # [H, D]
        out_dtype=out_dtype or packed_x.dtype,
        use_fp16_acc=use_fp16_acc,
        use_shared_b=use_shared_b,
        block_m=128,
        block_n=128,
        block_k=64,
    )
    return Y

