# grouped_gemm.py  (FIXED)
# -----------------------------------------------------------------------------
# From-scratch grouped GEMM (Triton) + SwiGLU MLP for packed MoE segments.
# This version avoids pointer arrays and uses base pointers + per-group offsets.
# -----------------------------------------------------------------------------

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


# =============================== Triton kernels ===============================

if TRITON_AVAILABLE:

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
        # Base pointer for this group (B only, since A/C bases change per tile)
        b_base = B_ptr + g * stride_b_g
        m0 = 0
        while m0 < M_g:
            # Recompute A/C bases for this M-tile, absolute from global start
            a_tile_base = A_ptr + (start_g + m0) * stride_a_m
            c_tile_base = C_ptr + (start_g + m0) * stride_c_m
            n0 = 0
            while n0 < N:
                offs_m = tl.arange(0, BLOCK_M)
                offs_n = n0 + tl.arange(0, BLOCK_N)
                offs_k = tl.arange(0, BLOCK_K)
                mask_m = offs_m < (M_g - m0)  # Remaining rows in group for this tile
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
                        a_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0
                    )
                    # B chunk: [BLOCK_K, BLOCK_N] from B_g
                    b_ptrs = (
                        b_base
                        + k_ids[:, None] * stride_b_k
                        + offs_n[None, :] * stride_b_n
                    )
                    b = tl.load(
                        b_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0
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
# ============================== Python wrappers ===============================


def _split_by_offsets(x: torch.Tensor, offsets: torch.Tensor) -> List[torch.Tensor]:
    """
    Given x of shape [S, D] and offsets [E+1], return a list of E views: x[start:end].
    Only used in the CPU / fallback path.
    """
    out = []
    for e in range(offsets.numel() - 1):
        s = int(offsets[e].item())
        t = int(offsets[e + 1].item())
        out.append(x.narrow(0, s, max(0, t - s)))
    return out


def _foreach_mm(
    A_blocks: List[torch.Tensor], B_blocks: List[torch.Tensor]
) -> List[torch.Tensor]:
    """Portable fallback (no Triton): per-group matmul."""
    Y = []
    for A, B in zip(A_blocks, B_blocks):
        if A.numel() == 0:
            Y.append(A.new_zeros((0, B.shape[1])))
        else:
            Y.append(A @ B)
    return Y


def grouped_mm_varM(
    A_packed: torch.Tensor,  # [S, K]
    offsets: torch.Tensor,  # [E+1]
    B_per_expert: Sequence[torch.Tensor],  # each [K, N], same K,N
    out_dtype: Optional[torch.dtype] = None,
    block_m: int = 128,
    block_n: int = 128,
    block_k: int = 64,
) -> torch.Tensor:
    """
    Compute concatenated C = concat_e (A_e @ B_e) with variable M_e.
    A_packed: concat_e A_e along M
    offsets:  starts/ends per expert in A_packed/C_packed
    B_per_expert: list of [K, N]
    Returns C_packed: [S, N]
    """
    device = A_packed.device
    dtype = A_packed.dtype
    out_dtype = out_dtype or dtype
    E = offsets.numel() - 1
    if E <= 0:
        return A_packed.new_zeros((0, 0))

    # Short-circuit: CPU or no Triton
    if (not TRITON_AVAILABLE) or (not A_packed.is_cuda):
        A_blocks = _split_by_offsets(A_packed, offsets)
        Y = _foreach_mm(A_blocks, list(B_per_expert))
        N = B_per_expert[0].shape[1] if len(B_per_expert) else 0
        C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)
        for y, s, t in zip(Y, offsets[:-1], offsets[1:]):
            s_i, t_i = int(s.item()), int(t.item())
            if t_i > s_i and y.numel() > 0:
                C_packed[s_i:t_i] = y.to(out_dtype)
        return C_packed

    # Triton path -------------------------------------------------------------
    # Ensure all B_e share K,N, dtype, device; stack as [E,K,N]
    K = B_per_expert[0].shape[0]
    N = B_per_expert[0].shape[1]
    B_cat = torch.stack([b.contiguous() for b in B_per_expert], dim=0).to(
        A_packed.dtype
    )  # [E,K,N]

    # Allocate output and build starts / M vectors
    C_packed = torch.empty((A_packed.shape[0], N), device=device, dtype=out_dtype)

    starts = offsets[:-1].to(torch.int32).contiguous()  # [E]
    M_per = (offsets[1:] - offsets[:-1]).to(torch.int32).contiguous()  # [E]

    # Strides (row-major assumptions)
    # A_packed: [S,K]
    stride_a_m, stride_a_k = A_packed.stride()
    # B_cat: [E,K,N]
    stride_b_g, stride_b_k, stride_b_n = B_cat.stride()
    # C_packed: [S,N]
    stride_c_m, stride_c_n = C_packed.stride()

    # Accumulation dtype
    acc_dtype = tl.float32

    # Launch: one program per group (expert)
    grid = (E,)

    _grouped_gemm_varM_kernel[grid](
        A_packed,
        B_cat,
        C_packed,
        starts,
        M_per,
        K,
        N,
        stride_a_m,
        stride_a_k,
        stride_b_g,
        stride_b_k,
        stride_b_n,
        stride_c_m,
        stride_c_n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
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
    w12_list, w3_list = [], []
    H = None
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
    return w12_list, w3_list, H


def grouped_mlp_swiglu(
    packed_x: torch.Tensor,  # [S, D]
    offsets: torch.Tensor,  # [E+1]
    experts: Sequence[torch.nn.Module],
    dropout_p: float = 0.0,
    training: bool = False,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    SwiGLU MLP in grouped mode over packed slices:
      GU = grouped_mm( packed_x, W12[e] )  # D -> 2H
      H  = SiLU(G) * U
      Y  = grouped_mm( H, W3[e] )          # H -> D
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
        block_m=128,
        block_n=128,
        block_k=64,
    )
    return Y
