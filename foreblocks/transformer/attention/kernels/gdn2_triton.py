"""
gdn2_triton.py — Triton chunk-parallel FORWARD kernel for Gated DeltaNet-2.

This implements a fast forward-only kernel that matches, bit-for-bit (within
bf16 tolerance), the pure-PyTorch oracle ``GatedDeltaNet2._chunk_parallel``.

The math is the exact WY chunk form from the oracle (read that method as the
spec). Per head, state ``S ∈ R^{dk×dv}``:

    e_r = b_r ⊙ k_r              # gated erase direction
    z_r = w_r ⊙ v_r              # gated write target
    γ_r = ∏_{i≤r} α_i            # cumulative within-chunk decay (per dk channel)
    K̄ = K · γ^{-1},  Ē = e · γ,  Q̄ = Q · γ
    A = (I + tril(Ē K̄ᵀ, -1))^{-1}
    U = A · (z − Ē · S0)
    O = Q̄ · S0 + tril(Q̄ K̄ᵀ, 0) · U
    S = Diag(γ_L) · S0 + (γ_L/γ_j ⊙ k_j)ᵀ · U

One Triton program handles one (batch*head) row and loops sequentially over
chunks, carrying the fp32 state ``S`` in registers/SRAM. The triangular inverse
is computed in-kernel via the FLA-style row-recurrence forward substitution.

Limitations
-----------
* Forward only (no autograd). The PyTorch ``_chunk_parallel`` remains the
  differentiable fallback.
* ``dk`` and ``dv`` must be powers of two in the supported set and ``≤ 128``.
* Designed/tested for ``C`` in {16, 32, 64}.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    _HAS_TRITON = False


_SUPPORTED_DK = (16, 32, 64, 128)
_SUPPORTED_DV = (16, 32, 64, 128)


if _HAS_TRITON:

    @triton.jit
    def _gdn2_chunk_fwd_kernel(
        Q, K, V, B, W, ALPHA,          # inputs [BH, T, d]
        S0,                            # initial state [BH, dk, dv]
        OUT,                           # output [BH, T, dv]
        SF,                            # final state [BH, dk, dv]
        T,                             # seq length (runtime int)
        s_qt, s_qd,                    # strides for [BH,T,dk] tensors (token, dim)
        s_vt, s_vd,                    # strides for [BH,T,dv] tensors
        s_sk, s_sv,                    # strides for state [BH,dk,dv]
        DK: tl.constexpr,
        DV: tl.constexpr,
        C: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        i_bh = tl.program_id(0)

        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, DV)

        # strictly-lower / lower-incl / identity masks over the CxC chunk
        m_strict = offs_c[:, None] > offs_c[None, :]   # r > j
        m_tril = offs_c[:, None] >= offs_c[None, :]     # r >= j
        m_eye = offs_c[:, None] == offs_c[None, :]

        # base pointers for this (batch*head) row
        q_base = Q + i_bh * T * s_qt
        k_base = K + i_bh * T * s_qt
        v_base = V + i_bh * T * s_vt
        b_base = B + i_bh * T * s_qt
        w_base = W + i_bh * T * s_vt
        a_base = ALPHA + i_bh * T * s_qt

        s_base = S0 + i_bh * DK * s_sk
        sf_base = SF + i_bh * DK * s_sk
        o_base = OUT + i_bh * T * s_vt

        # load incoming state S0 -> [DK, DV] fp32
        s_ptr = s_base + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        b_S = tl.load(s_ptr).to(tl.float32)

        n_chunks = tl.cdiv(T, C)
        for i_chunk in range(0, n_chunks):
            start = i_chunk * C
            rows = start + offs_c
            row_mask = rows < T                              # [C]

            # ---- load chunk tiles [C, d] (masked rows -> 0) ----
            qk_mask = row_mask[:, None] & (offs_k[None, :] < DK)
            v_mask = row_mask[:, None] & (offs_v[None, :] < DV)

            q_ptr = q_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            k_ptr = k_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            v_ptr = v_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            b_ptr = b_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            w_ptr = w_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            a_ptr = a_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd

            b_Qc = tl.load(q_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Kc = tl.load(k_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Vc = tl.load(v_ptr, mask=v_mask, other=0.0).to(tl.float32)
            b_bc = tl.load(b_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_wc = tl.load(w_ptr, mask=v_mask, other=0.0).to(tl.float32)
            # decay alpha: masked rows -> 1.0 so log is 0 (no effect)
            b_ac = tl.load(a_ptr, mask=qk_mask, other=1.0).to(tl.float32)

            # ---- cumulative within-chunk decay gamma (inclusive) ----
            log_alpha = tl.log(tl.maximum(b_ac, 1e-12))          # [C, DK]
            log_gamma = tl.cumsum(log_alpha, axis=0)             # inclusive cumsum
            b_gamma = tl.exp(log_gamma)                          # [C, DK]
            b_inv_gamma = tl.exp(-log_gamma)                     # γ^{-1}

            # gated erase / write directions
            b_edir = b_bc * b_Kc                                 # e = b ⊙ k  [C, DK]
            b_ztgt = b_wc * b_Vc                                 # z = w ⊙ v  [C, DV]

            # decay-normalised vectors
            b_Kbar = b_Kc * b_inv_gamma                          # [C, DK]
            b_Ebar = b_edir * b_gamma                            # [C, DK]
            b_Qbar = b_Qc * b_gamma                              # [C, DK]

            # ---- T = tril(Ē K̄ᵀ, -1), invert (I + T) ----
            b_EK = tl.dot(b_Ebar, tl.trans(b_Kbar), allow_tf32=ALLOW_TF32)              # [C, C]
            b_EK = tl.where(m_strict, b_EK, 0.0)

            # forward-substitution inverse of unit lower-tri (I + T):
            #   Ainv = I - T - sum_{j<i} T[i,j] Ainv[j, :]
            # FLA-style row recurrence, accumulating into b_Ainv.
            b_Ainv = tl.where(m_strict, -b_EK, 0.0)              # start: -T (strict)
            for i in range(1, C):
                # b_a = -T[i,:] (cols < i) + sum_k (-T[i,k]) * Ainv[k,:]
                negT_i = tl.sum(tl.where(offs_c[:, None] == i, -b_EK, 0.0), 0)  # [C]
                negT_i = tl.where(offs_c < i, negT_i, 0.0)
                corr = tl.sum(negT_i[:, None] * b_Ainv, 0)       # [C]
                new_row = negT_i + corr
                b_Ainv = tl.where((offs_c == i)[:, None], new_row[None, :], b_Ainv)
            b_Ainv = b_Ainv + tl.where(m_eye, 1.0, 0.0)          # add identity

            # ---- U = Ainv @ (z - Ē·S0) ----
            b_ES0 = tl.dot(b_Ebar, b_S, allow_tf32=ALLOW_TF32)                          # [C, DV]
            b_rhs = b_ztgt - b_ES0
            b_U = tl.dot(b_Ainv, b_rhs, allow_tf32=ALLOW_TF32)                          # [C, DV]

            # ---- O = Q̄·S0 + tril(Q̄ K̄ᵀ) @ U ----
            b_O = tl.dot(b_Qbar, b_S, allow_tf32=ALLOW_TF32)                            # inter-chunk
            b_QK = tl.dot(b_Qbar, tl.trans(b_Kbar), allow_tf32=ALLOW_TF32)             # [C, C]
            b_QK = tl.where(m_tril, b_QK, 0.0)
            b_O += tl.dot(b_QK, b_U, allow_tf32=ALLOW_TF32)                             # intra-chunk

            # store output (masked rows skipped)
            o_ptr = o_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            tl.store(o_ptr, b_O.to(OUT.dtype.element_ty), mask=v_mask)

            # ---- state carry: S = Diag(γ_L)·S0 + (γ_L/γ_j ⊙ k_j)ᵀ U ----
            # γ_L = gamma at last valid row of this chunk.
            valid = tl.where(row_mask, rows, -1)
            last_idx = tl.max(valid, 0)                          # last valid row index
            last_local = last_idx - start                        # within-chunk
            sel = offs_c == last_local
            gamma_end = tl.sum(tl.where(sel[:, None], b_gamma, 0.0), 0)  # [DK]

            b_Ktail = b_Kc * (gamma_end[None, :] * b_inv_gamma)  # [C, DK]
            b_KtU = tl.dot(tl.trans(b_Ktail), b_U, allow_tf32=ALLOW_TF32)               # [DK, DV]
            b_S = gamma_end[:, None] * b_S + b_KtU

        # store final state
        sf_ptr = sf_base + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        tl.store(sf_ptr, b_S.to(SF.dtype.element_ty))


def can_use_gdn2_triton(
    S0: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    C: int,
) -> bool:
    """Guard for the Triton GDN-2 forward path.

    Requires CUDA, fp16/bf16 inputs, supported dk/dv, and C in {16,32,64}.
    Ragged T (T % C != 0) is handled by row masking, so it is allowed.
    """
    import os

    if not _HAS_TRITON:
        return False
    if os.environ.get("GDN2_DISABLE_TRITON", "") == "1":
        return False
    if not (Q.is_cuda and K.is_cuda and V.is_cuda and S0.is_cuda):
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16):
        return False
    dk = Q.shape[-1]
    dv = V.shape[-1]
    if dk not in _SUPPORTED_DK or dv not in _SUPPORTED_DV:
        return False
    if C not in (16, 32, 64):
        return False
    return True


def gdn2_chunk_fwd_triton(
    S0: torch.Tensor,    # [BH, dk, dv]
    Q: torch.Tensor,     # [BH, T, dk]  (L2-normalised query)
    K: torch.Tensor,     # [BH, T, dk]  (L2-normalised key)
    V: torch.Tensor,     # [BH, T, dv]
    b: torch.Tensor,     # [BH, T, dk, 1] erase gate
    w: torch.Tensor,     # [BH, T, dv, 1] write gate
    alpha: torch.Tensor,  # [BH, T, dk, 1] channel-wise decay
    C: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Triton forward of GDN-2 chunk-parallel recurrence.

    Returns (out [BH, T, dv], S_final [BH, dk, dv]) matching the semantics of
    ``GatedDeltaNet2._chunk_parallel``.
    """
    BH, T, dk = Q.shape
    dv = V.shape[-1]

    Qc = Q.contiguous()
    Kc = K.contiguous()
    Vc = V.contiguous()
    bc = b.squeeze(-1).contiguous()        # [BH, T, dk]
    wc = w.squeeze(-1).contiguous()        # [BH, T, dv]
    ac = alpha.squeeze(-1).contiguous()    # [BH, T, dk]
    S0c = S0.contiguous()

    out = torch.empty(BH, T, dv, device=Q.device, dtype=Q.dtype)
    S_final = torch.empty(BH, dk, dv, device=Q.device, dtype=S0.dtype)

    grid = (BH,)
    _gdn2_chunk_fwd_kernel[grid](
        Qc, Kc, Vc, bc, wc, ac,
        S0c, out, S_final,
        T,
        Qc.stride(1), Qc.stride(2),
        Vc.stride(1), Vc.stride(2),
        S0c.stride(1), S0c.stride(2),
        DK=dk, DV=dv, C=C,
        ALLOW_TF32=(Q.dtype != torch.float32),
        num_warps=2,
    )
    return out, S_final
