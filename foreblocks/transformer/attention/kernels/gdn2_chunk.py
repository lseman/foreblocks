"""
gdn2_chunk.py — Differentiable Triton chunk-parallel FORWARD + BACKWARD kernel
for Gated DeltaNet-2 (GDN-2), adapted from FLA's KDA chunked delta-rule by
generalizing the scalar write-strength ``beta`` to two independent VECTOR gates:

  * erase gate  ``b`` ∈ R^dk   (replaces ``beta`` on the *erase* key)
  * write gate  ``w`` ∈ R^dv   (gates the *value* on the write side)

GDN-2 recurrence (per head, state S ∈ R^{dk×dv}, vector decay α ∈ R^dk):

    e_t = b_t ⊙ k_t                          # gated erase direction
    z_t = w_t ⊙ v_t                          # gated write target
    S_t = Diag(α_t) S_{t-1}
          − k_t ( e_t^T Diag(α_t) S_{t-1} )   # erase (write key is PLAIN k)
          + k_t z_t^T                          # write
    o_t = S_t^T q_t

With ``b = w = beta·1`` this reduces to FLA KDA (verified to bf16 tolerance).

Design
------
Single-program-per-(batch·head) chunked-scan formulation — the same WY math as
the oracle ``GatedDeltaNet2._chunk_parallel`` and the forward-only
``gdn2_triton.py``.  One Triton program owns one BH row, carries the fp32 state
S in registers, and loops sequentially over chunks.  Inside each chunk the work
is a handful of [C×C]/[C×d] GEMMs (the WY representation), so the arithmetic is
chunk-parallel even though the chunk *scan* is sequential.  Parallelism across
BH is exposed via the launch grid.

The backward is a second single-program-per-BH kernel that walks chunks in
reverse, recomputes the per-chunk WY quantities (A = (I+tril(Ē K̄ᵀ,-1))⁻¹, U)
from the saved inputs and the checkpointed per-chunk incoming states, then
applies reverse-mode rules derived (and unit-tested in fp64) against
``torch.autograd`` through ``_chunk_parallel``.

Layout convention
------------------
All tensors use the **[BH, T, *]** layout (BH = batch·heads), matching the
oracle ``_chunk_parallel`` and the rest of ``gated_deltanet2.py``.  ``g`` is the
log-decay (log α).  The public entry ``chunk_gdn2`` accepts/returns this layout.
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
    def _gdot(a, b, allow_tf32: tl.constexpr):
        return tl.dot(a, b, allow_tf32=allow_tf32)

    # ────────────────────────────────────────────────────────────────────────
    # FORWARD kernel: one program per BH, sequential chunk scan.
    # Emits output O, final state SF, and per-chunk incoming-state checkpoints
    # SCKPT (consumed by the backward pass).
    # ────────────────────────────────────────────────────────────────────────
    @triton.jit
    def _gdn2_chunk_fwd_kernel(
        Q, K, V, Bg, Wg, G,           # inputs [BH, T, d]  (G = log alpha)
        S0,                            # initial state [BH, dk, dv]
        OUT,                           # output [BH, T, dv]
        SF,                            # final state [BH, dk, dv]
        SCKPT,                         # per-chunk incoming state [BH, NCHUNK, dk, dv]
        T,
        s_qt, s_qd,                    # strides for [BH,T,dk]-style tensors
        s_vt, s_vd,                    # strides for [BH,T,dv]-style tensors
        s_sk, s_sv,                    # strides for state [BH,dk,dv]
        s_cn, s_ck, s_cv,              # strides for checkpoint [BH,NCHUNK,dk,dv]
        DK: tl.constexpr,
        DV: tl.constexpr,
        C: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        i_bh = tl.program_id(0)

        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, DV)

        m_strict = offs_c[:, None] > offs_c[None, :]
        m_tril = offs_c[:, None] >= offs_c[None, :]
        m_eye = offs_c[:, None] == offs_c[None, :]

        q_base = Q + i_bh * T * s_qt
        k_base = K + i_bh * T * s_qt
        v_base = V + i_bh * T * s_vt
        b_base = Bg + i_bh * T * s_qt
        w_base = Wg + i_bh * T * s_vt
        g_base = G + i_bh * T * s_qt

        s_base = S0 + i_bh * DK * s_sk
        sf_base = SF + i_bh * DK * s_sk
        o_base = OUT + i_bh * T * s_vt

        n_chunks = tl.cdiv(T, C)
        ck_base = SCKPT + i_bh * n_chunks * s_cn

        s_ptr = s_base + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        b_S = tl.load(s_ptr).to(tl.float32)

        for i_chunk in range(0, n_chunks):
            start = i_chunk * C
            rows = start + offs_c
            row_mask = rows < T

            qk_mask = row_mask[:, None] & (offs_k[None, :] < DK)
            v_mask = row_mask[:, None] & (offs_v[None, :] < DV)

            q_ptr = q_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            k_ptr = k_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            v_ptr = v_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            bg_ptr = b_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            wg_ptr = w_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            g_ptr = g_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd

            b_Qc = tl.load(q_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Kc = tl.load(k_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Vc = tl.load(v_ptr, mask=v_mask, other=0.0).to(tl.float32)
            b_bc = tl.load(bg_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_wc = tl.load(wg_ptr, mask=v_mask, other=0.0).to(tl.float32)
            # g is log-alpha; masked rows -> 0 (no decay contribution)
            b_gc = tl.load(g_ptr, mask=qk_mask, other=0.0).to(tl.float32)

            # checkpoint the incoming state for this chunk (fp32)
            ck_ptr = (ck_base + i_chunk * s_cn
                      + offs_k[:, None] * s_ck + offs_v[None, :] * s_cv)
            tl.store(ck_ptr, b_S)

            log_gamma = tl.cumsum(b_gc, axis=0)
            b_gamma = tl.exp(log_gamma)
            b_inv_gamma = tl.exp(-log_gamma)

            b_edir = b_bc * b_Kc
            b_ztgt = b_wc * b_Vc
            b_Kbar = b_Kc * b_inv_gamma
            b_Ebar = b_edir * b_gamma
            b_Qbar = b_Qc * b_gamma

            b_EK = _gdot(b_Ebar, tl.trans(b_Kbar), ALLOW_TF32)
            b_EK = tl.where(m_strict, b_EK, 0.0)

            # forward-substitution inverse of unit lower-tri (I + T)
            b_Ainv = tl.where(m_strict, -b_EK, 0.0)
            for i in range(1, C):
                negT_i = tl.sum(tl.where(offs_c[:, None] == i, -b_EK, 0.0), 0)
                negT_i = tl.where(offs_c < i, negT_i, 0.0)
                corr = tl.sum(negT_i[:, None] * b_Ainv, 0)
                new_row = negT_i + corr
                b_Ainv = tl.where((offs_c == i)[:, None], new_row[None, :], b_Ainv)
            b_Ainv = b_Ainv + tl.where(m_eye, 1.0, 0.0)

            b_ES0 = _gdot(b_Ebar, b_S, ALLOW_TF32)
            b_rhs = b_ztgt - b_ES0
            b_U = _gdot(b_Ainv, b_rhs, ALLOW_TF32)

            b_O = _gdot(b_Qbar, b_S, ALLOW_TF32)
            b_QK = _gdot(b_Qbar, tl.trans(b_Kbar), ALLOW_TF32)
            b_QK = tl.where(m_tril, b_QK, 0.0)
            b_O += _gdot(b_QK, b_U, ALLOW_TF32)

            o_ptr = o_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            tl.store(o_ptr, b_O.to(OUT.dtype.element_ty), mask=v_mask)

            valid = tl.where(row_mask, rows, -1)
            last_idx = tl.max(valid, 0)
            last_local = last_idx - start
            sel = offs_c == last_local
            gamma_end = tl.sum(tl.where(sel[:, None], b_gamma, 0.0), 0)  # [DK]

            b_Ktail = b_Kc * (gamma_end[None, :] * b_inv_gamma)
            b_KtU = _gdot(tl.trans(b_Ktail), b_U, ALLOW_TF32)
            b_S = gamma_end[:, None] * b_S + b_KtU

        sf_ptr = sf_base + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        tl.store(sf_ptr, b_S.to(SF.dtype.element_ty))

    # ────────────────────────────────────────────────────────────────────────
    # BACKWARD kernel: one program per BH, reverse chunk scan.
    # ────────────────────────────────────────────────────────────────────────
    @triton.jit
    def _gdn2_chunk_bwd_kernel(
        Q, K, V, Bg, Wg, G,            # inputs [BH, T, d]
        SCKPT,                         # per-chunk incoming state [BH, NCHUNK, dk, dv]
        DO,                            # grad of output [BH, T, dv]
        DSF,                           # grad of final state [BH, dk, dv]
        DQ, DKt, DVt, DBt, DWt, DGt,   # output grads [BH, T, d]
        DS0,                           # grad wrt initial state [BH, dk, dv]
        T,
        s_qt, s_qd,
        s_vt, s_vd,
        s_sk, s_sv,
        s_cn, s_ck, s_cv,
        DK: tl.constexpr,
        DV: tl.constexpr,
        C: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
    ):
        i_bh = tl.program_id(0)

        offs_c = tl.arange(0, C)
        offs_k = tl.arange(0, DK)
        offs_v = tl.arange(0, DV)

        m_strict = offs_c[:, None] > offs_c[None, :]
        m_tril = offs_c[:, None] >= offs_c[None, :]
        m_eye = offs_c[:, None] == offs_c[None, :]

        q_base = Q + i_bh * T * s_qt
        k_base = K + i_bh * T * s_qt
        v_base = V + i_bh * T * s_vt
        b_base = Bg + i_bh * T * s_qt
        w_base = Wg + i_bh * T * s_vt
        g_base = G + i_bh * T * s_qt

        n_chunks = tl.cdiv(T, C)
        ck_base = SCKPT + i_bh * n_chunks * s_cn

        dq_base = DQ + i_bh * T * s_qt
        dk_base = DKt + i_bh * T * s_qt
        dv_base = DVt + i_bh * T * s_vt
        db_base = DBt + i_bh * T * s_qt
        dw_base = DWt + i_bh * T * s_vt
        dg_base = DGt + i_bh * T * s_qt

        dsf_ptr = DSF + i_bh * DK * s_sk + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        b_dS = tl.load(dsf_ptr).to(tl.float32)

        for ci in range(0, n_chunks):
            i_chunk = n_chunks - 1 - ci
            start = i_chunk * C
            rows = start + offs_c
            row_mask = rows < T

            qk_mask = row_mask[:, None] & (offs_k[None, :] < DK)
            v_mask = row_mask[:, None] & (offs_v[None, :] < DV)

            q_ptr = q_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            k_ptr = k_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            v_ptr = v_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            bg_ptr = b_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd
            wg_ptr = w_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            g_ptr = g_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd

            b_Qc = tl.load(q_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Kc = tl.load(k_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_Vc = tl.load(v_ptr, mask=v_mask, other=0.0).to(tl.float32)
            b_bc = tl.load(bg_ptr, mask=qk_mask, other=0.0).to(tl.float32)
            b_wc = tl.load(wg_ptr, mask=v_mask, other=0.0).to(tl.float32)
            b_gc = tl.load(g_ptr, mask=qk_mask, other=0.0).to(tl.float32)

            ck_ptr = (ck_base + i_chunk * s_cn
                      + offs_k[:, None] * s_ck + offs_v[None, :] * s_cv)
            b_Sin = tl.load(ck_ptr).to(tl.float32)

            # ----- recompute forward intermediates -----
            log_gamma = tl.cumsum(b_gc, axis=0)
            b_gamma = tl.exp(log_gamma)
            b_inv_gamma = tl.exp(-log_gamma)

            b_edir = b_bc * b_Kc
            b_ztgt = b_wc * b_Vc
            b_Kbar = b_Kc * b_inv_gamma
            b_Ebar = b_edir * b_gamma
            b_Qbar = b_Qc * b_gamma

            b_EK = _gdot(b_Ebar, tl.trans(b_Kbar), ALLOW_TF32)
            b_EK = tl.where(m_strict, b_EK, 0.0)

            b_Ainv = tl.where(m_strict, -b_EK, 0.0)
            for i in range(1, C):
                negT_i = tl.sum(tl.where(offs_c[:, None] == i, -b_EK, 0.0), 0)
                negT_i = tl.where(offs_c < i, negT_i, 0.0)
                corr = tl.sum(negT_i[:, None] * b_Ainv, 0)
                new_row = negT_i + corr
                b_Ainv = tl.where((offs_c == i)[:, None], new_row[None, :], b_Ainv)
            b_Ainv = b_Ainv + tl.where(m_eye, 1.0, 0.0)

            b_ES0 = _gdot(b_Ebar, b_Sin, ALLOW_TF32)
            b_rhs = b_ztgt - b_ES0
            b_U = _gdot(b_Ainv, b_rhs, ALLOW_TF32)

            b_QK = _gdot(b_Qbar, tl.trans(b_Kbar), ALLOW_TF32)
            b_QK = tl.where(m_tril, b_QK, 0.0)

            valid = tl.where(row_mask, rows, -1)
            last_idx = tl.max(valid, 0)
            last_local = last_idx - start
            sel = offs_c == last_local
            gamma_end = tl.sum(tl.where(sel[:, None], b_gamma, 0.0), 0)  # [DK]
            coef_tail = gamma_end[None, :] * b_inv_gamma                  # [C,DK]
            b_Ktail = b_Kc * coef_tail

            do_ptr = DO + i_bh * T * s_vt + rows[:, None] * s_vt + offs_v[None, :] * s_vd
            b_dOc = tl.load(do_ptr, mask=v_mask, other=0.0).to(tl.float32)

            # Accumulators (kept live across the section). Initialise the four
            # per-token grads here and fold contributions in as they are formed,
            # so the compiler can free transient tiles early (SMEM pressure).
            b_dKc = tl.zeros([C, DK], dtype=tl.float32)
            b_dinv_gamma = tl.zeros([C, DK], dtype=tl.float32)
            b_dgamma = tl.zeros([C, DK], dtype=tl.float32)
            b_dKbar = tl.zeros([C, DK], dtype=tl.float32)
            b_dEbar = tl.zeros([C, DK], dtype=tl.float32)

            # ===== state-carry backward: S_new = diag(gamma_end) S_in + K_tail^T U
            b_dS_in = gamma_end[:, None] * b_dS                           # [DK,DV]
            dgamma_end = tl.sum(b_dS * b_Sin, 1)                          # [DK]
            # dK_tail[j,k] = sum_d dS[k,d] U[j,d] ; dU[j,d] = sum_k Ktail[j,k] dS[k,d]
            b_dK_tail = tl.trans(_gdot(b_dS, tl.trans(b_U), ALLOW_TF32))  # [C,DK]
            b_dU = _gdot(b_Ktail, b_dS, ALLOW_TF32)           # [C,DV]
            b_dKc += b_dK_tail * coef_tail
            dgamma_end += tl.sum((b_dK_tail * b_Kc) * b_inv_gamma, 0)     # [DK]
            b_dinv_gamma += (b_dK_tail * b_Kc) * gamma_end[None, :]       # [C,DK]

            # ===== output backward: O = Q_bar S_in + QK U =====
            b_dU += _gdot(tl.trans(b_QK), b_dOc, ALLOW_TF32)  # [C,DV]
            b_dQbar = _gdot(b_dOc, tl.trans(b_Sin), ALLOW_TF32)  # [C,DK]
            b_dS_in += _gdot(tl.trans(b_Qbar), b_dOc, ALLOW_TF32)  # [DK,DV]
            # dQK = tril(dOc U^T); fold its two contributions, then drop it.
            b_dQK = tl.where(m_tril, _gdot(b_dOc, tl.trans(b_U), ALLOW_TF32), 0.0)
            b_dQbar += _gdot(b_dQK, b_Kbar, ALLOW_TF32)       # [C,DK]
            b_dKbar += _gdot(tl.trans(b_dQK), b_Qbar, ALLOW_TF32)  # [C,DK]

            # ===== U = A rhs ; rhs = z_tgt - E_bar S_in =====
            b_P = _gdot(tl.trans(b_Ainv), b_dU, ALLOW_TF32)   # A^T dU [C,DV]
            b_dEbar += -_gdot(b_P, tl.trans(b_Sin), ALLOW_TF32)  # [C,DK]
            b_dS_in -= _gdot(tl.trans(b_Ebar), b_P, ALLOW_TF32)   # [DK,DV]
            # dEK = tril(-P U^T, -1); fold contributions, then drop it.
            b_dEK = tl.where(m_strict, -_gdot(b_P, tl.trans(b_U), ALLOW_TF32), 0.0)
            b_dEbar += _gdot(b_dEK, b_Kbar, ALLOW_TF32)       # [C,DK]
            b_dKbar += _gdot(tl.trans(b_dEK), b_Ebar, ALLOW_TF32)  # [C,DK]
            b_dwc = b_P * b_Vc                                            # dz_tgt = drhs = P
            b_dVc = b_P * b_wc

            # ===== bar-transforms + gates =====
            b_dQc = b_dQbar * b_gamma
            b_dgamma += b_dQbar * b_Qc
            b_de_dir = b_dEbar * b_gamma
            b_dgamma += b_dEbar * b_edir
            b_dbc = b_de_dir * b_Kc
            b_dKc += b_de_dir * b_bc
            b_dKc += b_dKbar * b_inv_gamma
            b_dinv_gamma += b_dKbar * b_Kc
            b_dgamma += tl.where(sel[:, None], dgamma_end[None, :], 0.0)

            # gamma=exp(lg), inv_gamma=exp(-lg); log_gamma = cumsum(gc)
            b_dlog_gamma = b_dgamma * b_gamma - b_dinv_gamma * b_inv_gamma
            tot = tl.sum(b_dlog_gamma, 0)                                 # [DK]
            b_dgc = tot[None, :] - (tl.cumsum(b_dlog_gamma, axis=0) - b_dlog_gamma)

            # ----- store grads -----
            tl.store(dq_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd,
                     b_dQc.to(DQ.dtype.element_ty), mask=qk_mask)
            tl.store(dk_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd,
                     b_dKc.to(DKt.dtype.element_ty), mask=qk_mask)
            tl.store(dv_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd,
                     b_dVc.to(DVt.dtype.element_ty), mask=v_mask)
            tl.store(db_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd,
                     b_dbc.to(DBt.dtype.element_ty), mask=qk_mask)
            tl.store(dw_base + rows[:, None] * s_vt + offs_v[None, :] * s_vd,
                     b_dwc.to(DWt.dtype.element_ty), mask=v_mask)
            tl.store(dg_base + rows[:, None] * s_qt + offs_k[None, :] * s_qd,
                     b_dgc.to(DGt.dtype.element_ty), mask=qk_mask)

            b_dS = b_dS_in

        ds0_ptr = DS0 + i_bh * DK * s_sk + offs_k[:, None] * s_sk + offs_v[None, :] * s_sv
        tl.store(ds0_ptr, b_dS.to(DS0.dtype.element_ty))


def can_use_gdn2_chunk(
    S0: torch.Tensor,
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    C: int,
) -> bool:
    """Guard for the differentiable Triton GDN-2 chunk path."""
    import os

    if not _HAS_TRITON:
        return False
    if os.environ.get("GDN2_DISABLE_TRITON", "") == "1":
        return False
    if not (Q.is_cuda and K.is_cuda and V.is_cuda and S0.is_cuda):
        return False
    if Q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    dk = Q.shape[-1]
    dv = V.shape[-1]
    if dk not in _SUPPORTED_DK or dv not in _SUPPORTED_DV:
        return False
    if C not in (16, 32, 64):
        return False
    return True


class GDN2ChunkFunction(torch.autograd.Function):
    """Differentiable GDN-2 chunk-parallel kernel (forward + backward).

    All tensors are [BH, T, *].  ``g`` is log-decay (log α).
    """

    @staticmethod
    def forward(ctx, q, k, v, b, w, g, scale, initial_state, output_final_state, C):
        BH, T, dk = q.shape
        dv = v.shape[-1]
        n_chunks = (T + C - 1) // C

        qc = (q * scale).contiguous() if scale != 1.0 else q.contiguous()
        kc = k.contiguous()
        vc = v.contiguous()
        bc = b.contiguous()
        wc = w.contiguous()
        gc = g.contiguous()

        if initial_state is None:
            S0 = torch.zeros(BH, dk, dv, device=q.device, dtype=torch.float32)
        else:
            S0 = initial_state.contiguous().to(torch.float32)

        out = torch.empty(BH, T, dv, device=q.device, dtype=q.dtype)
        S_final = torch.empty(BH, dk, dv, device=q.device, dtype=torch.float32)
        sckpt = torch.empty(BH, n_chunks, dk, dv, device=q.device, dtype=torch.float32)

        grid = (BH,)
        _gdn2_chunk_fwd_kernel[grid](
            qc, kc, vc, bc, wc, gc,
            S0, out, S_final, sckpt,
            T,
            qc.stride(1), qc.stride(2),
            vc.stride(1), vc.stride(2),
            S0.stride(1), S0.stride(2),
            sckpt.stride(1), sckpt.stride(2), sckpt.stride(3),
            DK=dk, DV=dv, C=C,
            ALLOW_TF32=False,
            num_warps=4,
        )

        ctx.save_for_backward(qc, kc, vc, bc, wc, gc, sckpt)
        ctx.scale = scale
        ctx.C = C
        ctx.has_initial_state = initial_state is not None
        return out, (S_final if output_final_state else None)

    @staticmethod
    def backward(ctx, do, dS_final):
        qc, kc, vc, bc, wc, gc, sckpt = ctx.saved_tensors
        BH, T, dk = qc.shape
        dv = vc.shape[-1]
        C = ctx.C
        scale = ctx.scale

        do = do.contiguous()
        if dS_final is None:
            dS_final = torch.zeros(BH, dk, dv, device=qc.device, dtype=torch.float32)
        else:
            dS_final = dS_final.contiguous().to(torch.float32)

        dq = torch.empty(BH, T, dk, device=qc.device, dtype=torch.float32)
        dk_ = torch.empty(BH, T, dk, device=qc.device, dtype=torch.float32)
        dv_ = torch.empty(BH, T, dv, device=qc.device, dtype=torch.float32)
        db = torch.empty(BH, T, dk, device=qc.device, dtype=torch.float32)
        dw = torch.empty(BH, T, dv, device=qc.device, dtype=torch.float32)
        dg = torch.empty(BH, T, dk, device=qc.device, dtype=torch.float32)
        dS0 = torch.empty(BH, dk, dv, device=qc.device, dtype=torch.float32)

        grid = (BH,)
        _gdn2_chunk_bwd_kernel[grid](
            qc, kc, vc, bc, wc, gc,
            sckpt, do, dS_final,
            dq, dk_, dv_, db, dw, dg, dS0,
            T,
            qc.stride(1), qc.stride(2),
            vc.stride(1), vc.stride(2),
            dS0.stride(1), dS0.stride(2),
            sckpt.stride(1), sckpt.stride(2), sckpt.stride(3),
            DK=dk, DV=dv, C=C,
            ALLOW_TF32=False,
            num_warps=4,
        )

        # dq currently holds grad wrt (scale*q); chain the scale.
        if scale != 1.0:
            dq = dq * scale

        dq = dq.to(qc.dtype)
        dk_ = dk_.to(kc.dtype)
        dv_ = dv_.to(vc.dtype)
        db = db.to(bc.dtype)
        dw = dw.to(wc.dtype)
        dg = dg.to(gc.dtype)
        d_init = dS0.to(qc.dtype) if ctx.has_initial_state else None

        # forward signature: (q,k,v,b,w,g,scale,initial_state,output_final_state,C)
        return dq, dk_, dv_, db, dw, dg, None, d_init, None, None


def chunk_gdn2(
    q: torch.Tensor,     # [BH, T, dk]  (L2-normalised query)
    k: torch.Tensor,     # [BH, T, dk]  (L2-normalised key)
    v: torch.Tensor,     # [BH, T, dv]
    b: torch.Tensor,     # [BH, T, dk]  erase gate
    w: torch.Tensor,     # [BH, T, dv]  write gate
    g: torch.Tensor,     # [BH, T, dk]  log-decay (log alpha)
    scale: float = 1.0,
    initial_state: torch.Tensor | None = None,  # [BH, dk, dv]
    output_final_state: bool = False,
    chunk_size: int = 64,
):
    """Differentiable GDN-2 chunk-parallel kernel.

    Layout: [BH, T, *].  ``g`` is log α.  ``scale`` multiplies the query.

    Returns (out [BH, T, dv], final_state [BH, dk, dv] or None).
    """
    return GDN2ChunkFunction.apply(
        q, k, v, b, w, g, scale, initial_state, output_final_state, chunk_size
    )
