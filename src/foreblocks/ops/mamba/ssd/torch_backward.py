"""Pure-PyTorch vectorized chunked-SSD forward/backward.

This is the production backward path (see api.py's ``SSD_TRITON_BACKWARD_MIN_SEQLEN``
comment): the vectorized backward here is always used, regardless of which
forward path (Triton or PyTorch) ran, because the fused Triton backward is
O(chunks) sequential.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from foreblocks.kernels.mamba.segment_sum import _segment_sum_log


def _chunked_ssd_forward_torch_trapezoid(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int,
    adt: torch.Tensor | None,
    trap: torch.Tensor,
) -> torch.Tensor:
    if adt is None:
        raise ValueError("trapezoidal (trap) discretisation requires adt (Mamba3)")
    trap = trap.float()
    Bsz, T, H, P = u.shape
    # current tap: trap-weighted dt, real B/u, keeps the D-skip.
    y_cur = _chunked_ssd_forward_torch(
        u, dt * trap, A, B, C, D, chunk_size=chunk_size, adt=adt
    )
    # previous tap: (1-trap)-weighted dt, B/u shifted right by one (token -1
    # has no predecessor → zero), no D-skip.
    dt_prev = dt * (1.0 - trap)
    B_prev = F.pad(B, (0, 0, 0, 0, 1, 0))[:, :T]  # shift along time, pad front
    u_prev = F.pad(u, (0, 0, 0, 0, 1, 0))[:, :T]
    D_zero = torch.zeros_like(D)
    y_prev = _chunked_ssd_forward_torch(
        u_prev, dt_prev, A, B_prev, C, D_zero, chunk_size=chunk_size, adt=adt
    )
    return y_cur + y_prev


def _chunked_ssd_forward_torch(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
    trap: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    if trap is not None:
        return _chunked_ssd_forward_torch_trapezoid(
            u, dt, A, B, C, D, chunk_size=chunk_size, adt=adt, trap=trap
        )
    if u.ndim != 4:
        raise ValueError("u must have shape [B, T, H, P]")
    if dt.ndim != 3:
        raise ValueError("dt must have shape [B, T, H]")
    if A.ndim != 1 and adt is None:
        raise ValueError("A must have shape [H] for diagonal-A chunked SSD")
    if B.shape != C.shape or B.ndim != 4:
        raise ValueError("B and C must have matching shape [B, T, H, N]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    if dt.shape != (Bsz, T, H):
        raise ValueError("dt shape must match [B, T, H]")
    if A.shape != (H,) and adt is None:
        raise ValueError("A shape must match [H]")
    if B.shape[:3] != (Bsz, T, H):
        raise ValueError("B and C shape must match [B, T, H, N]")
    if D.shape != (H, P) and D.shape != (H,):
        raise ValueError("D shape must match [H, P] or [H]")

    out_dtype = u.dtype
    u = u.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()

    # ── pad to chunk boundary ──────────────────────────────────────
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    # ── dtA = dt * A — shape [B, T, H] ─────────────────────────────
    if adt is not None:
        # Mamba3: A is time-dependent [B, T, H], use pre-computed ADT
        dtA = adt.float()
        # Pad adt alongside dt/B/C
        if pad > 0:
            dtA = F.pad(dtA, (0, 0, 0, pad))
    else:
        dtA = dt * A  # broadcasting: [B, T, H] * [H] → [B, T, H]

    # ── reshape to chunks ──────────────────────────────────────────
    dtA = dtA.view(Bsz, nc, chunk_size, H)  # [B, nc, C, H]
    dt_raw_c = dt.view(Bsz, nc, chunk_size, H)  # [B, nc, C, H]
    u = u.view(Bsz, nc, chunk_size, H, P)
    B = B.view(Bsz, nc, chunk_size, H, N)
    C = C.view(Bsz, nc, chunk_size, H, N)

    # ── cumsum along chunk-time ────────────────────────────────────
    cumsum_dtA = torch.cumsum(dtA, dim=2)  # [B, nc, C, H]

    # ── L[c, t, j, h] = exp(sum(dt*A for k in j+1:t)) for j <= t ──
    # The state update applies decay before adding the current token, so the
    # source token j does not decay itself. This gives L[t, t] = 1.
    L_diff = cumsum_dtA.unsqueeze(-2) - cumsum_dtA.unsqueeze(-3)  # [B, nc, C, C, H]
    tril = torch.tril(
        torch.ones(chunk_size, chunk_size, device=u.device, dtype=torch.bool)
    )
    L_diff = L_diff.masked_fill(
        ~tril.unsqueeze(0).unsqueeze(0).unsqueeze(-1), float("-inf")
    )
    L = torch.exp(L_diff)  # [B, nc, C, C, H]

    # ── G[c, t, j, h] = sum_n C[c,t,n] * B[c,j,n] ────────────────
    G = (C.unsqueeze(3) * B.unsqueeze(2)).sum(dim=-1)  # [B, nc, C, C, H]

    # ── seq_idx masking: zero L, G where sequence boundaries differ ─
    # When seq_idx changes within a chunk, tokens from different sequences
    # should not interact (no cross-sequence state leakage).
    if seq_idx is not None:
        t_idx = torch.arange(chunk_size, device=u.device)  # [C]
        t_abs = (nc * chunk_size) + t_idx  # absolute token indices in this chunk
        seq_t = seq_idx[:, t_abs].unsqueeze(2)  # [B, C, 1]
        seq_j = seq_idx[:, t_abs].unsqueeze(1)  # [B, 1, C]
        same_seq = seq_t == seq_j  # [B, C, C]
        L = torch.where(same_seq[:, :, :, None].unsqueeze(0), L, torch.zeros_like(L))  # type: ignore[assignment]
        G = torch.where(same_seq.unsqueeze(0).unsqueeze(-1), G, torch.zeros_like(G))  # type: ignore[assignment]

    # ── Intra-chunk output ─────────────────────────────────────────
    # Y_intra[c,t,h,p] = sum_j L[c,t,j,h] * dt_raw[c,j,h] * G[c,t,j,h] * u[c,j,h,p]
    # Note: the second term in SSM is dt (not dtA = dt*A). dtA was used
    # only for the decay matrix abar = exp(dtA).
    # dt_raw_c reshaped to [B, nc, 1, C, H] to index by source time j
    LdtG = L * dt_raw_c[:, :, None, :, :] * G  # [B, nc, C_t, C_j, H]
    Y_intra = torch.einsum("bctjh,bcjhp->bcthp", LdtG, u)  # [B, nc, C, H, P]

    # ── Inter-chunk state ──────────────────────────────────────────
    # state_end[c, h, p, n] = accumulated intra-chunk state at end of chunk
    # L_last[c, j, h] = exp(cumsum_dtA[c, C-1, h] - cumsum_dtA[c, j, h])
    cumsum_dtA_last = cumsum_dtA[:, :, -1:, None, :]  # [B, nc, 1, 1, H]
    L_last = torch.exp(
        cumsum_dtA_last - cumsum_dtA[:, :, None, :, :]
    )  # [B, nc, 1, C, H]
    Ldt_last = L_last.squeeze(2) * dt_raw_c  # [B, nc, C, H]
    LB_last = Ldt_last.unsqueeze(-1) * B  # [B, nc, C, H, N]
    # state_end[c, h, p, n] = sum_j Ldt_last[c,j,h] * B[c,j,h,n] * u[c,j,h,p]
    state_end = torch.einsum("bcjhn,bcjhp->bchpn", LB_last, u)  # [B, nc, H, P, N]

    # ── Full parallel inter-chunk prefix scan ──────────────────────
    # Recurrence: boundary[c + 1] = decay_chunk[c] * boundary[c] + state_end[c].
    # FLA computes all boundary states with a lower-triangular decay matrix over
    # chunk summaries. We do the same in log space, avoiding a Python chunk loop.
    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)  # [B, nc+1, H, P, N]
    chunk_log_decay = cumsum_dtA[:, :, -1, :].transpose(1, 2)  # [B, H, nc]
    chunk_log_decay = F.pad(chunk_log_decay, (1, 0))  # [B, H, nc+1]
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)  # [B, nc+1, H, P, N]
    states_boundary = boundary_all[:, :-1]  # [B, nc, H, P, N]

    # ── Inter-chunk output ─────────────────────────────────────────
    # state_entered[c, t, h, p, n] = states_boundary[c, h, p, n] * decay_from_start[c, t, h]
    decay_from_start = torch.exp(cumsum_dtA)  # [B, nc, C, H]
    state_entered = states_boundary.unsqueeze(2) * decay_from_start.unsqueeze(
        -1
    ).unsqueeze(-1)  # [B, nc, C, H, P, N]
    # y_inter[c, t, h, p] = sum_n C[c,t,h,n] * state_entered[c,t,h,p,n]
    y_inter = torch.einsum("bcthn,bcthpn->bcthp", C, state_entered)  # [B, nc, C, H, P]

    # ── Total output: y = Y_intra + y_inter + D * u ───────────────
    if D.ndim == 2:
        # Mamba2: D is [H, P]
        y = Y_intra + y_inter + D.unsqueeze(0).unsqueeze(0) * u  # [B, nc, C, H, P]
    else:
        # Mamba3: D is [H] — broadcast over B, nc, C, P
        y = Y_intra + y_inter + D[:, None] * u  # [B, nc, C, H, P]

    # ── reshape + trim padding ─────────────────────────────────────
    y = y.reshape(Bsz, T_pad, H, P)
    if pad > 0:
        y = y[:, :T]
    return y.to(out_dtype)


def _chunked_ssd_backward_torch(
    grad_y: torch.Tensor,
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    needs_input_grad: tuple[bool, ...] | None = None,
    adt: torch.Tensor | None = None,
    needs_adt_grad: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    if needs_input_grad is None:
        needs_input_grad = (True,) * 6
    use_adt = adt is not None

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    D_per_head = D.ndim == 1

    gy = grad_y.float()
    u = u.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        gy = F.pad(gy, (0, 0, 0, 0, 0, pad))
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    T_pad = T + pad
    cs_ = chunk_size
    nc = T_pad // cs_

    if use_adt:
        dtA = adt.float()
        if pad > 0:
            dtA = F.pad(dtA, (0, 0, 0, pad))
    else:
        dtA = dt * A  # [B, T, H]

    dtA = dtA.view(Bsz, nc, cs_, H)
    dt_c = dt.view(Bsz, nc, cs_, H)
    u_c = u.view(Bsz, nc, cs_, H, P)
    B_c = B.view(Bsz, nc, cs_, H, N)
    C_c = C.view(Bsz, nc, cs_, H, N)
    gy_c = gy.view(Bsz, nc, cs_, H, P)

    cumsum_dtA = torch.cumsum(dtA, dim=2)  # [B,nc,C,H]
    tril = torch.tril(torch.ones(cs_, cs_, device=u.device, dtype=torch.bool))

    # ── recompute the forward quantities the backward needs ──────────────
    L_diff = cumsum_dtA.unsqueeze(-2) - cumsum_dtA.unsqueeze(-3)  # [B,nc,Ct,Cj,H]
    L = torch.exp(L_diff.masked_fill(~tril[None, None, :, :, None], float("-inf")))
    decay_from_start = torch.exp(cumsum_dtA)  # a[t]  [B,nc,C,H]
    cs_last = cumsum_dtA[:, :, -1:, :]  # [B,nc,1,H]

    # forward inter-chunk boundary states S_in[c]  [B,nc,H,P,N]
    Ldt_last = torch.exp(cs_last - cumsum_dtA) * dt_c  # [B,nc,C,H]
    state_end = torch.einsum("bcjh,bcjhn,bcjhp->bchpn", Ldt_last, B_c, u_c)
    zero_state = torch.zeros(Bsz, 1, H, P, N, device=u.device, dtype=torch.float32)
    state_summaries = torch.cat([zero_state, state_end], dim=1)  # [B,nc+1,H,P,N]
    chunk_log_decay = F.pad(cs_last.squeeze(2).transpose(1, 2), (1, 0))  # [B,H,nc+1]
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(1, 3)
    boundary_all = (
        decay_prefix[..., None, None] * state_summaries[:, :, None, ...]
    ).sum(dim=1)
    S_in = boundary_all[:, :-1]  # [B,nc,H,P,N]

    # ── adjoints ─────────────────────────────────────────────────────────
    # 1) D skip and du from it
    du = torch.zeros_like(u_c) if needs_input_grad[0] else None
    if D_per_head:
        dD = (gy_c * u_c).sum(dim=(0, 1, 2, 4)) if needs_input_grad[5] else None  # [H]
        Dexp = D[None, None, None, :, None]
    else:
        dD = (gy_c * u_c).sum(dim=(0, 1, 2)) if needs_input_grad[5] else None  # [H,P]
        Dexp = D[None, None, None, :, :]
    if du is not None:
        du = du + gy_c * Dexp

    # 2) y_inter[t,p] = a[t] * sum_n C[t,n] S_in[p,n]  → dC, dS_in, d(a[t]).
    #    Done with einsums to avoid materialising [B,nc,C,H,P,N] intermediates.
    a_t = decay_from_start  # [B,nc,C,H]
    gy_a = gy_c * a_t.unsqueeze(-1)  # gy[t,p] * a[t]  [B,nc,C,H,P]
    # dC[t,n] = a[t] * sum_p gy[t,p] S_in[p,n]
    dC = torch.einsum("bcthp,bchpn->bcthn", gy_a, S_in) if needs_input_grad[4] else None
    # dS_in[p,n] = sum_t a[t] gy[t,p] C[t,n]
    dS_in = torch.einsum("bcthp,bcthn->bchpn", gy_a, C_c)  # [B,nc,H,P,N]
    # d(cs[t]) from a[t]=exp(cs[t]):  a[t] * sum_{p,n} gy[t,p] C[t,n] S_in[p,n]
    dcs = torch.einsum("bcthp,bcthn,bchpn->bcth", gy_a, C_c, S_in)  # [B,nc,C,H]

    # 3) Y_intra = einsum LdtG · u  with LdtG[t,j]=L[t,j]·dt[j]·G[t,j], G=C[t]·B[j]
    G = (C_c.unsqueeze(3) * B_c.unsqueeze(2)).sum(dim=-1)  # [B,nc,Ct,Cj,H]
    LdtG = L * dt_c[:, :, None, :, :] * G  # [B,nc,Ct,Cj,H]
    # du[j] += sum_t LdtG[t,j] gy[t]
    if du is not None:
        du = du + torch.einsum("bctjh,bcthp->bcjhp", LdtG, gy_c)
    # d(LdtG)[t,j] = sum_p gy[t,p] u[j,p]
    gLdtG = torch.einsum("bcthp,bcjhp->bctjh", gy_c, u_c)  # [B,nc,Ct,Cj,H]
    Ldt = L * dt_c[:, :, None, :, :]
    gG = gLdtG * Ldt  # d(G)
    # G = sum_n C[t,n] B[j,n]  → dC[t] += sum_j gG[t,j] B[j];  dB[j] += sum_t gG[t,j] C[t]
    if dC is not None:
        dC = dC + torch.einsum("bctjh,bcjhn->bcthn", gG, B_c)
    dB = torch.einsum("bctjh,bcthn->bcjhn", gG, C_c) if needs_input_grad[3] else None
    # d(dt[j]) from LdtG: sum_t gLdtG[t,j] L[t,j] G[t,j]
    ddt = torch.zeros_like(dt_c) if needs_input_grad[1] else None
    if ddt is not None:
        ddt = ddt + (gLdtG * L * G).sum(dim=2)  # sum over t → [B,nc,Cj,H]
    # d(L[t,j]) from LdtG
    gL = gLdtG * dt_c[:, :, None, :, :] * G  # [B,nc,Ct,Cj,H]

    # 4) S_end_local feeds the boundary scan; backprop dS_in → d(state_end), d(decay)
    #    boundary_all[i] = sum_j decay_prefix[j,i] state_summaries[j]  (j source, i target)
    #    decay_prefix is [B, j, i, H] = exp(segsum[b,h,i,j]) after the forward transpose.
    #    dS_in is grad wrt boundary_all[:, :-1]; pad a zero for the unused last boundary.
    g_boundary = F.pad(dS_in, (0, 0, 0, 0, 0, 0, 0, 1))  # [B,nc+1,H,P,N]
    # d(state_summaries[j]) = sum_i decay_prefix[j,i] g_boundary[i]
    g_state_summaries = torch.einsum("bjih,bihpn->bjhpn", decay_prefix, g_boundary)
    g_state_end = g_state_summaries[:, 1:]  # [B,nc,H,P,N]
    # d(decay_prefix[j,i]) = sum_{p,n} g_boundary[i] state_summaries[j]
    g_decay_prefix = torch.einsum(
        "bihpn,bjhpn->bjih", g_boundary, state_summaries
    )  # [B,j,i,H]
    # decay_prefix[j,i] = exp(segsum[i,j]); segsum[i,j] = sum_{k=j+1..i} cld[k] (j<=i).
    # d(cld[k]) = sum_{i,j : j < k <= i} g_decay_prefix[j,i] * decay_prefix[j,i]
    gseg = g_decay_prefix * decay_prefix  # [B,j,i,H]
    ncp = nc + 1
    idx = torch.arange(ncp, device=u.device)
    kk = idx[:, None, None]
    ii = idx[None, None, :]
    jj = idx[None, :, None]
    seg_mask = ((jj < kk) & (kk <= ii)).to(gseg.dtype)  # [k, j, i]
    g_cld = torch.einsum("bjih,kji->bhk", gseg, seg_mask)  # [B,H,nc+1]
    g_cld = g_cld[
        :, :, 1:
    ]  # drop the padded leading zero → [B,H,nc]  d(cs_last per chunk)

    # 5) state_end[c] adjoint: state_end = einsum(Ldt_last,B,u); also gets g_state_end.
    #    Ldt_last[j] = exp(cs_last - cs[j]) dt[j];  let w[j]=exp(cs_last-cs[j])
    w = torch.exp(cs_last - cumsum_dtA)  # [B,nc,C,H]
    # du[j] += sum_n g_state_end[.,n] (w dt)[j] B[j,n]
    wdt = w * dt_c  # [B,nc,C,H]
    if du is not None:
        du = du + torch.einsum("bchpn,bcjh,bcjhn->bcjhp", g_state_end, wdt, B_c)
    # dB[j] += sum_p g_state_end[p,.] (w dt)[j] u[j,p]
    if dB is not None:
        dB = dB + torch.einsum("bchpn,bcjh,bcjhp->bcjhn", g_state_end, wdt, u_c)
    # d(wdt)[j] = sum_{p,n} g_state_end[p,n] B[j,n] u[j,p]
    g_wdt = torch.einsum("bchpn,bcjhn,bcjhp->bcjh", g_state_end, B_c, u_c)  # [B,nc,C,H]
    if ddt is not None:
        ddt = ddt + g_wdt * w
    g_w = g_wdt * dt_c  # d(w[j])
    # w[j]=exp(cs_last-cs[j]) → d(cs_last)+= g_w*w ; d(cs[j]) -= g_w*w
    dcs_last_from_w = (g_w * w).sum(dim=2)  # [B,nc,H]
    dcs = dcs - (g_w * w)  # into d(cumsum_dtA)[j]

    # 6) gather d(cumsum_dtA): from a[t] (dcs), from L[t,j], from chunk decay, from w
    # L[t,j]=exp(cs[t]-cs[j]) (t>=j): d(cs[t]) += gL*L ; d(cs[j]) -= gL*L
    gLL = gL * L  # [B,nc,Ct,Cj,H], already zero where t<j (L=0 there)
    dcs = dcs + gLL.sum(dim=3)  # over j → contributes to cs[t]
    dcs = dcs - gLL.sum(dim=2)  # over t → contributes to cs[j]
    # chunk_log_decay = cs[-1]; combine the two cs_last grads (from w and from scan)
    dcs_last_total = dcs_last_from_w + g_cld.transpose(1, 2)  # [B,nc,H]
    # add to the last time-step of each chunk's cumsum
    dcs[:, :, -1, :] = dcs[:, :, -1, :] + dcs_last_total

    # 7) cumsum_dtA = cumsum(dtA, dim=time); adjoint is reverse-cumsum
    d_dtA = torch.flip(
        torch.cumsum(torch.flip(dcs, dims=[2]), dim=2), dims=[2]
    )  # [B,nc,C,H]

    # 8) split d_dtA into dt/A (Mamba2) or dadt (Mamba3)
    dadt = None
    dA = None
    if use_adt:
        if needs_adt_grad:
            dadt = d_dtA  # dtA == adt
    else:
        # dtA = dt * A
        if ddt is not None:
            ddt = ddt + d_dtA * A[None, None, None, :]
        if needs_input_grad[2]:
            dA = (d_dtA * dt_c).sum(dim=(0, 1, 2))  # [H]

    # ── reshape back to [B, T, ...] and trim padding ─────────────────────
    def _unchunk(t, vec_dims):
        if t is None:
            return None
        t = t.reshape((Bsz, T_pad) + t.shape[3:])
        return t[:, :T] if pad > 0 else t

    du = _unchunk(du, None)
    ddt = _unchunk(ddt, None)
    dB = _unchunk(dB, None)
    dC = _unchunk(dC, None)
    dadt = _unchunk(dadt, None)

    out_dtype = grad_y.dtype
    cast = lambda x: None if x is None else x.to(out_dtype)
    return (
        cast(du),
        cast(ddt),
        dA if dA is None else dA.to(out_dtype),
        cast(dB),
        cast(dC),
        dD if dD is None else dD.to(out_dtype),
        cast(dadt),
    )

