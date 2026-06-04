from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    CHUNKED_SSD_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    CHUNKED_SSD_TRITON_AVAILABLE = False


# Backward selection (when the Triton forward ran): the fused Triton backward is
# memory-flat and on par with the vectorised backward at long T, but ~15-20%
# slower at short T (per-chunk launch + boundary recompute). The vectorised
# backward is faster at short T but its memory grows with T (materialises the
# [B, nc, C, C, H] intra-chunk matrices) and eventually OOMs. So we pick by
# sequence length: vectorised below the threshold, Triton at/above it.
SSD_TRITON_BACKWARD_MIN_SEQLEN = 1024


# ── helpers ──────────────────────────────────────────────────────────


def segment_sum(x: torch.Tensor) -> torch.Tensor:
    """Exp(lower-triangular cumulative sum).

    Given ``x`` of shape ``[..., C]``, returns ``L`` of shape ``[..., C, C]``
    with ``L[i, j] = exp(sum(x[j : i+1]))`` for ``j <= i`` and ``0`` otherwise.

    Stable formulation using cumsum differences.
    """
    cumsum = torch.cumsum(x, dim=-1)  # [..., C]
    cumsum_pad = F.pad(cumsum, (1, 0), value=0.0)  # [..., C+1]
    cumsum_pad = cumsum_pad[..., :-1]  # [..., C], cumsum_pad[j] = cumsum[j-1]
    diff = cumsum[..., :, None] - cumsum_pad[..., None, :]  # [..., C, C]
    tril = torch.tril(
        torch.ones(x.shape[-1], x.shape[-1], device=x.device, dtype=torch.bool)
    )
    diff = diff.masked_fill(~tril, float("-inf"))
    return torch.exp(diff)  # [..., C, C]


def _segment_sum_log(x: torch.Tensor) -> torch.Tensor:
    """FLA-style stable lower-triangular segment sums in log space."""
    size = x.size(-1)
    expanded = x[..., None].expand(*x.shape, size)
    strict_lower = torch.tril(
        torch.ones(size, size, device=x.device, dtype=torch.bool),
        diagonal=-1,
    )
    expanded = expanded.masked_fill(~strict_lower, 0.0)
    segsum = torch.cumsum(expanded, dim=-2)
    lower = torch.tril(torch.ones(size, size, device=x.device, dtype=torch.bool))
    return segsum.masked_fill(~lower, float("-inf"))


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
    """Trapezoidal-discretisation forward, exact via scan linearity.

    State update:
        h_t = abar_t * h_{t-1} + dt_t * [trap_t * k_t + (1 - trap_t) * k_{t-1}]
    where ``k_t = B_t u_t``.  The state is linear in the per-step rank-1
    injection ``B u``, so the trapezoidal state equals the sum of two standard
    (Euler) scans that share decay/C/D:

      * "current" tap : weight ``trap * dt``,        source (B_t,   u_t)
      * "previous" tap: weight ``(1 - trap) * dt``,  source (B_{t-1}, u_{t-1})

    The D-skip ``D * u_t`` is part of the output (not the state), so it is
    applied once — by the current-tap scan only; the previous-tap scan passes
    ``D = 0``.

    Requires ``adt`` (Mamba3): the decay must come from ``adt`` so that splitting
    ``dt`` into trap-weighted taps changes only the injection, not the decay.
    """
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
) -> torch.Tensor:
    """Torch chunked SSM forward with **diagonal A** (one scalar per head).

    Implements the Mamba2-style scan:

    1. Intra-chunk output via the ``L`` matrix (causal attention pattern)
    2. Inter-chunk state propagation

    Args:
        u: ``[B, T, H, P]`` — input sequence
        dt: ``[B, T, H]`` — discretised time-step (post softplus + clamp)
        A: ``[H]`` — scalar per head (diagonal A matrix, already negated)
        B: ``[B, T, H, N]`` — B projection
        C: ``[B, T, H, N]`` — C projection
        D: ``[H, P]`` — skip connection
        chunk_size: tokens per chunk
        trap: ``[B, T, H]`` or None — trapezoidal-discretisation gate (Mamba3).
            When given, the input injected at step ``t`` is the blend
            ``dt_t * [trap_t * (B_t u_t) + (1 - trap_t) * (B_{t-1} u_{t-1})]``
            instead of the plain Euler ``dt_t * B_t u_t`` (``trap=1`` recovers
            Euler).  Implemented exactly via the linearity of the scan: the sum
            of the standard scan on the "current" tap and on the one-step-shifted
            "previous" tap (the D-skip is applied once).

    Returns:
        y: ``[B, T, H, P]`` — SSM output
    """
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
    """Vectorised chunked backward matching ``_chunked_ssd_forward_torch``.

    Parallel over chunks (no Python time loop) and bounded by the small
    ``[B, nc, C, C, H]`` intra-chunk matrices (kept tiny by a modest chunk_size,
    FLA-style). Returns ``(du, ddt, dA, dB, dC, dD, dadt)``.

    Forward (per chunk, S_in = state entering the chunk; cs = cumsum of dtA):
        Y_intra[t] = sum_{j<=t} L[t,j] dt[j] (C[t]·B[j]) u[j],  L[t,j]=exp(cs[t]-cs[j])
        y_inter[t] = (C[t] · (exp(cs[t]) ⊙ S_in))            # over state dim n
        y[t]       = Y_intra[t] + y_inter[t] + D u[t]
        S_end[c]   = exp(cs[-1]) S_in + sum_j exp(cs[-1]-cs[j]) dt[j] B[j]⊗u[j]
    with S_in given by a prefix scan over the per-chunk local end-states.
    """
    if needs_input_grad is None:
        needs_input_grad = (True,) * 6
    use_adt = adt is not None

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    D_per_head = D.ndim == 1

    gy = grad_y.float()
    u = u.float(); dt = dt.float(); A = A.float()
    B = B.float(); C = C.float(); D = D.float()

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
    boundary_all = (decay_prefix[..., None, None] * state_summaries[:, :, None, ...]).sum(dim=1)
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
    g_decay_prefix = torch.einsum("bihpn,bjhpn->bjih", g_boundary, state_summaries)  # [B,j,i,H]
    # decay_prefix[j,i] = exp(segsum[i,j]); segsum[i,j] = sum_{k=j+1..i} cld[k] (j<=i).
    # d(cld[k]) = sum_{i,j : j < k <= i} g_decay_prefix[j,i] * decay_prefix[j,i]
    gseg = g_decay_prefix * decay_prefix  # [B,j,i,H]
    ncp = nc + 1
    idx = torch.arange(ncp, device=u.device)
    kk = idx[:, None, None]; ii = idx[None, None, :]; jj = idx[None, :, None]
    seg_mask = ((jj < kk) & (kk <= ii)).to(gseg.dtype)  # [k, j, i]
    g_cld = torch.einsum("bjih,kji->bhk", gseg, seg_mask)  # [B,H,nc+1]
    g_cld = g_cld[:, :, 1:]  # drop the padded leading zero → [B,H,nc]  d(cs_last per chunk)

    # 5) state_end[c] adjoint: state_end = einsum(Ldt_last,B,u); also gets g_state_end.
    #    Ldt_last[j] = exp(cs_last - cs[j]) dt[j];  let w[j]=exp(cs_last-cs[j])
    w = torch.exp(cs_last - cumsum_dtA)  # [B,nc,C,H]
    # du[j] += sum_n g_state_end[.,n] (w dt)[j] B[j,n]
    wdt = (w * dt_c)  # [B,nc,C,H]
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
    gLL = (gL * L)  # [B,nc,Ct,Cj,H], already zero where t<j (L=0 there)
    dcs = dcs + gLL.sum(dim=3)  # over j → contributes to cs[t]
    dcs = dcs - gLL.sum(dim=2)  # over t → contributes to cs[j]
    # chunk_log_decay = cs[-1]; combine the two cs_last grads (from w and from scan)
    dcs_last_total = dcs_last_from_w + g_cld.transpose(1, 2)  # [B,nc,H]
    # add to the last time-step of each chunk's cumsum
    dcs[:, :, -1, :] = dcs[:, :, -1, :] + dcs_last_total

    # 7) cumsum_dtA = cumsum(dtA, dim=time); adjoint is reverse-cumsum
    d_dtA = torch.flip(torch.cumsum(torch.flip(dcs, dims=[2]), dim=2), dims=[2])  # [B,nc,C,H]

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
    return (cast(du), cast(ddt), dA if dA is None else dA.to(out_dtype),
            cast(dB), cast(dC), dD if dD is None else dD.to(out_dtype), cast(dadt))


if CHUNKED_SSD_TRITON_AVAILABLE:

    @triton.jit
    def _chunked_ssd_forward_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        adt_ptr,
        state_ptr,
        out_ptr,
        T,
        H,
        P,
        N,
        CHUNK_INDEX: tl.constexpr,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
    ):
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H
        t0 = CHUNK_INDEX * CHUNK_SIZE

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N

        state_base = ((b * H + h) * P + p_offs[:, None]) * N + n_offs[None, :]
        state_mask = p_mask[:, None] & n_mask[None, :]
        state = tl.load(state_ptr + state_base, mask=state_mask, other=0.0).to(
            tl.float32
        )

        # A is only used in the per-head (dt*A) parametrisation.
        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        # D: [H, P] (per head_dim) or [H] (per head, broadcast over P).
        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(tl.float32)

        for ti in tl.range(0, CHUNK_SIZE):
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h

            u_vals = tl.load(
                u_ptr + base_bth * P + p_offs,
                mask=active & p_mask,
                other=0.0,
            ).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(
                B_ptr + base_bth * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)
            c_vals = tl.load(
                C_ptr + base_bth * N + n_offs,
                mask=active & n_mask,
                other=0.0,
            ).to(tl.float32)

            if USE_ADT:
                # time-dependent log-decay (Mamba3): decay = exp(adt_t)
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)
            new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
            state = tl.where(active, new_state, state)

            y_vals = tl.sum(state * c_vals[None, :], axis=1) + d_vals * u_vals
            tl.store(out_ptr + base_bth * P + p_offs, y_vals, mask=active & p_mask)

        tl.store(state_ptr + state_base, state, mask=state_mask)

    @triton.jit
    def _chunked_ssd_boundary_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        adt_ptr,
        bstate_ptr,  # [B, H, nc+1, P, N] — entry state of each chunk (bstate[0]=0)
        T,
        H,
        P,
        N,
        NC,
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
    ):
        """Recompute and store the per-chunk entry states for the backward pass."""
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)
        b = pid_bh // H
        h = pid_bh % H

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N
        pn_mask = p_mask[:, None] & n_mask[None, :]

        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

        # bstate layout stride: ((b*H + h)*(NC+1) + c)*P*N + p*N + n
        bstride_c = P * N
        bbase = ((b * H + h) * (NC + 1)) * bstride_c + p_offs[:, None] * N + n_offs[None, :]
        # chunk 0 entry state is zero
        tl.store(bstate_ptr + bbase, state, mask=pn_mask)

        for c in tl.range(0, NC):
            t0 = c * CHUNK_SIZE
            for ti in tl.range(0, CHUNK_SIZE):
                t = t0 + ti
                active = t < T
                base_bth = (b * T + t) * H + h
                u_vals = tl.load(u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0).to(tl.float32)
                dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
                b_vals = tl.load(B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0).to(tl.float32)
                if USE_ADT:
                    log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
                else:
                    log_decay = dt_val * a_val
                decay = tl.exp(log_decay)
                new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
                state = tl.where(active, new_state, state)
            # store entry state of chunk c+1
            wbase = ((b * H + h) * (NC + 1) + (c + 1)) * bstride_c + p_offs[:, None] * N + n_offs[None, :]
            tl.store(bstate_ptr + wbase, state, mask=pn_mask)

    @triton.jit
    def _chunked_ssd_backward_kernel(
        gy_ptr,
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        adt_ptr,
        bstate_ptr,  # [B, H, nc+1, P, N]
        scratch_ptr,  # [B*H, BLOCK_P_GRID, CHUNK_SIZE, BLOCK_P, BLOCK_N] forward states
        du_ptr,
        ddt_ptr,
        dA_ptr,      # [H]            (atomic; unused when USE_ADT)
        dB_ptr,      # [B, T, H, N]   (atomic over p-blocks)
        dC_ptr,      # [B, T, H, N]   (atomic over p-blocks)
        dD_ptr,      # [H] or [H, P]  (atomic)
        dadt_ptr,    # [B, T, H]      (atomic over p-blocks; unused when not USE_ADT)
        gstate_ptr,  # [B, H, P, N] running grad-state carried across chunks (reverse)
        T,
        H,
        P,
        N,
        NC,
        NPB,         # number of p-blocks (grid dim 1)
        CHUNK_SIZE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_ADT: tl.constexpr,
        D_PER_HEAD: tl.constexpr,
        CHUNK_INDEX: tl.constexpr,
    ):
        """Reverse-time backward for one chunk (driver loops chunks high→low).

        Numerically stable: recomputes the chunk's forward states into a scratch
        buffer (a forward sweep from bstate[c]), then walks reverse reading state_t
        and state_{t-1} from scratch — no division by the (possibly tiny) decay.
        grad_state is carried between chunks via gstate_ptr.
        """
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)
        b = pid_bh // H
        h = pid_bh % H

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N
        pn_mask = p_mask[:, None] & n_mask[None, :]

        a_val = tl.load(A_ptr + h).to(tl.float32) if not USE_ADT else 0.0
        if D_PER_HEAD:
            d_vals = tl.load(D_ptr + h).to(tl.float32) + 0.0 * p_offs
        else:
            d_vals = tl.load(D_ptr + h * P + p_offs, mask=p_mask, other=0.0).to(tl.float32)

        bstride_c = P * N
        # scratch base for this (pid_bh, pid_p): [CHUNK_SIZE, BLOCK_P, BLOCK_N]
        sc_block = CHUNK_SIZE * BLOCK_P * BLOCK_N
        sc0 = (pid_bh * NPB + pid_p) * sc_block
        pn_idx = tl.arange(0, BLOCK_P)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]

        t0 = CHUNK_INDEX * CHUNK_SIZE

        # ── forward sweep: recompute & store state_after[ti] for this chunk ──
        entry_base = ((b * H + h) * (NC + 1) + CHUNK_INDEX) * bstride_c + p_offs[:, None] * N + n_offs[None, :]
        state = tl.load(bstate_ptr + entry_base, mask=pn_mask, other=0.0).to(tl.float32)
        for ti in tl.range(0, CHUNK_SIZE):
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h
            u_vals = tl.load(u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0).to(tl.float32)
            if USE_ADT:
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)
            new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
            state = tl.where(active, new_state, state)
            tl.store(scratch_ptr + sc0 + ti * (BLOCK_P * BLOCK_N) + pn_idx, state)

        # ── reverse sweep ──
        gstate_base = ((b * H + h) * P + p_offs[:, None]) * N + n_offs[None, :]
        grad_state = tl.load(gstate_ptr + gstate_base, mask=pn_mask, other=0.0).to(tl.float32)

        dA_acc = 0.0
        for ti_rev in tl.range(0, CHUNK_SIZE):
            ti = CHUNK_SIZE - 1 - ti_rev
            t = t0 + ti
            active = t < T
            base_bth = (b * T + t) * H + h

            u_vals = tl.load(u_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0).to(tl.float32)
            dt_val = tl.load(dt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            b_vals = tl.load(B_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0).to(tl.float32)
            c_vals = tl.load(C_ptr + base_bth * N + n_offs, mask=active & n_mask, other=0.0).to(tl.float32)
            gy_vals = tl.load(gy_ptr + base_bth * P + p_offs, mask=active & p_mask, other=0.0).to(tl.float32)
            if USE_ADT:
                log_decay = tl.load(adt_ptr + base_bth, mask=active, other=0.0).to(tl.float32)
            else:
                log_decay = dt_val * a_val
            decay = tl.exp(log_decay)

            # state_t = state_after[ti]; state_prev = state_after[ti-1] (entry if ti==0)
            state_t = tl.load(scratch_ptr + sc0 + ti * (BLOCK_P * BLOCK_N) + pn_idx)
            if ti == 0:
                state_prev = tl.load(bstate_ptr + entry_base, mask=pn_mask, other=0.0).to(tl.float32)
            else:
                state_prev = tl.load(scratch_ptr + sc0 + (ti - 1) * (BLOCK_P * BLOCK_N) + pn_idx)

            # ── output term: y[p] = sum_n state_t[p,n] c[n] + D u[p] ──
            dC_partial = tl.sum(gy_vals[:, None] * state_t, axis=0)  # [BLOCK_N]
            tl.atomic_add(dC_ptr + base_bth * N + n_offs, dC_partial, mask=active & n_mask)
            if D_PER_HEAD:
                tl.atomic_add(dD_ptr + h, tl.sum(gy_vals * u_vals, axis=0), mask=active)
            else:
                tl.atomic_add(dD_ptr + h * P + p_offs, gy_vals * u_vals, mask=active & p_mask)
            du_acc = gy_vals * d_vals  # du from D-skip
            grad_state = grad_state + gy_vals[:, None] * c_vals[None, :]

            # ── state term: state_t = decay state_prev + dt u⊗B ──
            du_acc = du_acc + tl.sum(grad_state * (dt_val * b_vals[None, :]), axis=1)
            tl.atomic_add(du_ptr + base_bth * P + p_offs, du_acc, mask=active & p_mask)
            dB_partial = tl.sum(grad_state * (dt_val * u_vals[:, None]), axis=0)
            tl.atomic_add(dB_ptr + base_bth * N + n_offs, dB_partial, mask=active & n_mask)
            ddt_partial = tl.sum(grad_state * (u_vals[:, None] * b_vals[None, :]))
            g_logdecay = tl.sum(grad_state * decay * state_prev)
            if USE_ADT:
                tl.atomic_add(dadt_ptr + base_bth, g_logdecay, mask=active)
                tl.atomic_add(ddt_ptr + base_bth, ddt_partial, mask=active)
            else:
                tl.atomic_add(ddt_ptr + base_bth, ddt_partial + g_logdecay * a_val, mask=active)
                dA_acc += tl.where(active, g_logdecay * dt_val, 0.0)
            grad_state = grad_state * decay

        tl.store(gstate_ptr + gstate_base, grad_state, mask=pn_mask)
        if not USE_ADT:
            tl.atomic_add(dA_ptr + h, dA_acc)


def chunked_ssd_forward_triton(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Triton chunk kernel for SSD forward.

    Supports both decay parametrisations:

    * ``adt is None`` (Mamba2): per-head scalar ``A`` ``[H]``, decay ``exp(dt*A)``.
    * ``adt`` given (Mamba3): time-dependent log-decay ``[B, T, H]``, decay
      ``exp(adt)``; ``A`` is ignored.

    ``D`` may be per-head_dim ``[H, P]`` or per-head ``[H]``.

    The kernel fuses the recurrent work within each chunk and carries only the
    compact boundary state between chunks. Inter-chunk prefix propagation is
    still sequential, but the large intra-chunk ``L``/``G`` tensors are avoided.
    """
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_forward_triton called but Triton is unavailable"
        )
    use_adt = adt is not None
    tensors = [u, dt, A, B, C, D] + ([adt] if use_adt else [])
    if not all(t.is_cuda for t in tensors):
        raise ValueError("chunked_ssd_forward_triton expects CUDA tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch_size, seqlen, num_heads, head_dim = u.shape
    d_state = B.shape[-1]
    if dt.shape != (batch_size, seqlen, num_heads):
        raise ValueError("dt shape must match [B, T, H]")
    if not use_adt and A.shape != (num_heads,):
        raise ValueError("A shape must match [H]")
    if use_adt and adt.shape != (batch_size, seqlen, num_heads):
        raise ValueError("adt shape must match [B, T, H]")
    if B.shape != (batch_size, seqlen, num_heads, d_state) or C.shape != B.shape:
        raise ValueError("B and C must have shape [B, T, H, N]")
    d_per_head = D.shape == (num_heads,)
    if D.shape != (num_heads, head_dim) and not d_per_head:
        raise ValueError("D shape must match [H, P] or [H]")
    if d_state > 128:
        raise ValueError("chunked_ssd_forward_triton currently supports d_state <= 128")

    u_contig = u.contiguous()
    dt_contig = dt.contiguous()
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    C_contig = C.contiguous()
    D_contig = D.contiguous()
    adt_contig = adt.contiguous() if use_adt else u_contig  # dummy ptr when unused
    out = torch.empty_like(u_contig)

    state = torch.zeros(
        batch_size,
        num_heads,
        head_dim,
        d_state,
        device=u.device,
        dtype=torch.float32,
    )
    block_p = min(max(triton.next_power_of_2(head_dim), 16), 128)
    block_n = max(8, triton.next_power_of_2(d_state))
    n_chunks = triton.cdiv(seqlen, chunk_size)

    grid = (batch_size * num_heads, triton.cdiv(head_dim, block_p))
    for chunk_idx in range(n_chunks):
        _chunked_ssd_forward_kernel[grid](
            u_contig,
            dt_contig,
            A_contig,
            B_contig,
            C_contig,
            D_contig,
            adt_contig,
            state,
            out,
            seqlen,
            num_heads,
            head_dim,
            d_state,
            CHUNK_INDEX=chunk_idx,
            CHUNK_SIZE=chunk_size,
            BLOCK_P=block_p,
            BLOCK_N=block_n,
            USE_ADT=use_adt,
            D_PER_HEAD=d_per_head,
        )
    return out


def chunked_ssd_backward_triton(
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
    """Fused Triton backward (reverse-time per-chunk recurrence).

    Two passes: (1) recompute per-chunk entry states; (2) walk chunks high→low,
    reconstructing state_{t-1}=(state_t - dt u⊗B)/decay and accumulating grads.
    Stable only when the per-step decay is not tiny (guaranteed by Mamba2/Mamba3
    A-init); callers should fall back to the vectorised path otherwise.

    Returns ``(du, ddt, dA, dB, dC, dD, dadt)``.
    """
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError("chunked_ssd_backward_triton called but Triton is unavailable")
    use_adt = adt is not None
    B_, T, H, P = u.shape
    N = B.shape[-1]
    d_per_head = D.ndim == 1

    gy = grad_y.contiguous()
    u_c = u.contiguous(); dt_c = dt.contiguous()
    A_c = A.contiguous(); Bm = B.contiguous(); Cm = C.contiguous(); Dm = D.contiguous()
    adt_c = adt.contiguous() if use_adt else u_c
    nc = triton.cdiv(T, chunk_size)

    bstate = torch.empty(B_, H, nc + 1, P, N, device=u.device, dtype=torch.float32)
    # grads (fp32 accumulators for atomics)
    du = torch.zeros_like(u_c, dtype=torch.float32)
    ddt = torch.zeros(B_, T, H, device=u.device, dtype=torch.float32)
    dA = torch.zeros(H, device=u.device, dtype=torch.float32)
    dB = torch.zeros(B_, T, H, N, device=u.device, dtype=torch.float32)
    dC = torch.zeros(B_, T, H, N, device=u.device, dtype=torch.float32)
    dD = torch.zeros_like(Dm, dtype=torch.float32)
    dadt = torch.zeros(B_, T, H, device=u.device, dtype=torch.float32)
    gstate = torch.zeros(B_, H, P, N, device=u.device, dtype=torch.float32)

    block_p = min(max(triton.next_power_of_2(P), 16), 128)
    block_n = max(8, triton.next_power_of_2(N))
    npb = triton.cdiv(P, block_p)
    grid = (B_ * H, npb)

    _chunked_ssd_boundary_kernel[grid](
        u_c, dt_c, A_c, Bm, adt_c, bstate,
        T, H, P, N, nc,
        CHUNK_SIZE=chunk_size, BLOCK_P=block_p, BLOCK_N=block_n, USE_ADT=use_adt,
    )
    # scratch for recomputed forward states of the chunk under processing
    scratch = torch.empty(
        B_ * H, npb, chunk_size, block_p, block_n, device=u.device, dtype=torch.float32
    )
    for chunk_idx in range(nc - 1, -1, -1):
        _chunked_ssd_backward_kernel[grid](
            gy, u_c, dt_c, A_c, Bm, Cm, Dm, adt_c, bstate, scratch,
            du, ddt, dA, dB, dC, dD, dadt, gstate,
            T, H, P, N, nc, npb,
            CHUNK_SIZE=chunk_size, BLOCK_P=block_p, BLOCK_N=block_n,
            USE_ADT=use_adt, D_PER_HEAD=d_per_head, CHUNK_INDEX=chunk_idx,
        )

    od = grad_y.dtype
    out_du = du.to(od) if needs_input_grad[0] else None
    out_ddt = ddt.to(od) if needs_input_grad[1] else None
    out_dA = (dA.to(od) if (needs_input_grad[2] and not use_adt) else None)
    out_dB = dB.to(od) if needs_input_grad[3] else None
    out_dC = dC.to(od) if needs_input_grad[4] else None
    out_dD = dD.to(od) if needs_input_grad[5] else None
    out_dadt = (dadt.to(od) if (use_adt and needs_adt_grad) else None)
    return (out_du, out_ddt, out_dA, out_dB, out_dC, out_dD, out_dadt)


class _ChunkedSSDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C, D, chunk_size: int, use_triton: bool, adt=None):
        ctx.chunk_size = chunk_size
        if use_triton:
            y = chunked_ssd_forward_triton(u, dt, A, B, C, D, chunk_size=chunk_size, adt=adt)
        else:
            y = _chunked_ssd_forward_torch(u, dt, A, B, C, D, chunk_size=chunk_size, adt=adt)
        # Backward path is chosen at backward() time by sequence length (only the
        # fused Triton kernels are viable when the forward also ran on Triton).
        ctx.use_triton = use_triton
        ctx.save_for_backward(u, dt, A, B, C, D, adt)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, dt, A, B, C, D, adt = ctx.saved_tensors
        seqlen = u.shape[1]
        use_triton_bwd = ctx.use_triton and seqlen >= SSD_TRITON_BACKWARD_MIN_SEQLEN
        bwd = chunked_ssd_backward_triton if use_triton_bwd else _chunked_ssd_backward_torch
        grads = bwd(
            grad_y,
            u,
            dt,
            A,
            B,
            C,
            D,
            chunk_size=ctx.chunk_size,
            needs_input_grad=ctx.needs_input_grad[:6],
            adt=adt,
            needs_adt_grad=ctx.needs_input_grad[8],
        )
        du, ddt, dA, dB, dC, dD, dadt = grads
        return (du, ddt, dA, dB, dC, dD, None, None, dadt)


def chunked_ssd_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    use_triton: bool = False,
    adt: torch.Tensor | None = None,
    trap: torch.Tensor | None = None,
) -> torch.Tensor:
    can_use_triton = (
        use_triton
        and CHUNKED_SSD_TRITON_AVAILABLE
        and u.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and B.is_cuda
        and C.is_cuda
        and D.is_cuda
        and B.shape[-1] <= 128
    )
    if trap is not None:
        # Trapezoidal discretisation (Mamba3). The trapezoidal state is exactly
        # the sum of two standard (Euler) scans that share decay/C/D — a
        # "current" tap (weight trap*dt, real B/u, with the D-skip) and a
        # one-step-shifted "previous" tap (weight (1-trap)*dt, no D-skip). Using
        # the existing _ChunkedSSDFn for each tap gives the fused triton kernel
        # at eval and the verified per-tap backward during training, for free.
        if adt is None:
            raise ValueError("trapezoidal (trap) discretisation requires adt (Mamba3)")
        T = u.shape[1]
        trap_h = trap  # [B, T, H], broadcasts against dt [B, T, H]
        y_cur = _ChunkedSSDFn.apply(
            u, dt * trap_h, A, B, C, D, chunk_size, can_use_triton, adt
        )
        B_prev = F.pad(B, (0, 0, 0, 0, 1, 0))[:, :T]
        u_prev = F.pad(u, (0, 0, 0, 0, 1, 0))[:, :T]
        y_prev = _ChunkedSSDFn.apply(
            u_prev, dt * (1.0 - trap_h), A, B_prev, C, torch.zeros_like(D),
            chunk_size, can_use_triton, adt,
        )
        return y_cur + y_prev
    return _ChunkedSSDFn.apply(u, dt, A, B, C, D, chunk_size, can_use_triton, adt)


def chunked_ssd_forward_reference(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Simple (but correct) chunked forward used as a reference for testing.

    Runs the sequential scan within each chunk, propagating state across chunks.
    This is **not** the fast L-matrix path; it is a correctness reference.
    """
    Bsz, T, H, P = u.shape
    N = B.shape[-1]

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    T_pad = T + pad

    state = torch.zeros(Bsz, H, P, N, device=u.device, dtype=torch.float32)
    ys: list[torch.Tensor] = []

    for t in range(T_pad):
        u_t = u[:, t].float()  # [B, H, P]
        dt_t = dt[:, t].float()  # [B, H]
        B_t = B[:, t].float()  # [B, H, N]
        C_t = C[:, t].float()  # [B, H, N]

        abar = torch.exp(
            dt_t.unsqueeze(-1).unsqueeze(-1)
            * A.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        state = abar * state + dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(
            -2
        ) * u_t.unsqueeze(-1)
        y_t = (C_t.unsqueeze(-2) * state).sum(dim=-1) + D.unsqueeze(0) * u_t
        ys.append(y_t.to(u.dtype))

    y = torch.stack(ys, dim=1)
    if pad > 0:
        y = y[:, :T]
    return y


def chunked_ssd_backward_reference(
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
    """Analytic backward for ``chunked_ssd_forward_reference``.

    Uses the sequential reverse-time algorithm (the same sequential diagonal-A
    scan used as the correctness oracle).

    Supports both parametrisations of the per-step log-decay:

    * Mamba2: ``log_decay = dt * A`` (``A`` per-head ``[H]``) — produces ``dA``.
    * Mamba3: ``log_decay = adt`` (time-dependent ``[B, T, H]``) — produces
      ``dadt`` and leaves ``dA`` zero (the passed ``A`` is a dummy).

    ``D`` may be per-head ``[H]`` (Mamba3) or per-head_dim ``[H, P]`` (Mamba2).

    Returns ``(du, ddt, dA, dB, dC, dD, dadt)``.
    """
    if needs_input_grad is None:
        needs_input_grad = (True,) * 6

    use_adt = adt is not None
    Bsz, T, H, P = u.shape
    N = B.shape[-1]

    u32 = u.float()
    dt32 = dt.float()
    A32 = A.float()
    B32 = B.float()
    C32 = C.float()
    D32 = D.float()
    gy32 = grad_y.float()
    adt32 = adt.float() if use_adt else None
    D_per_head = D32.ndim == 1  # [H] vs [H, P]

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        gy32 = F.pad(gy32, (0, 0, 0, 0, 0, pad))
        u32 = F.pad(u32, (0, 0, 0, 0, 0, pad))
        dt32 = F.pad(dt32, (0, 0, 0, pad))
        B32 = F.pad(B32, (0, 0, 0, 0, 0, pad))
        C32 = F.pad(C32, (0, 0, 0, 0, 0, pad))
        if use_adt:
            adt32 = F.pad(adt32, (0, 0, 0, pad))
    T_pad = T + pad

    def _log_decay(t_idx):
        # [B, H, 1, 1]
        if use_adt:
            return adt32[:, t_idx].unsqueeze(-1).unsqueeze(-1)
        return (
            dt32[:, t_idx].unsqueeze(-1).unsqueeze(-1)
            * A32.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )

    state = torch.zeros(Bsz, H, P, N, device=u.device, dtype=torch.float32)
    states_after: list[torch.Tensor] = []

    for t in range(T_pad):
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        abar = torch.exp(_log_decay(t))
        state = abar * state + dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(
            -2
        ) * u_t.unsqueeze(-1)
        states_after.append(state)

    du = torch.zeros_like(u32) if needs_input_grad[0] else None
    ddt = torch.zeros_like(dt32) if needs_input_grad[1] else None
    dA = torch.zeros_like(A32) if (needs_input_grad[2] and not use_adt) else None
    dB = torch.zeros_like(B32) if needs_input_grad[3] else None
    dC = torch.zeros_like(C32) if needs_input_grad[4] else None
    dD = torch.zeros_like(D32) if needs_input_grad[5] else None
    dadt = torch.zeros_like(adt32) if (use_adt and needs_adt_grad) else None

    grad_state = torch.zeros(Bsz, H, P, N, device=u.device, dtype=torch.float32)

    for t in range(T_pad - 1, -1, -1):
        gy_t = gy32[:, t]
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        C_t = C32[:, t]
        state_t = states_after[t]

        state_prev = states_after[t - 1] if t > 0 else torch.zeros_like(state_t)

        if dC is not None:
            dC[:, t] = (gy_t.unsqueeze(-1) * state_t).sum(dim=2)
        # D term: y += D * u.  D is [H,P] (per head_dim) or [H] (per head).
        if dD is not None:
            if D_per_head:
                dD += (gy_t * u_t).sum(dim=(0, 2))  # [H]
            else:
                dD += (gy_t * u_t).sum(dim=0)  # [H, P]
        if du is not None:
            Dterm = D32.unsqueeze(0) if not D_per_head else D32[None, :, None]
            du[:, t] += gy_t * Dterm

        grad_state = grad_state + gy_t.unsqueeze(-1) * C_t.unsqueeze(-2)

        if du is not None:
            du[:, t] += (
                grad_state * dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(-2)
            ).sum(dim=-1)
        if dB is not None:
            dB[:, t] = (
                grad_state * dt_t.unsqueeze(-1).unsqueeze(-1) * u_t.unsqueeze(-1)
            ).sum(dim=2)
        if ddt is not None:
            ddt[:, t] += (grad_state * B_t.unsqueeze(-2) * u_t.unsqueeze(-1)).sum(
                dim=(2, 3)
            )

        abar = torch.exp(_log_decay(t))
        decay_grad = grad_state * state_prev  # [B, H, P, N]
        # d(log_decay) flows from abar = exp(log_decay): grad = decay_grad * abar
        log_decay_grad = (decay_grad * abar).sum(dim=(2, 3))  # [B, H]
        if use_adt:
            if dadt is not None:
                dadt[:, t] += log_decay_grad
        else:
            # log_decay = dt * A
            if dA is not None:
                dA += (log_decay_grad * dt_t).sum(dim=0)  # [H]
            if ddt is not None:
                ddt[:, t] += log_decay_grad * A32.unsqueeze(0)  # [B, H]

        grad_state = grad_state * abar

    out: list[torch.Tensor | None] = []
    tensors = [du, ddt, dA, dB, dC, dD, dadt]
    for tgrad in tensors:
        if tgrad is None:
            out.append(None)
        elif tgrad.ndim in (3, 4):
            out.append(tgrad[:, :T] if pad > 0 else tgrad)
        else:
            out.append(tgrad)
    return tuple(out)  # (du, ddt, dA, dB, dC, dD, dadt)
