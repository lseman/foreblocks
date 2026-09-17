"""Modular 3-stage chunked-SSD kernels (chunk-state, state-passing, chunk-scan).

Each stage has a forward/backward pair; ``_chunked_ssd_forward_modular`` and
``_chunked_ssd_backward_modular`` orchestrate them with intermediate reuse,
and ``_ChunkedSSDFnModular``/``chunked_ssd_forward_modular`` expose the
autograd-wrapped public entry point for this path.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .segment_sum import _segment_sum_log
from .torch_backward import _chunked_ssd_backward_torch
from .triton_kernels import (
    chunked_ssd_forward_triton_parallel,
    chunked_ssd_forward_triton_tiled,
)


# ── Stage 1: chunk_state_fwd ──────────────────────────────────────────
# Computes the per-chunk local state end: the accumulated state at the
# end of each chunk, ignoring inter-chunk propagation.
#
#   state_end[c, h, p, n] = sum_j exp(cs_last - cs[j]) * dt[j] * B[c,j,h,n] * u[c,j,h,p]
#
# where cs = cumsum(dtA) along the chunk-time dimension.


def _chunk_state_fwd(
    B_c: torch.Tensor,
    u_c: torch.Tensor,
    dt_c: torch.Tensor,
    cs_last: torch.Tensor,
    cumsum_dtA: torch.Tensor,
) -> torch.Tensor:
    # w[j] = exp(cs_last - cs[j])  [B, nc, C, H]
    w = torch.exp(cs_last - cumsum_dtA)
    # wdt[j] = w[j] * dt[j]  [B, nc, C, H]
    wdt = w * dt_c
    # state_end[c,h,p,n] = sum_j wdt[c,j,h] * B[c,j,h,n] * u[c,j,h,p]
    # wdt[:, :, :, :, None] * B_c → [B, nc, C, H, N]
    # then * u_c[:, :, :, :, :, None] → [B, nc, C, H, N, P]
    # sum over j (dim 2) → [B, nc, H, N, P] → transpose → [B, nc, H, P, N]
    # state_end[c,h,p,n] = sum_j wdt[c,j,h] * B[c,j,h,n] * u[c,j,h,p]
    # wdt: [B,nc,C,H], B_c: [B,nc,C,H,N] → LB: [B,nc,C,H,N]
    LB = wdt[..., None] * B_c  # [B, nc, C, H, N]
    # Expand both to [B,nc,C,H,N,P] then sum over C
    LB_6d = LB[:, :, :, :, :, None]  # [B, nc, C, H, N, 1]
    u_6d = u_c[:, :, :, :, None, :]  # [B, nc, C, H, 1, P]
    LBu = LB_6d * u_6d  # [B, nc, C, H, N, P]
    # sum over j (dim 2): [B, nc, H, N, P] → transpose: [B, nc, H, P, N]
    state_end = LBu.sum(dim=2).transpose(-2, -1)  # [B, nc, H, P, N]
    return state_end


def _chunk_state_bwd(
    g_state_end: torch.Tensor,
    B_c: torch.Tensor,
    u_c: torch.Tensor,
    dt_c: torch.Tensor,
    cumsum_dtA: torch.Tensor,
    cs_last: torch.Tensor,
    needs_dB: bool,
    needs_du: bool,
    needs_ddt: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    w = torch.exp(cs_last - cumsum_dtA)
    wdt = w * dt_c

    g_wdt = torch.einsum(
        "bchpn,bcjhn,bcjhp->bcjh", g_state_end, B_c, u_c
    )  # [B, nc, C, H]

    dB = None
    ddu = None
    ddt_from_state = None

    if needs_dB:
        dB = torch.einsum("bchpn,bcjh,bcjhp->bcjhn", g_state_end, wdt, u_c)

    if needs_du:
        ddu = torch.einsum("bchpn,bcjh,bcjhn->bcjhp", g_state_end, wdt, B_c)

    if needs_ddt:
        # d(wdt)/d(dt) = w, so ddt = g_wdt * w
        ddt_from_state = g_wdt * w  # [B, nc, C, H]

    return dB, ddu, ddt_from_state


# ── Stage 2: state_passing_fwd ────────────────────────────────────────
# Parallel state propagation across chunks using the L-matrix trick.
#
#   S_in[c] = sum_j decay_prefix[c,j] * state_summaries[j]
#
# where:
#   state_summaries = [zero_state, state_end]  (padded at front)
#   decay_prefix[c,j] = exp(segsum_log(chunk_log_decay))
#   chunk_log_decay = cumsum_dtA[:, :, -1, :] (cs_last per chunk)


def _state_passing_fwd(
    state_end: torch.Tensor,
    cumsum_dtA: torch.Tensor,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    Bsz = state_end.shape[0]
    nc = state_end.shape[1]
    H = state_end.shape[2]
    P = state_end.shape[3]
    N = state_end.shape[4]

    # Pad with initial_states (single initial entry) or zero: [B, nc+1, H, P, N]
    # state_end: [B, nc, H, P, N], zero_state: [B, 1, H, P, N]
    if initial_states is not None:
        # initial_states: [B, H, P, N] → [B, 1, H, P, N]
        zero_state = initial_states.unsqueeze(1)  # [B, 1, H, P, N]
    else:
        zero_state = torch.zeros(
            Bsz, 1, H, P, N, device=state_end.device, dtype=state_end.dtype
        )
    state_summaries = torch.cat([zero_state, state_end], dim=1)  # [B, nc+1, H, P, N]

    # chunk_log_decay: cs_last per chunk, [B, H, nc+1] (padded with 0 at front)
    cs_last = cumsum_dtA[:, :, -1, :]  # [B, nc, H]
    chunk_log_decay = F.pad(cs_last.transpose(1, 2), (1, 0))  # [B, H, nc+1]

    # decay_prefix: segsum_log gives [B, H, nc+1, nc+1], transpose to [B, nc+1, nc+1, H]
    # This represents: decay_prefix[j, i] = exp(segsum from j+1 to i)
    decay_prefix = torch.exp(_segment_sum_log(chunk_log_decay)).transpose(
        1, 3
    )  # [B, nc+1, nc+1, H]

    # Apply seq_idx-based boundary reset: zero decay_prefix at seq boundaries
    # so state doesn't leak across sequences.
    if seq_idx is not None:
        T_pad = seq_idx.shape[1]
        cs = T_pad // nc  # chunk_size
        # Get per-chunk seq_idx from per-token seq_idx
        chunk_last_tok = (torch.arange(nc, device=seq_idx.device) + 1) * cs - 1
        chunk_last_tok = torch.clamp(chunk_last_tok, 0, T_pad - 1)  # [nc]
        # seq_per_chunk[b, c] = seq_idx[b, chunk_last_tok[c]]
        seq_per_chunk = seq_idx.gather(
            1, chunk_last_tok[None, :].expand(Bsz, nc)
        )  # [B, nc]
        # Pad with -1 at front (initial state boundary)
        seq_padded = F.pad(seq_per_chunk, (1, 0), value=-1)  # [B, nc+1]
        # Detect boundaries: 1 where seq changes between consecutive chunks
        boundary = (seq_padded[:, 1:] != seq_padded[:, :-1]).float()  # [B, nc]
        # cumsum of boundaries
        b_cumsum = torch.cat(
            [torch.zeros(Bsz, 1, device=seq_idx.device), torch.cumsum(boundary, dim=1)],
            dim=1,
        )  # [B, nc+1]
        # has_boundary[b, j, i] = True iff any boundary in (j, i]
        # = (b_cumsum[b, i] - b_cumsum[b, j]) > 0
        # b_cumsum is [B, nc+1], need outer diff: [B, nc+1, nc+1]
        b_cumsum_i = b_cumsum.unsqueeze(2).expand(
            Bsz, nc + 1, nc + 1
        )  # [B, nc+1, nc+1]
        b_cumsum_j = b_cumsum.unsqueeze(1).expand(
            Bsz, nc + 1, nc + 1
        )  # [B, nc+1, nc+1]
        has_boundary = (b_cumsum_i - b_cumsum_j) > 0  # [B, nc+1, nc+1]
        # Zero out decay_prefix where boundary exists
        decay_prefix = decay_prefix * (~has_boundary.unsqueeze(-1)).to(
            decay_prefix.dtype
        )

    # boundary_all[i] = sum_j decay_prefix[j, i] * state_summaries[j]
    # decay_prefix: [B, nc+1, nc+1, H] = [b, j, i, h]
    # state_summaries: [B, nc+1, H, P, N] = [b, j, h, p, n]
    # → boundary_all: [B, nc+1, H, P, N] = [b, i, h, p, n]
    boundary_all = torch.einsum("bjih,bjhpn->bihpn", decay_prefix, state_summaries)
    states_boundary = boundary_all[:, :-1]  # [B, nc, H, P, N]

    # Final states: the last entry in boundary_all (for KV cache continuation)
    final_states = boundary_all[:, -1:]  # [B, 1, H, P, N]
    return states_boundary, decay_prefix, chunk_log_decay, state_summaries, final_states


def _state_passing_bwd(
    g_states_boundary: torch.Tensor,
    decay_prefix: torch.Tensor,
    chunk_log_decay: torch.Tensor,
    state_summaries: torch.Tensor,
    cumsum_dtA: torch.Tensor,
    cs_last: torch.Tensor,
    dfinal_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    Bsz = g_states_boundary.shape[0]
    nc = g_states_boundary.shape[1]
    H = g_states_boundary.shape[2]
    P = g_states_boundary.shape[3]
    N = g_states_boundary.shape[4]

    # Combine g_states_boundary with dfinal_states
    # g_boundary: [B, nc+1, H, P, N] where last entry is dfinal_states or zero
    if dfinal_states is not None:
        g_boundary = torch.cat(
            [g_states_boundary, dfinal_states], dim=1
        )  # [B, nc+1, H, P, N]
    else:
        g_boundary = F.pad(g_states_boundary, (0, 0, 0, 0, 0, 0, 0, 1))

    # d(state_summaries[j]) = sum_i decay_prefix[j, i] * g_boundary[i]
    g_state_summaries = torch.einsum(
        "bjih,bihpn->bjhpn", decay_prefix, g_boundary
    )  # [B, nc+1, H, P, N]

    # g_state_end = g_state_summaries[:, 1:]  (skip the initial state entry)
    g_state_end = g_state_summaries[:, 1:]  # [B, nc, H, P, N]

    # Gradient of initial states
    g_initial_states = (
        g_state_summaries[:, 0] if dfinal_states is not None else None
    )  # [B, H, P, N]

    # d(decay_prefix[j, i]) = sum_{h,p,n} g_boundary[i] * state_summaries[j]
    g_decay_prefix = torch.einsum(
        "bihpn,bjhpn->bjih", g_boundary, state_summaries
    )  # [B, nc+1, nc+1, H]

    # gseg[j, i, h] = g_decay_prefix[j, i, h] * decay_prefix[j, i, h]
    gseg = g_decay_prefix * decay_prefix  # [B, nc+1, nc+1, H]

    # d(cld[k]) = sum_{j < k <= i} gseg[j, i, h]
    ncp = chunk_log_decay.shape[-1]  # nc + 1
    idx = torch.arange(ncp, device=chunk_log_decay.device)
    kk = idx[:, None, None]  # [k, 1, 1]
    ii = idx[None, None, :]  # [1, 1, i]
    jj = idx[None, :, None]  # [1, j, 1]
    seg_mask = ((jj < kk) & (kk <= ii)).to(gseg.dtype)  # [k, j, i]

    g_cld = torch.einsum("bjih,kji->bhk", gseg, seg_mask)  # [B, H, nc+1]
    d_cs_last = g_cld[:, :, 1:].transpose(1, 2)  # [B, nc, H]

    return g_state_end, d_cs_last, g_initial_states


# ── Stage 3: chunk_scan_fwd ───────────────────────────────────────────
# Intra-chunk scan: computes output from L-matrix + propagated states.


def _chunk_scan_fwd(
    L: torch.Tensor,
    dt_c: torch.Tensor,
    G: torch.Tensor,
    u_c: torch.Tensor,
    states_boundary: torch.Tensor,
    C_c: torch.Tensor,
    cumsum_dtA: torch.Tensor,
    D: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    # Y_intra = sum_j L[c,t,j] * dt[j] * G[c,t,j] * u[c,j]
    LdtG = L * dt_c[:, :, None, :, :] * G  # [B, nc, C, C, H]
    Y_intra = torch.einsum("bctjh,bcjhp->bcthp", LdtG, u_c)  # [B, nc, C, H, P]

    # y_inter: propagate states_boundary through C
    decay_from_start = torch.exp(cumsum_dtA)  # [B, nc, C, H]
    state_entered = states_boundary.unsqueeze(2) * decay_from_start.unsqueeze(
        -1
    ).unsqueeze(-1)  # [B, nc, C, H, P, N]
    y_inter = torch.einsum(
        "bcthn,bcthpn->bcthp", C_c, state_entered
    )  # [B, nc, C, H, P]

    # Total output
    if D.ndim == 2:
        y = Y_intra + y_inter + D.unsqueeze(0).unsqueeze(0) * u_c
    else:
        y = Y_intra + y_inter + D[:, None] * u_c  # [B, nc, C, H, P]

    intermediates = {
        "L": L,
        "G": G,
        "LdtG": LdtG,
        "dt_c": dt_c,
        "decay_from_start": decay_from_start,
        "state_entered": state_entered,
        "states_boundary": states_boundary,
        "C_c": C_c,
        "cumsum_dtA": cumsum_dtA,
        "gy_c_needed": True,  # flag that gy_c is available for G backward
    }
    return y, intermediates


def _chunk_scan_bwd(
    gy_c: torch.Tensor,
    intermediates: dict,
    D: torch.Tensor,
    needs_dC: bool,
    needs_dS_in: bool,
    needs_ddt: bool,
    needs_du: bool,
) -> tuple[
    torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None
]:
    L = intermediates["L"]
    G = intermediates["G"]
    LdtG = intermediates["LdtG"]
    dt_c = intermediates["dt_c"]
    u_c = intermediates["u_c"]
    decay_from_start = intermediates["decay_from_start"]
    state_entered = intermediates["state_entered"]
    states_boundary = intermediates["states_boundary"]
    C_c = intermediates["C_c"]
    cumsum_dtA = intermediates["cumsum_dtA"]

    # 1) y_inter backward: y_inter = einsum("bcthn,bcthpn->bcthp", C_c, state_entered)
    a_t = decay_from_start  # [B, nc, C, H]
    gy_a = gy_c * a_t.unsqueeze(-1)  # [B, nc, C, H, P]

    g_states_boundary = None
    g_C_inter = None
    dcs_from_inter = None

    if needs_dS_in:
        g_states_boundary = torch.einsum(
            "bcthp,bcthn->bchpn", gy_a, C_c
        )  # [B, nc, H, P, N]
    if needs_dC:
        g_C_inter = torch.einsum(
            "bcthp,bchpn->bcthn", gy_a, states_boundary
        )  # [B, nc, C, H, N]

    # dcs from y_inter: sum_{p,n} a[t] * gy[t,p] * C[t,n] * S_in[p,n]
    if needs_dS_in:
        dcs_from_inter = torch.einsum(
            "bcthp,bcthn,bchpn->bcth", gy_a, C_c, states_boundary
        )

    # 2) Y_intra backward: Y_intra = einsum("bctjh,bcjhp->bcthp", LdtG, u_c)
    g_ddt_intra = None
    g_du_intra = None
    dcs_from_intra = None

    if needs_du or needs_ddt:
        # gY_intra = gy_c
        # du[j] += sum_t LdtG[t,j] * gy[t]
        if needs_du:
            g_du_intra = torch.einsum(
                "bctjh,bcthp->bcjhp", LdtG, gy_c
            )  # [B, nc, C, H, P]

        # d(LdtG)[t,j] = sum_p gy[t,p] * u[j,p]
        gLdtG = torch.einsum("bcthp,bcjhp->bctjh", gy_c, u_c)  # [B, nc, Ct, Cj, H]

        # ddt from LdtG: ddt[j] += sum_t gLdtG[t,j] * L[t,j] * G[t,j]
        if needs_ddt:
            g_ddt_intra = (gLdtG * L * G).sum(dim=2)  # [B, nc, C, H] — sum over t

        # d(L[t,j]): gL = gLdtG * dt_c * G
        Ldt = L * dt_c[:, :, None, :, :]
        gL = gLdtG * dt_c[:, :, None, :, :] * G  # [B, nc, Ct, Cj, H]
        # d(cs[t]) += sum_j gL*L, d(cs[j]) -= sum_t gL*L
        gLL = gL * L  # [B, nc, Ct, Cj, H]
        dcs_from_intra = gLL.sum(dim=3) - gLL.sum(dim=2)  # [B, nc, Ct, H]

    return g_states_boundary, g_C_inter, g_ddt_intra, dcs_from_inter, dcs_from_intra


# ── Unified forward with intermediate saving ──────────────────────────


def _chunked_ssd_forward_modular(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    adt: torch.Tensor | None = None,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    Bsz, T, H, P = u.shape
    N = B.shape[-1]

    out_dtype = u.dtype
    u = u.float()
    dt = dt.float()
    A = A.float()
    B = B.float()
    C = C.float()
    D = D.float()

    # Pad to chunk boundary
    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        u = F.pad(u, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    T_pad = T + pad
    nc = T_pad // chunk_size

    # dtA
    if adt is not None:
        dtA = adt.float()
        if pad > 0:
            dtA = F.pad(dtA, (0, 0, 0, pad))
    else:
        dtA = dt * A

    # Reshape to chunks
    dtA_c = dtA.view(Bsz, nc, chunk_size, H)
    dt_c = dt.view(Bsz, nc, chunk_size, H)
    u_c = u.view(Bsz, nc, chunk_size, H, P)
    B_c = B.view(Bsz, nc, chunk_size, H, N)
    C_c = C.view(Bsz, nc, chunk_size, H, N)

    # Cumsum
    cumsum_dtA = torch.cumsum(dtA_c, dim=2)  # [B, nc, C, H]

    # ── Stage 1: chunk_state_fwd ────────────────────────────────────
    cs_last = cumsum_dtA[:, :, -1:, :]  # [B, nc, 1, H]
    state_end = _chunk_state_fwd(
        B_c, u_c, dt_c, cs_last, cumsum_dtA
    )  # [B, nc, H, P, N]

    # ── Stage 2: state_passing_fwd ──────────────────────────────────
    states_boundary, decay_prefix, chunk_log_decay, state_summaries, final_states = (
        _state_passing_fwd(state_end, cumsum_dtA, seq_idx, initial_states)
    )

    # ── Stage 3: chunk_scan_fwd ─────────────────────────────────────
    # L matrix: L[t,j] = exp(cs[t] - cs[j]) for j <= t
    L_diff = cumsum_dtA.unsqueeze(-2) - cumsum_dtA.unsqueeze(-3)  # [B, nc, C, C, H]
    tril = torch.tril(
        torch.ones(chunk_size, chunk_size, device=u.device, dtype=torch.bool)
    )
    L_diff = L_diff.masked_fill(
        ~tril.unsqueeze(0).unsqueeze(0).unsqueeze(-1), float("-inf")
    )
    L = torch.exp(L_diff)  # [B, nc, C, C, H]

    # G matrix: G[t,j] = sum_n C[t,n] * B[j,n]
    G = (C_c.unsqueeze(3) * B_c.unsqueeze(2)).sum(dim=-1)  # [B, nc, C, C, H]

    # ── seq_idx masking: zero L, G where sequence boundaries differ ─
    if seq_idx is not None:
        t_idx = torch.arange(chunk_size, device=u.device)  # [C]
        t_abs = (nc * chunk_size) + t_idx  # absolute token indices in this chunk
        seq_t = seq_idx[:, t_abs].unsqueeze(2)  # [B, C, 1]
        seq_j = seq_idx[:, t_abs].unsqueeze(1)  # [B, 1, C]
        same_seq = seq_t == seq_j  # [B, C, C]
        L = torch.where(same_seq[:, :, :, None].unsqueeze(0), L, torch.zeros_like(L))  # type: ignore[assignment]
        G = torch.where(same_seq.unsqueeze(0).unsqueeze(-1), G, torch.zeros_like(G))  # type: ignore[assignment]

    y_chunked, scan_intermediates = _chunk_scan_fwd(
        L, dt_c, G, u_c, states_boundary, C_c, cumsum_dtA, D
    )

    # Reshape + trim padding
    y = y_chunked.reshape(Bsz, T_pad, H, P)
    if pad > 0:
        y = y[:, :T]

    intermediates = {
        "L": L,
        "G": G,
        "decay_prefix": decay_prefix,
        "chunk_log_decay": chunk_log_decay,
        "state_summaries": state_summaries,
        "cumsum_dtA": cumsum_dtA,
        "cs_last": cs_last,
        "dt_c": dt_c,
        "u_c": u_c,
        "B_c": B_c,
        "C_c": C_c,
        "pad": pad,
        "Bsz": Bsz,
        "T": T,
        "T_pad": T_pad,
        "H": H,
        "P": P,
        "N": N,
        "nc": nc,
        "chunk_size": chunk_size,
        "final_states": final_states,
        **scan_intermediates,
    }

    return y.to(out_dtype), intermediates


# ── Unified backward with intermediate reuse ──────────────────────────


def _chunked_ssd_backward_modular(
    grad_y: torch.Tensor,
    intermediates: dict,
    A: torch.Tensor,
    D: torch.Tensor,
    adt: torch.Tensor | None,
    needs_input_grad: tuple[bool, ...] | None = None,
    needs_adt_grad: bool = False,
) -> tuple[torch.Tensor | None, ...]:
    if needs_input_grad is None:
        needs_input_grad = (True,) * 6

    use_adt = adt is not None

    # Unpack intermediates
    L = intermediates["L"]
    G = intermediates["G"]
    decay_prefix = intermediates["decay_prefix"]
    chunk_log_decay = intermediates["chunk_log_decay"]
    state_summaries = intermediates["state_summaries"]
    cumsum_dtA = intermediates["cumsum_dtA"]
    cs_last = intermediates["cs_last"]
    dt_c = intermediates["dt_c"]
    u_c = intermediates["u_c"]
    B_c = intermediates["B_c"]
    C_c = intermediates["C_c"]
    states_boundary = intermediates["states_boundary"]
    decay_from_start = intermediates["decay_from_start"]
    state_entered = intermediates["state_entered"]
    scan_inter = {
        k: intermediates[k]
        for k in [
            "L",
            "G",
            "LdtG",
            "dt_c",
            "decay_from_start",
            "state_entered",
            "states_boundary",
            "C_c",
            "cumsum_dtA",
            "u_c",
        ]
    }

    pad = intermediates["pad"]
    Bsz = intermediates["Bsz"]
    T = intermediates["T"]
    T_pad = intermediates.get("T_pad", T)
    H = intermediates["H"]
    P = intermediates["P"]
    nc = intermediates["nc"]
    chunk_size = intermediates["chunk_size"]
    pad = intermediates.get("pad", 0)

    # grad_y is [B, T, H, P]; pad to [B, T_pad, H, P] if needed
    gy = grad_y
    if pad > 0:
        gy = F.pad(gy.float(), (0, 0, 0, 0, 0, pad))  # pad last dim (T)
    gy_c = gy.reshape(Bsz, nc, chunk_size, H, P)
    D_per_head = D.ndim == 1

    # ── Step 1: chunk_scan_bwd ──────────────────────────────────────
    g_states_boundary, g_C_inter, g_ddt_intra, dcs_from_inter, dcs_from_intra = (
        _chunk_scan_bwd(
            gy_c,
            scan_inter,
            D,
            needs_dC=needs_input_grad[4],
            needs_dS_in=True,
            needs_ddt=needs_input_grad[1],
            needs_du=True,
        )
    )

    # ── Step 2: state_passing_bwd ───────────────────────────────────
    dfinal = intermediates.get("dfinal_states")
    g_state_end, d_cs_last, g_initial_states = _state_passing_bwd(
        g_states_boundary,
        decay_prefix,
        chunk_log_decay,
        state_summaries,
        cumsum_dtA,
        cs_last,
        dfinal_states=dfinal,
    )
    # Save g_initial_states for initial_states gradient
    if g_initial_states is not None:
        intermediates["g_initial_states"] = g_initial_states

    # ── Step 3: chunk_state_bwd ─────────────────────────────────────
    dB_chunk_state, du_chunk_state, ddt_chunk_state = _chunk_state_bwd(
        g_state_end,
        B_c,
        u_c,
        dt_c,
        cumsum_dtA,
        cs_last,
        needs_dB=needs_input_grad[3],
        needs_du=needs_input_grad[0],
        needs_ddt=needs_input_grad[1],
    )

    # ── Step 4: Assemble gradients ──────────────────────────────────

    # dC: from scan backward + from G backward (G = C·B)
    dC = g_C_inter if g_C_inter is not None else torch.zeros_like(C_c)
    # Need gG = gLdtG * Ldt where gLdtG = einsum(gy, u)
    gLdtG_scan = torch.einsum("bcthp,bcjhp->bctjh", gy_c, u_c)
    Ldt = L * dt_c[:, :, None, :, :]
    gG = gLdtG_scan * Ldt
    dC = dC + torch.einsum("bctjh,bcjhn->bcthn", gG, B_c)

    # dB: from chunk_state_bwd + from G backward
    dB = dB_chunk_state if dB_chunk_state is not None else torch.zeros_like(B_c)
    dB = dB + torch.einsum("bctjh,bcthn->bcjhn", gG, C_c)

    # ddt: from scan backward + from chunk_state_bwd
    ddt = torch.zeros_like(dt_c) if needs_input_grad[1] else None
    if ddt is not None:
        if g_ddt_intra is not None:
            ddt = ddt + g_ddt_intra
        if ddt_chunk_state is not None:
            ddt = ddt + ddt_chunk_state

    # du: from scan backward + from chunk_state_bwd + from D*u
    Dexp = D[None, None, None, :, None] if D_per_head else D[None, None, None, :, :]
    du = torch.zeros_like(u_c) if needs_input_grad[0] else None
    if du is not None:
        du = du + gy_c * Dexp  # D*u term
        # du from intra: already handled via gLdtG chain
        # du from LdtG: du[j] += sum_t LdtG[t,j] * gy[t]
        du = du + torch.einsum(
            "bctjh,bcthp->bcjhp", L * dt_c[:, :, None, :, :] * G, gy_c
        )
        if du_chunk_state is not None:
            du = du + du_chunk_state

    # dA / dadt: from dcs (cumsum adjoint = reverse cumsum)
    # Gather all dcs contributions
    dcs = dcs_from_intra if dcs_from_intra is not None else torch.zeros_like(cumsum_dtA)
    if dcs_from_inter is not None:
        dcs = dcs + dcs_from_inter

    # dcs from chunk_state: w[j] = exp(cs_last - cs[j]), ddt_chunk_state = g_wdt * w
    # gw_w = g_w * w = g_wdt * dt_c * w = ddt_chunk_state * dt_c
    w = torch.exp(cs_last - cumsum_dtA)
    if ddt_chunk_state is not None:
        gw_w = ddt_chunk_state * dt_c  # [B, nc, C, H]  (safe, no division)
        # d(cs_last) += sum_j gw_w[j] over chunk-time dim (dim 2)
        d_cs_last_from_state = gw_w.sum(dim=2)  # [B, nc, H]
        # d(cs[j]) -= gw_w[j] for all j in chunk
        dcs = dcs - gw_w  # [B, nc, C, H]

    # d_cs_last total: from state_passing + from chunk_state
    d_cs_last_total = d_cs_last
    if ddt_chunk_state is not None:
        d_cs_last_total = d_cs_last + d_cs_last_from_state

    # Add d_cs_last_total to last token of each chunk
    dcs[:, :, -1, :] = dcs[:, :, -1, :] + d_cs_last_total

    # Adjoint of cumsum = reverse cumsum.
    # reverse_cumsum(dcs)[k] = sum_{j=k}^{nc-1} dcs[j]
    # Stable form: total_dcs - cumsum_dcs[k-1], with cumsum shifted.
    total_dcs = dcs.sum(dim=2, keepdim=True)  # [B, nc, 1, H]
    cumsum_dcs = torch.cumsum(dcs, dim=2)  # [B, nc, C, H]
    # reverse_cumsum[k] = total - cumsum[k-1], with cumsum[-1]=0
    # = total - cumsum + dcs  (since cumsum[k] = cumsum[k-1] + dcs[k])
    d_dtA = total_dcs - cumsum_dcs + dcs

    # Split into dA and ddt (Mamba2) or dadt (Mamba3)
    dA = None
    dadt = None
    if use_adt:
        if needs_adt_grad:
            dadt = d_dtA
    else:
        if ddt is not None:
            ddt = ddt + d_dtA * A[None, None, None, :]
        if needs_input_grad[2]:
            dA = (d_dtA * dt_c).sum(dim=(0, 1, 2))  # [H]

    # ── Unchunk + trim padding ──────────────────────────────────────
    def _unchunk(t, n_dims):
        if t is None:
            return None
        t = t.reshape((Bsz, T_pad) + t.shape[3:])
        if pad > 0:
            return t[:, :T]
        return t

    du_out = _unchunk(du, 4) if du is not None else None
    ddt_out = _unchunk(ddt, 3) if ddt is not None else None
    dB_out = _unchunk(dB, 4) if dB is not None else None
    dC_out = _unchunk(dC, 4) if dC is not None else None
    dadt_out = _unchunk(dadt, 3) if dadt is not None else None

    out_dtype = grad_y.dtype
    cast = lambda x: None if x is None else x.to(out_dtype)

    # dD
    dD = None
    if needs_input_grad[5]:
        if D_per_head:
            dD = (gy_c * u_c).sum(dim=(0, 1, 2, 4))  # [H]
        else:
            dD = (gy_c * u_c).sum(dim=(0, 1, 2))  # [H, P]

    return (
        cast(du_out),
        cast(ddt_out),
        dA.to(out_dtype) if (needs_input_grad[2] and dA is not None) else None,
        cast(dB_out),
        cast(dC_out),
        cast(dD),
        cast(dadt_out),
    )


# ── Autograd Function ─────────────────────────────────────────────────


class _ChunkedSSDFnModular(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        u,
        dt,
        A,
        B,
        C,
        D,
        chunk_size: int,
        use_triton: bool,
        use_parallel: bool,
        adt=None,
        seq_idx=None,
        initial_states=None,
        dfinal_states=None,
    ):
        ctx.chunk_size = chunk_size
        ctx.use_triton = use_triton
        ctx.seq_idx = seq_idx
        ctx.initial_states = initial_states
        ctx.needs_input_grad = (
            u.requires_grad,
            dt.requires_grad,
            A.requires_grad,
            B.requires_grad,
            C.requires_grad,
            D.requires_grad,
        )
        ctx.needs_adt_grad = adt.requires_grad if adt is not None else False

        if use_triton:
            if use_parallel == "direct":
                fwd = chunked_ssd_forward_triton_parallel
            elif use_parallel:
                fwd = chunked_ssd_forward_triton_tiled
            else:
                pass

            if use_parallel == "direct" or use_parallel:
                y = fwd(u, dt, A, B, C, D, chunk_size=chunk_size, adt=adt)
            else:
                y, intermediates = _chunked_ssd_forward_modular(
                    u,
                    dt,
                    A,
                    B,
                    C,
                    D,
                    chunk_size=chunk_size,
                    adt=adt,
                    seq_idx=seq_idx,
                    initial_states=initial_states,
                )
                ctx.save_for_backward(u, dt, A, B, C, D, adt)
                ctx._intermediates = intermediates
                return y

            ctx.save_for_backward(u, dt, A, B, C, D, adt)
            return y
        else:
            y, intermediates = _chunked_ssd_forward_modular(
                u,
                dt,
                A,
                B,
                C,
                D,
                chunk_size=chunk_size,
                adt=adt,
                seq_idx=seq_idx,
                initial_states=initial_states,
            )
            ctx.save_for_backward(u, dt, A, B, C, D, adt)
            ctx._intermediates = intermediates
            # Store dfinal_states for backward
            if dfinal_states is not None:
                intermediates["dfinal_states"] = dfinal_states
            return y

    @staticmethod
    def backward(ctx, grad_y):
        if hasattr(ctx, "_intermediates") and ctx._intermediates is not None:
            u, dt, A, B, C, D, adt = ctx.saved_tensors
            grads = _chunked_ssd_backward_modular(
                grad_y,
                ctx._intermediates,
                A,
                D,
                adt,
                needs_input_grad=ctx.needs_input_grad,
                needs_adt_grad=ctx.needs_adt_grad,
            )
            du, ddt, dA, dB, dC, dD, dadt = grads
            # dfinal_states gradient: forward output of final_states
            final_states_grad = None
            if hasattr(ctx, "dfinal_states") and ctx.dfinal_states is not None:
                # grad_y needs to be trimmed for final_states gradient
                pass  # dfinal_states grad would come from external KV cache
            return (
                du,
                ddt,
                dA,
                dB,
                dC,
                dD,
                None,
                None,
                None,
                dadt,
                final_states_grad,
                None,
                None,
            )
        else:
            u, dt, A, B, C, D, adt = ctx.saved_tensors
            # always use vectorized torch backward — Triton bwd is O(chunks) sequential
            bwd = _chunked_ssd_backward_torch
            grads = bwd(
                grad_y,
                u,
                dt,
                A,
                B,
                C,
                D,
                chunk_size=ctx.chunk_size,
                needs_input_grad=ctx.needs_input_grad,
                adt=adt,
                needs_adt_grad=ctx.needs_adt_grad,
            )
            du, ddt, dA, dB, dC, dD, dadt = grads
            return (du, ddt, dA, dB, dC, dD, None, None, None, dadt, None, None, None)


# ── Public API ────────────────────────────────────────────────────────


def chunked_ssd_forward_modular(
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
    parallel: bool | str = False,
    seq_idx: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    dfinal_states: torch.Tensor | None = None,
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
        if adt is None:
            raise ValueError("trapezoidal discretisation requires adt (Mamba3)")
        T = u.shape[1]
        y_cur = chunked_ssd_forward_modular(
            u,
            dt * trap,
            A,
            B,
            C,
            D,
            chunk_size,
            use_triton,
            adt,
            None,
            parallel,
            seq_idx=seq_idx,
            initial_states=initial_states,
        )
        B_prev = F.pad(B, (0, 0, 0, 0, 1, 0))[:, :T]
        u_prev = F.pad(u, (0, 0, 0, 0, 1, 0))[:, :T]
        y_prev = chunked_ssd_forward_modular(
            u_prev,
            dt * (1.0 - trap),
            A,
            B_prev,
            C,
            torch.zeros_like(D),
            chunk_size,
            use_triton,
            adt,
            None,
            parallel,
            seq_idx=seq_idx,
            initial_states=initial_states,
        )
        return y_cur + y_prev

    return _ChunkedSSDFnModular.apply(
        u,
        dt,
        A,
        B,
        C,
        D,
        chunk_size,
        can_use_triton,
        parallel,
        adt,
        seq_idx,
        initial_states,
        dfinal_states,
    )


