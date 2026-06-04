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


def _chunked_ssd_forward_torch(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
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

    Returns:
        y: ``[B, T, H, P]`` — SSM output
    """
    if u.ndim != 4:
        raise ValueError("u must have shape [B, T, H, P]")
    if dt.ndim != 3:
        raise ValueError("dt must have shape [B, T, H]")
    if A.ndim != 1:
        raise ValueError("A must have shape [H] for diagonal-A chunked SSD")
    if B.shape != C.shape or B.ndim != 4:
        raise ValueError("B and C must have matching shape [B, T, H, N]")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    Bsz, T, H, P = u.shape
    N = B.shape[-1]
    if dt.shape != (Bsz, T, H):
        raise ValueError("dt shape must match [B, T, H]")
    if A.shape != (H,):
        raise ValueError("A shape must match [H]")
    if B.shape[:3] != (Bsz, T, H):
        raise ValueError("B and C shape must match [B, T, H, N]")
    if D.shape != (H, P):
        raise ValueError("D shape must match [H, P]")

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
    y = Y_intra + y_inter + D.unsqueeze(0).unsqueeze(0) * u  # [B, nc, C, H, P]

    # ── reshape + trim padding ─────────────────────────────────────
    y = y.reshape(Bsz, T_pad, H, P)
    if pad > 0:
        y = y[:, :T]
    return y.to(out_dtype)


if CHUNKED_SSD_TRITON_AVAILABLE:

    @triton.jit
    def _chunked_ssd_forward_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
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

        a_val = tl.load(A_ptr + h).to(tl.float32)
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

            decay = tl.exp(dt_val * a_val)
            new_state = state * decay + dt_val * u_vals[:, None] * b_vals[None, :]
            state = tl.where(active, new_state, state)

            y_vals = tl.sum(state * c_vals[None, :], axis=1) + d_vals * u_vals
            tl.store(out_ptr + base_bth * P + p_offs, y_vals, mask=active & p_mask)

        tl.store(state_ptr + state_base, state, mask=state_mask)


def chunked_ssd_forward_triton(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Triton chunk kernel for diagonal-A SSD forward.

    The kernel fuses the recurrent work within each chunk and carries only the
    compact boundary state between chunks. Inter-chunk prefix propagation is
    still sequential for now, but the large intra-chunk ``L``/``G`` tensors are
    avoided on CUDA.
    """
    if not CHUNKED_SSD_TRITON_AVAILABLE:
        raise RuntimeError(
            "chunked_ssd_forward_triton called but Triton is unavailable"
        )
    if not (
        u.is_cuda and dt.is_cuda and A.is_cuda and B.is_cuda and C.is_cuda and D.is_cuda
    ):
        raise ValueError("chunked_ssd_forward_triton expects CUDA tensors")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    batch_size, seqlen, num_heads, head_dim = u.shape
    d_state = B.shape[-1]
    if dt.shape != (batch_size, seqlen, num_heads):
        raise ValueError("dt shape must match [B, T, H]")
    if A.shape != (num_heads,):
        raise ValueError("A shape must match [H]")
    if B.shape != (batch_size, seqlen, num_heads, d_state) or C.shape != B.shape:
        raise ValueError("B and C must have shape [B, T, H, N]")
    if D.shape != (num_heads, head_dim):
        raise ValueError("D shape must match [H, P]")
    if d_state > 128:
        raise ValueError("chunked_ssd_forward_triton currently supports d_state <= 128")

    u_contig = u.contiguous()
    dt_contig = dt.contiguous()
    A_contig = A.contiguous()
    B_contig = B.contiguous()
    C_contig = C.contiguous()
    D_contig = D.contiguous()
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
        )
    return out


class _ChunkedSSDFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, B, C, D, chunk_size: int, use_triton: bool):
        ctx.chunk_size = chunk_size
        if use_triton:
            y = chunked_ssd_forward_triton(u, dt, A, B, C, D, chunk_size=chunk_size)
        else:
            y = _chunked_ssd_forward_torch(u, dt, A, B, C, D, chunk_size=chunk_size)
        ctx.save_for_backward(u, dt, A, B, C, D)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, dt, A, B, C, D = ctx.saved_tensors
        grads = chunked_ssd_backward_reference(
            grad_y,
            u,
            dt,
            A,
            B,
            C,
            D,
            chunk_size=ctx.chunk_size,
            needs_input_grad=ctx.needs_input_grad[:6],
        )
        return (*grads, None, None)


def chunked_ssd_forward(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
    use_triton: bool = False,
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
    return _ChunkedSSDFn.apply(u, dt, A, B, C, D, chunk_size, can_use_triton)


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
) -> tuple[torch.Tensor | None, ...]:
    """Analytic backward for ``chunked_ssd_forward_reference``.

    Uses the sequential reverse-time algorithm (same as
    sequential diagonal-A scan used as the correctness oracle.
    """
    if needs_input_grad is None:
        needs_input_grad = (True,) * 6

    Bsz, T, H, P = u.shape
    N = B.shape[-1]

    u32 = u.float()
    dt32 = dt.float()
    A32 = A.float()
    B32 = B.float()
    C32 = C.float()
    D32 = D.float()
    gy32 = grad_y.float()

    pad = (chunk_size - T % chunk_size) % chunk_size
    if pad > 0:
        gy32 = F.pad(gy32, (0, 0, 0, 0, 0, pad))
        u32 = F.pad(u32, (0, 0, 0, 0, 0, pad))
        dt32 = F.pad(dt32, (0, 0, 0, pad))
        B32 = F.pad(B32, (0, 0, 0, 0, 0, pad))
        C32 = F.pad(C32, (0, 0, 0, 0, 0, pad))
    T_pad = T + pad

    state = torch.zeros(Bsz, H, P, N, device=u.device, dtype=torch.float32)
    states_after: list[torch.Tensor] = []

    for t in range(T_pad):
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        abar = torch.exp(
            dt_t.unsqueeze(-1).unsqueeze(-1)
            * A32.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        state = abar * state + dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(
            -2
        ) * u_t.unsqueeze(-1)
        states_after.append(state)

    du = torch.zeros_like(u32) if needs_input_grad[0] else None
    ddt = torch.zeros_like(dt32) if needs_input_grad[1] else None
    dA = torch.zeros_like(A32) if needs_input_grad[2] else None
    dB = torch.zeros_like(B32) if needs_input_grad[3] else None
    dC = torch.zeros_like(C32) if needs_input_grad[4] else None
    dD = torch.zeros_like(D32) if needs_input_grad[5] else None

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
        if dD is not None:
            dD += (gy_t * u_t).sum(dim=0)
        if du is not None:
            du[:, t] += gy_t * D32.unsqueeze(0)

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

        abar = torch.exp(
            dt_t.unsqueeze(-1).unsqueeze(-1)
            * A32.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        decay_grad = grad_state * state_prev
        if dA is not None:
            dA += (decay_grad * abar * dt_t.unsqueeze(-1).unsqueeze(-1)).sum(
                dim=(0, 2, 3)
            )
        if ddt is not None:
            ddt[:, t] += (
                decay_grad * abar * A32.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=(2, 3))

        grad_state = grad_state * abar

    out: list[torch.Tensor | None] = []
    tensors = [du, ddt, dA, dB, dC, dD]
    for t in tensors:
        if t is None:
            out.append(None)
        elif t.ndim in (3, 4):
            out.append(t[:, :T] if pad > 0 else t)
        else:
            out.append(t)
    return tuple(out)  # (du, ddt, dA, dB, dC, dD)
