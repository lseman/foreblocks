from __future__ import annotations

import torch


try:
    import triton
    import triton.language as tl

    GROUPED_SSD_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    GROUPED_SSD_TRITON_AVAILABLE = False


def grouped_ssd_scan_reference(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    delta_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if u.ndim != 4:
        raise ValueError("u must have shape [B, T, H, P]")
    if dt.ndim != 3:
        raise ValueError("dt must have shape [B, T, H]")
    if A.ndim != 2:
        raise ValueError("A must have shape [H, N]")
    if Bpar.shape != Cpar.shape:
        raise ValueError("Bpar and Cpar must have the same shape")
    if Bpar.ndim != 4:
        raise ValueError("Bpar and Cpar must have shape [B, T, H, N]")

    batch_size, seqlen, num_heads, head_dim = u.shape
    _, _, b_heads, d_state = Bpar.shape
    if b_heads != num_heads:
        raise ValueError("head dimension mismatch between u and Bpar/Cpar")
    if dt.shape != (batch_size, seqlen, num_heads):
        raise ValueError("dt shape must match [B, T, H]")
    if A.shape != (num_heads, d_state):
        raise ValueError("A shape must match [H, N]")
    if Dskip.shape != (num_heads, head_dim):
        raise ValueError("Dskip shape must match [H, P]")
    if delta_gate is not None and delta_gate.shape != (batch_size, seqlen, num_heads):
        raise ValueError("delta_gate must have shape [B, T, H]")

    state = torch.zeros(
        batch_size, num_heads, head_dim, d_state, device=u.device, dtype=torch.float32
    )
    ys = []
    A32 = A.float()
    D32 = Dskip.float()

    for t in range(seqlen):
        u_t = u[:, t].float()
        dt_t = dt[:, t].float()
        B_t = Bpar[:, t].float()
        C_t = Cpar[:, t].float()

        decay = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        decayed_state = decay.unsqueeze(-2) * state
        delta = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(-2) * u_t.unsqueeze(-1)

        if delta_gate is not None:
            gate_t = torch.sigmoid(delta_gate[:, t].float()).unsqueeze(-1).unsqueeze(-1)
            state = decayed_state + gate_t * delta
        else:
            state = decayed_state + delta

        y_t = (C_t.unsqueeze(-2) * state).sum(dim=-1) + D32.unsqueeze(0) * u_t
        ys.append(y_t.to(u.dtype))

    return torch.stack(ys, dim=1)


def grouped_ssd_scan_backward_reference(
    grad_y: torch.Tensor,
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    delta_gate: torch.Tensor | None = None,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool, bool] | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """Portable analytical backward for grouped_ssd_scan_reference."""
    if needs_input_grad is None:
        needs_input_grad = (True, True, True, True, True, True, delta_gate is not None)

    batch_size, seqlen, num_heads, head_dim = u.shape
    d_state = A.shape[1]
    device = u.device

    u32 = u.float()
    dt32 = dt.float()
    A32 = A.float()
    B32 = Bpar.float()
    C32 = Cpar.float()
    D32 = Dskip.float()
    gate32 = delta_gate.float() if delta_gate is not None else None

    state = torch.zeros(
        batch_size, num_heads, head_dim, d_state, device=device, dtype=torch.float32
    )
    states_before: list[torch.Tensor] = []
    states_after: list[torch.Tensor] = []

    for t in range(seqlen):
        states_before.append(state)
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        decay = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        delta = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(-2) * u_t.unsqueeze(-1)
        if gate32 is not None:
            gate_t = torch.sigmoid(gate32[:, t]).unsqueeze(-1).unsqueeze(-1)
            state = decay.unsqueeze(-2) * state + gate_t * delta
        else:
            state = decay.unsqueeze(-2) * state + delta
        states_after.append(state)

    du = torch.zeros_like(u32) if needs_input_grad[0] else None
    ddt = torch.zeros_like(dt32) if needs_input_grad[1] else None
    dA = torch.zeros_like(A32) if needs_input_grad[2] else None
    dB = torch.zeros_like(B32) if needs_input_grad[3] else None
    dC = torch.zeros_like(C32) if needs_input_grad[4] else None
    dD = torch.zeros_like(D32) if needs_input_grad[5] else None
    dgate = (
        torch.zeros_like(gate32)
        if gate32 is not None and needs_input_grad[6]
        else None
    )

    grad_state = torch.zeros(
        batch_size, num_heads, head_dim, d_state, device=device, dtype=torch.float32
    )

    for t in range(seqlen - 1, -1, -1):
        gy_t = grad_y[:, t].float()
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        C_t = C32[:, t]
        state_prev = states_before[t]
        state_t = states_after[t]

        if dC is not None:
            dC[:, t] = (gy_t.unsqueeze(-1) * state_t).sum(dim=2)
        if dD is not None:
            dD += (gy_t * u_t).sum(dim=0)
        if du is not None:
            du[:, t] += gy_t * D32.unsqueeze(0)

        grad_state = grad_state + gy_t.unsqueeze(-1) * C_t.unsqueeze(-2)

        decay = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        delta_base = dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(-2) * u_t.unsqueeze(-1)
        if gate32 is not None:
            gate_sig = torch.sigmoid(gate32[:, t])
            gate_factor = gate_sig.unsqueeze(-1).unsqueeze(-1)
            delta_grad = grad_state * gate_factor
            if dgate is not None:
                dgate[:, t] = (
                    grad_state * delta_base * gate_factor * (1.0 - gate_factor)
                ).sum(dim=(2, 3))
        else:
            delta_grad = grad_state

        if du is not None:
            du[:, t] += (
                delta_grad * dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(-2)
            ).sum(dim=-1)
        if dB is not None:
            dB[:, t] = (
                delta_grad * dt_t.unsqueeze(-1).unsqueeze(-1) * u_t.unsqueeze(-1)
            ).sum(dim=2)
        if ddt is not None:
            ddt[:, t] += (
                delta_grad * B_t.unsqueeze(-2) * u_t.unsqueeze(-1)
            ).sum(dim=(2, 3))

        decay_grad = grad_state * state_prev
        if dA is not None:
            dA += (
                decay_grad
                * decay.unsqueeze(-2)
                * dt_t.unsqueeze(-1).unsqueeze(-1)
            ).sum(dim=(0, 2))
        if ddt is not None:
            ddt[:, t] += (
                decay_grad * decay.unsqueeze(-2) * A32.unsqueeze(0).unsqueeze(-2)
            ).sum(dim=(2, 3))

        grad_state = grad_state * decay.unsqueeze(-2)

    return (
        du.to(u.dtype) if du is not None else None,
        ddt.to(dt.dtype) if ddt is not None else None,
        dA.to(A.dtype) if dA is not None else None,
        dB.to(Bpar.dtype) if dB is not None else None,
        dC.to(Cpar.dtype) if dC is not None else None,
        dD.to(Dskip.dtype) if dD is not None else None,
        dgate.to(delta_gate.dtype)
        if dgate is not None and delta_gate is not None
        else None,
    )


if GROUPED_SSD_TRITON_AVAILABLE:

    @triton.jit
    def _grouped_ssd_scan_kernel(
        u_ptr,
        dt_ptr,
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        gate_ptr,
        out_ptr,
        T,
        H,
        P,
        N,
        USE_GATE: tl.constexpr,
        BLOCK_P: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_bh = tl.program_id(axis=0)
        pid_p = tl.program_id(axis=1)

        b = pid_bh // H
        h = pid_bh % H

        p_offs = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)
        n_offs = tl.arange(0, BLOCK_N)
        p_mask = p_offs < P
        n_mask = n_offs < N

        a_ptrs = A_ptr + h * N + n_offs
        a_vals = tl.load(a_ptrs, mask=n_mask, other=0.0).to(tl.float32)

        d_ptrs = D_ptr + h * P + p_offs
        d_vals = tl.load(d_ptrs, mask=p_mask, other=0.0).to(tl.float32)

        state = tl.zeros((BLOCK_P, BLOCK_N), dtype=tl.float32)

        for t in tl.range(0, T):
            base_bth = (b * T + t) * H + h

            u_ptrs = u_ptr + base_bth * P + p_offs
            u_vals = tl.load(u_ptrs, mask=p_mask, other=0.0).to(tl.float32)

            dt_val = tl.load(dt_ptr + base_bth).to(tl.float32)
            b_vals = tl.load(B_ptr + base_bth * N + n_offs, mask=n_mask, other=0.0).to(
                tl.float32
            )
            c_vals = tl.load(C_ptr + base_bth * N + n_offs, mask=n_mask, other=0.0).to(
                tl.float32
            )

            decay = tl.exp(dt_val * a_vals)
            decayed_state = state * decay[None, :]
            delta = dt_val * u_vals[:, None] * b_vals[None, :]

            if USE_GATE:
                gate_raw = tl.load(gate_ptr + base_bth).to(tl.float32)
                gate = 1.0 / (1.0 + tl.exp(-gate_raw))
                state = decayed_state + gate * delta
            else:
                state = decayed_state + delta

            y_vals = tl.sum(state * c_vals[None, :], axis=1) + d_vals * u_vals
            out_ptrs = out_ptr + base_bth * P + p_offs
            tl.store(out_ptrs, y_vals, mask=p_mask)


def grouped_ssd_scan_triton(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    delta_gate: torch.Tensor | None = None,
) -> torch.Tensor:
    if not GROUPED_SSD_TRITON_AVAILABLE:
        raise RuntimeError("grouped_ssd_scan_triton called but Triton is not available")
    if not (
        u.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and Bpar.is_cuda
        and Cpar.is_cuda
        and Dskip.is_cuda
    ):
        raise ValueError("grouped_ssd_scan_triton expects CUDA tensors")
    if delta_gate is not None and not delta_gate.is_cuda:
        raise ValueError("delta_gate must be CUDA when provided")

    batch_size, seqlen, num_heads, head_dim = u.shape
    d_state = A.shape[1]
    if d_state > 128:
        raise ValueError("grouped_ssd_scan_triton currently supports d_state <= 128")

    u_contig = u.contiguous()
    dt_contig = dt.contiguous()
    A_contig = A.contiguous()
    B_contig = Bpar.contiguous()
    C_contig = Cpar.contiguous()
    D_contig = Dskip.contiguous()
    gate_contig = delta_gate.contiguous() if delta_gate is not None else None

    out = torch.empty_like(u_contig)
    block_p = min(max(triton.next_power_of_2(head_dim), 16), 128)
    block_n = max(8, triton.next_power_of_2(d_state))

    grid = (batch_size * num_heads, triton.cdiv(head_dim, block_p))
    _grouped_ssd_scan_kernel[grid](
        u_contig,
        dt_contig,
        A_contig,
        B_contig,
        C_contig,
        D_contig,
        gate_contig if gate_contig is not None else u_contig,
        out,
        seqlen,
        num_heads,
        head_dim,
        d_state,
        USE_GATE=gate_contig is not None,
        BLOCK_P=block_p,
        BLOCK_N=block_n,
    )
    return out


class _GroupedSSDScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, Bpar, Cpar, Dskip, delta_gate, use_triton: bool):
        has_gate = delta_gate is not None
        ctx.has_gate = has_gate
        ctx.use_triton = use_triton

        if use_triton:
            y = grouped_ssd_scan_triton(
                u, dt, A, Bpar, Cpar, Dskip, delta_gate=delta_gate
            )
        else:
            y = grouped_ssd_scan_reference(
                u, dt, A, Bpar, Cpar, Dskip, delta_gate=delta_gate
            )

        gate_tensor = (
            delta_gate if has_gate else torch.tensor([], device=u.device, dtype=u.dtype)
        )
        ctx.save_for_backward(u, dt, A, Bpar, Cpar, Dskip, gate_tensor)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        u, dt, A, Bpar, Cpar, Dskip, gate_tensor = ctx.saved_tensors
        delta_gate = gate_tensor if ctx.has_gate else None
        grads = grouped_ssd_scan_backward_reference(
            grad_y,
            u,
            dt,
            A,
            Bpar,
            Cpar,
            Dskip,
            delta_gate=delta_gate,
            needs_input_grad=ctx.needs_input_grad[:7],
        )
        du, ddt, dA, dB, dC, dD, dgate = grads
        return du, ddt, dA, dB, dC, dD, dgate, None


def grouped_ssd_scan(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    delta_gate: torch.Tensor | None = None,
    use_triton: bool = True,
) -> torch.Tensor:
    can_use_triton = (
        use_triton
        and GROUPED_SSD_TRITON_AVAILABLE
        and u.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and Bpar.is_cuda
        and Cpar.is_cuda
        and Dskip.is_cuda
        and (delta_gate is None or delta_gate.is_cuda)
        and A.shape[1] <= 128
    )
    return _GroupedSSDScanFn.apply(
        u, dt, A, Bpar, Cpar, Dskip, delta_gate, can_use_triton
    )
