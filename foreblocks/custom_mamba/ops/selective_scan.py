from __future__ import annotations

import torch

from ..cuda import load_selective_scan_extension


def selective_scan_reference(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
) -> torch.Tensor:
    Bsz, T, D = u.shape
    N = A.shape[1]

    state = torch.zeros(Bsz, D, N, device=u.device, dtype=torch.float32)
    ys = []

    A32 = A.float()
    D32 = Dskip.float()

    for t in range(T):
        u_t = u[:, t, :].float()
        dt_t = dt[:, t, :].float()
        B_t = Bpar[:, t, :, :].float()
        C_t = Cpar[:, t, :, :].float()

        abar = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        state = abar * state + dt_t.unsqueeze(-1) * B_t * u_t.unsqueeze(-1)
        y_t = (C_t * state).sum(dim=-1) + D32.unsqueeze(0) * u_t
        ys.append(y_t.to(u.dtype))

    return torch.stack(ys, dim=1)


def selective_scan_backward_reference(
    grad_y: torch.Tensor,
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    needs_input_grad: tuple[bool, bool, bool, bool, bool, bool] | None = None,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if needs_input_grad is None:
        needs_input_grad = (True, True, True, True, True, True)

    Bsz, T, D = u.shape
    N = A.shape[1]

    u32 = u.float()
    dt32 = dt.float()
    A32 = A.float()
    B32 = Bpar.float()
    C32 = Cpar.float()
    D32 = Dskip.float()

    state = torch.zeros(Bsz, D, N, device=u.device, dtype=torch.float32)
    states_before: list[torch.Tensor] = []
    states_after: list[torch.Tensor] = []
    for t in range(T):
        states_before.append(state)
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        abar = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        state = abar * state + dt_t.unsqueeze(-1) * B32[:, t] * u_t.unsqueeze(-1)
        states_after.append(state)

    du = torch.zeros_like(u32) if needs_input_grad[0] else None
    ddt = torch.zeros_like(dt32) if needs_input_grad[1] else None
    dA = torch.zeros_like(A32) if needs_input_grad[2] else None
    dB = torch.zeros_like(B32) if needs_input_grad[3] else None
    dC = torch.zeros_like(C32) if needs_input_grad[4] else None
    dD = torch.zeros_like(D32) if needs_input_grad[5] else None
    grad_state = torch.zeros(Bsz, D, N, device=u.device, dtype=torch.float32)

    for t in range(T - 1, -1, -1):
        gy_t = grad_y[:, t].float()
        u_t = u32[:, t]
        dt_t = dt32[:, t]
        B_t = B32[:, t]
        C_t = C32[:, t]
        state_prev = states_before[t]
        state_t = states_after[t]

        if dC is not None:
            dC[:, t] = gy_t.unsqueeze(-1) * state_t
        if dD is not None:
            dD += (gy_t * u_t).sum(dim=0)
        if du is not None:
            du[:, t] += gy_t * D32.unsqueeze(0)

        grad_state = grad_state + gy_t.unsqueeze(-1) * C_t

        if du is not None:
            du[:, t] += (grad_state * dt_t.unsqueeze(-1) * B_t).sum(dim=-1)
        if dB is not None:
            dB[:, t] = grad_state * dt_t.unsqueeze(-1) * u_t.unsqueeze(-1)
        if ddt is not None:
            ddt[:, t] += (grad_state * B_t * u_t.unsqueeze(-1)).sum(dim=-1)

        abar = torch.exp(dt_t.unsqueeze(-1) * A32.unsqueeze(0))
        decay_grad = grad_state * state_prev
        if dA is not None:
            dA += (decay_grad * abar * dt_t.unsqueeze(-1)).sum(dim=0)
        if ddt is not None:
            ddt[:, t] += (decay_grad * abar * A32.unsqueeze(0)).sum(dim=-1)

        grad_state = grad_state * abar

    return (
        du.to(u.dtype) if du is not None else None,
        ddt.to(dt.dtype) if ddt is not None else None,
        dA.to(A.dtype) if dA is not None else None,
        dB.to(Bpar.dtype) if dB is not None else None,
        dC.to(Cpar.dtype) if dC is not None else None,
        dD.to(Dskip.dtype) if dD is not None else None,
    )


class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, Bpar, Cpar, Dskip, use_cuda_kernel: bool):
        ctx.use_cuda_kernel = use_cuda_kernel

        if use_cuda_kernel:
            try:
                ext = load_selective_scan_extension(verbose=False)
                y, states = ext.selective_scan_fwd(
                    u.contiguous(),
                    dt.contiguous(),
                    A.contiguous(),
                    Bpar.contiguous(),
                    Cpar.contiguous(),
                    Dskip.contiguous(),
                )
                ctx.save_for_backward(u, dt, A, Bpar, Cpar, Dskip, states)
                return y
            except Exception:
                ctx.use_cuda_kernel = False

        y = selective_scan_reference(u, dt, A, Bpar, Cpar, Dskip)
        ctx.save_for_backward(u, dt, A, Bpar, Cpar, Dskip)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.use_cuda_kernel:
            u, dt, A, Bpar, Cpar, Dskip, states = ctx.saved_tensors
            try:
                ext = load_selective_scan_extension(verbose=False)
                du, ddt, dA, dB, dC, dD = ext.selective_scan_bwd(
                    u.contiguous(),
                    dt.contiguous(),
                    A.contiguous(),
                    Bpar.contiguous(),
                    Cpar.contiguous(),
                    Dskip.contiguous(),
                    grad_y.contiguous(),
                    states.contiguous(),
                )
                return du, ddt, dA, dB, dC, dD, None
            except Exception:
                grads = selective_scan_backward_reference(
                    grad_y,
                    u,
                    dt,
                    A,
                    Bpar,
                    Cpar,
                    Dskip,
                    needs_input_grad=ctx.needs_input_grad[:6],
                )
                return (*grads, None)

        u, dt, A, Bpar, Cpar, Dskip = ctx.saved_tensors
        grads = selective_scan_backward_reference(
            grad_y,
            u,
            dt,
            A,
            Bpar,
            Cpar,
            Dskip,
            needs_input_grad=ctx.needs_input_grad[:6],
        )
        return (*grads, None)


def selective_scan(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
    use_cuda_kernel: bool = True,
) -> torch.Tensor:
    can_use_cuda = (
        use_cuda_kernel
        and u.is_cuda
        and dt.is_cuda
        and A.is_cuda
        and Bpar.is_cuda
        and Cpar.is_cuda
        and Dskip.is_cuda
    )
    return _SelectiveScanFn.apply(u, dt, A, Bpar, Cpar, Dskip, can_use_cuda)
