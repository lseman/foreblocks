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


class _SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, dt, A, Bpar, Cpar, Dskip, use_cuda_kernel: bool):
        ctx.use_cuda_kernel = use_cuda_kernel

        if use_cuda_kernel:
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
        else:
            y = selective_scan_reference(u, dt, A, Bpar, Cpar, Dskip)
            ctx.save_for_backward(u, dt, A, Bpar, Cpar, Dskip)

        return y

    @staticmethod
    def backward(ctx, grad_y):
        if ctx.use_cuda_kernel:
            u, dt, A, Bpar, Cpar, Dskip, states = ctx.saved_tensors
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

        u, dt, A, Bpar, Cpar, Dskip = ctx.saved_tensors

        with torch.enable_grad():
            u_ = u.detach().requires_grad_(u.requires_grad)
            dt_ = dt.detach().requires_grad_(dt.requires_grad)
            A_ = A.detach().requires_grad_(A.requires_grad)
            B_ = Bpar.detach().requires_grad_(Bpar.requires_grad)
            C_ = Cpar.detach().requires_grad_(Cpar.requires_grad)
            D_ = Dskip.detach().requires_grad_(Dskip.requires_grad)

            y_ref = selective_scan_reference(u_, dt_, A_, B_, C_, D_)
            all_inputs = (u_, dt_, A_, B_, C_, D_)
            grad_inputs = tuple(t for t in all_inputs if t.requires_grad)

            grads_required = torch.autograd.grad(
                outputs=y_ref,
                inputs=grad_inputs,
                grad_outputs=grad_y,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        grads_iter = iter(grads_required)
        grads = tuple(next(grads_iter) if t.requires_grad else None for t in all_inputs)
        du, ddt, dA, dB, dC, dD = grads
        return du, ddt, dA, dB, dC, dD, None


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
