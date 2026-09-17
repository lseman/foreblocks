"""Simplest reference chunked-SSD forward/backward, used for correctness testing."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chunked_ssd_forward_reference(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    chunk_size: int = 64,
) -> torch.Tensor:
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
            assert adt32 is not None
            adt32 = F.pad(adt32, (0, 0, 0, pad))
    T_pad = T + pad

    def _log_decay(t_idx):
        # [B, H, 1, 1]
        if use_adt:
            assert adt32 is not None
            return adt32[:, t_idx].unsqueeze(-1).unsqueeze(-1)
        return dt32[:, t_idx].unsqueeze(-1).unsqueeze(-1) * A32.unsqueeze(0).unsqueeze(
            -1
        ).unsqueeze(-1)

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
    dadt = torch.zeros_like(adt32) if (use_adt and needs_adt_grad) else None  # type: ignore[arg-type]

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
