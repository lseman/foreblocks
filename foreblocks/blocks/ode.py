from typing import Optional

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint, odeint_adjoint

    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False
    print("Warning: torchdiffeq not installed → falling back to custom solvers")


class NeuralODE(nn.Module):
    """
    Neural ODE layer for time-series modeling.
    Supports both custom fixed/adaptive solvers and torchdiffeq (recommended).

    Input:  [B, T, C] or [B, T, C] + times [T]
    Output: [B, T, C]  (evaluated at the same time points)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int | None = None,
        solver: str = "rk4",  # "euler", "rk4", "dopri5", "rk4_adjoint", ...
        step_size: float = 0.05,
        rtol: float = 1e-4,
        atol: float = 1e-6,
        ode_net: Optional[nn.Module] = None,
        use_adjoint: bool = True,  # only if torchdiffeq available
        max_steps: int = 100_000,
        min_dt: float = 1e-7,
        max_dt: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size or input_size * 2
        self.solver = solver.lower()
        self.rtol = rtol
        self.atol = atol
        self.step_size = step_size
        self.use_adjoint = use_adjoint and HAS_TORCHDIFFEQ
        self.max_steps = max_steps
        self.min_dt = min_dt
        self.max_dt = max_dt

        # Dynamics network f(t,y) → dy/dt
        if ode_net is not None:
            self.func = ode_net
        else:
            self.func = nn.Sequential(
                nn.Linear(input_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
                nn.Linear(self.hidden_size, input_size),
            )

    def forward(
        self,
        x: torch.Tensor,  # [B, T, C]
        times: Optional[torch.Tensor] = None,
        return_all: bool = True,
    ) -> torch.Tensor:
        """
        x:      [B, T, C] initial time series (y0 = x[:,0,:])
        times:  [T] time points (increasing). If None → uniform [0,1]
        """
        B, T, C = x.shape
        device, dtype = x.device, x.dtype

        if times is None:
            t = torch.linspace(0.0, 1.0, T, device=device, dtype=dtype)
        else:
            t = times.to(device=device, dtype=dtype).view(-1)
            if t.numel() != T:
                raise ValueError(f"times must have length {T}, got {t.numel()}")

        # Ensure sorted
        if not torch.all(t[1:] >= t[:-1]):
            raise ValueError("Time points must be non-decreasing.")

        y0 = x[:, 0, :]  # [B, C]

        if HAS_TORCHDIFFEQ and self.solver not in ("euler", "rk4_fixed"):
            # Use torchdiffeq (most efficient & accurate)
            solver_name = "rk4" if self.solver == "rk4" else self.solver
            method = (
                "adjoint" if self.use_adjoint and solver_name != "rk4" else solver_name
            )

            y = odeint_adjoint(
                self.func,
                y0,
                t,
                method=method,
                rtol=self.rtol,
                atol=self.atol,
                options={"step_size": self.step_size} if method == "rk4" else None,
            )  # [T, B, C]

            y = y.permute(1, 0, 2)  # [B, T, C]

        else:
            # Custom fixed-step fallback (slower but dependency-free)
            y = torch.zeros(B, T, C, device=device, dtype=dtype)
            y[:, 0] = y0

            for i in range(1, T):
                t0, t1 = float(t[i - 1]), float(t[i])
                dt_total = t1 - t0
                if dt_total <= 0:
                    y[:, i] = y[:, i - 1]
                    continue

                y_curr = y[:, i - 1]
                steps = max(1, int(abs(dt_total) / self.step_size) + 1)
                dt = dt_total / steps

                for _ in range(steps):
                    if self.solver == "euler":
                        y_curr = y_curr + dt * self.func(t0, y_curr)
                    elif self.solver == "rk4_fixed":
                        k1 = self.func(t0, y_curr)
                        k2 = self.func(t0 + 0.5 * dt, y_curr + 0.5 * dt * k1)
                        k3 = self.func(t0 + 0.5 * dt, y_curr + 0.5 * dt * k2)
                        k4 = self.func(t0 + dt, y_curr + dt * k3)
                        y_curr = y_curr + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                    else:
                        raise ValueError(f"Unsupported custom solver: {self.solver}")
                    t0 += dt

                y[:, i] = y_curr

        # Safety check
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("Warning: NaN/Inf detected in NeuralODE output")

        return y if return_all else y[:, -1, :]


# Quick test
if __name__ == "__main__":
    torch.manual_seed(0)
    model = NeuralODE(input_size=8, hidden_size=32, solver="rk4")

    x = torch.randn(4, 50, 8)  # B,T,C
    out = model(x)
    print(out.shape)  # torch.Size([4, 50, 8])
