from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralODE(nn.Module):
    """
    Neural ODE block for time series.
    Input:  [B, T, C]
    Output: [B, T, C]
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        solver: str = "rk4",          # 'euler' | 'midpoint' | 'rk4'
        step_size: float = 0.1,
        adaptive: bool = False,        # adaptive only meaningful for 'rk4' here
        rtol: float = 1e-3,
        atol: float = 1e-4,
        ode_net: Optional[nn.Module] = None,  # optional external dynamics
        max_substeps: int = 10_000,    # safety
        min_dt: float = 1e-6,          # safety
        max_dt: float = 1.0,           # safety
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver = solver.lower()
        self.step_size = float(step_size)
        self.adaptive = bool(adaptive)
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.max_substeps = int(max_substeps)
        self.min_dt = float(min_dt)
        self.max_dt = float(max_dt)

        # Default neural dynamics if none provided: f(y) -> dy/dt
        self.ode_func_net = ode_net or nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size),
        )

    def _ode_func(self, t: float, y: torch.Tensor) -> torch.Tensor:
        # Autonomy by default; ignore t. Replace with a t-aware net if needed.
        return self.ode_func_net(y)

    # Fixed-step integrators
    def _euler_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        return y + dt * self._ode_func(t, y)

    def _midpoint_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        k1 = self._ode_func(t, y)
        k2 = self._ode_func(t + 0.5 * dt, y + 0.5 * dt * k1)
        return y + dt * k2

    def _rk4_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        k1 = self._ode_func(t, y)
        k2 = self._ode_func(t + 0.5 * dt, y + 0.5 * dt * k1)
        k3 = self._ode_func(t + 0.5 * dt, y + 0.5 * dt * k2)
        k4 = self._ode_func(t + dt,       y + dt * k3)
        return y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    def _odeint(self, y0: torch.Tensor, t0: float, t1: float, steps_hint: int) -> torch.Tensor:
        """Integrate y' = f(t, y) from t0 to t1; returns y(t1)."""
        y = y0
        t = float(t0)
        direction = 1.0 if t1 >= t0 else -1.0
        total = abs(t1 - t0)

        if self.solver in ("euler", "midpoint") or (self.solver == "rk4" and not self.adaptive):
            n = max(1, int(max(steps_hint, total / self.step_size)))
            dt = direction * (total / n)
            step = {
                "euler":    self._euler_step,
                "midpoint": self._midpoint_step,
                "rk4":      self._rk4_step,
            }[self.solver]
            for _ in range(n):
                y = step(y, t, dt)
                t += dt
            return y

        if self.solver != "rk4":  # adaptive only implemented for rk4 here
            raise ValueError(f"Unknown solver: {self.solver}")

        # -------- Adaptive RK4 with step-doubling error estimate --------
        # Keep dt as a Python float to avoid type issues with min/max.
        dt = float(min(self.step_size, total)) * direction
        n_sub = 0

        while (direction * (t1 - t)) > 0:
            if n_sub > self.max_substeps:
                # Safety escape; return best effort
                break
            n_sub += 1

            # Clip dt not to step beyond t1
            remaining = (t1 - t)
            if abs(dt) > abs(remaining):
                dt = float(remaining)

            # Full step
            y_full_step = self._rk4_step(y, t, dt)

            # Two half steps (step-doubling)
            y_half = self._rk4_step(y, t, 0.5 * dt)
            y_two_half = self._rk4_step(y_half, t + 0.5 * dt, 0.5 * dt)

            # Error estimate (L2 over features, then max over batch)
            err = torch.norm(y_full_step - y_two_half, dim=-1)          # [B]
            ref = torch.max(torch.norm(y_full_step, dim=-1),
                            torch.norm(y_two_half,  dim=-1))             # [B]
            tol = self.atol + self.rtol * ref                            # [B]
            err_ratio = (err / (tol.clamp_min(1e-12))).amax()            # scalar tensor

            if torch.isfinite(err_ratio):
                er = float(err_ratio.item())
            else:
                er = float('inf')

            if er <= 1.0:
                # Accept
                y = y_two_half
                t = float(t + dt)

                # Step-size update: 0.9 * er^(-1/5) for RK4 (bounded)
                scale = 0.9 * (er ** (-0.2)) if er > 1e-12 else 5.0
                scale = max(0.2, min(5.0, scale))
                dt = float(max(self.min_dt, min(self.max_dt, abs(dt) * scale))) * direction
            else:
                # Reject; shrink dt and retry (don’t advance t or y)
                scale = 0.9 * (er ** (-0.2))
                scale = max(0.2, min(0.5, scale))  # be conservative on reject
                dt = float(max(self.min_dt, abs(dt) * scale)) * direction

                # If dt got too small, bail out to avoid stalling
                if abs(dt) <= self.min_dt + 1e-12:
                    # Accept anyway to make progress
                    y = y_two_half
                    t = float(t + dt)
                    # modest growth afterwards
                    dt = float(min(self.max_dt, abs(dt) * 1.5)) * direction

        return y

    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:     [B, T, C]
        times: [T] (monotone). If None, uses uniform linspace in [0, 1].
        """
        B, T, C = x.shape
        if times is None:
            times = torch.linspace(0., 1., T, device=x.device, dtype=x.dtype)
        else:
            # Ensure 1D, same device/dtype
            times = times.to(device=x.device, dtype=x.dtype).view(-1)
            if times.numel() != T:
                raise ValueError("times must have length equal to seq_len (T).")

        # Optional: check monotonicity (warn instead of error)
        # if not torch.all(times[1:] >= times[:-1]):
        #     raise ValueError("times must be non-decreasing.")

        y_t = x[:, 0, :]                   # initial state
        outputs = [y_t]

        for i in range(1, T):
            t0 = float(times[i - 1].item())
            t1 = float(times[i].item())

            # steps_hint: proportional to interval length
            steps_hint = max(1, int(abs(t1 - t0) / max(self.step_size, self.min_dt)))
            y_t = self._odeint(outputs[-1], t0, t1, steps_hint)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)
