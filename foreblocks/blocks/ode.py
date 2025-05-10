import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union

class NeuralODE(nn.Module):
    """
    Neural ordinary differential equation (ODE) block for time series modeling.
    
    Uses numerical ODE solvers to model continuous time dynamics with neural networks.
    Especially useful for irregularly sampled time series and for modeling complex dynamics.
    
    Based on the Neural ODE paper (https://arxiv.org/abs/1806.07366) with modifications
    for time series applications.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        solver: str = 'rk4',
        step_size: float = 0.1,
        adaptive: bool = False,
        rtol: float = 1e-3,
        atol: float = 1e-4
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden dimension for ODE function
            solver: ODE solver ('euler', 'midpoint', 'rk4')
            step_size: Step size for fixed-step solvers
            adaptive: Whether to use adaptive step size (only for 'rk4')
            rtol: Relative tolerance for adaptive stepping
            atol: Absolute tolerance for adaptive stepping
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.solver = solver
        self.step_size = step_size
        self.adaptive = adaptive
        self.rtol = rtol
        self.atol = atol
        
        # Neural network for ODE function
        self.ode_func = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, input_size)
        )
        
    def _ode_func(self, t: float, y: torch.Tensor) -> torch.Tensor:
        """ODE function to integrate: dy/dt = f(t, y)"""
        return self.ode_func(y)
    
    def _euler_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Euler method step"""
        return y + dt * self._ode_func(t, y)
    
    def _midpoint_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Midpoint method step"""
        k1 = self._ode_func(t, y)
        k2 = self._ode_func(t + dt/2, y + dt/2 * k1)
        return y + dt * k2
    
    def _rk4_step(self, y: torch.Tensor, t: float, dt: float) -> torch.Tensor:
        """Runge-Kutta 4th order step"""
        k1 = self._ode_func(t, y)
        k2 = self._ode_func(t + dt/2, y + dt/2 * k1)
        k3 = self._ode_func(t + dt/2, y + dt/2 * k2)
        k4 = self._ode_func(t + dt, y + dt * k3)
        return y + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    def _odeint(
        self, 
        y0: torch.Tensor, 
        t_span: Tuple[float, float],
        steps: int
    ) -> torch.Tensor:
        """
        Integrate ODE from t_span[0] to t_span[1].
        
        Args:
            y0: Initial state [batch_size, input_size]
            t_span: Tuple of (t_start, t_end)
            steps: Number of integration steps
            
        Returns:
            Final state at t_end [batch_size, input_size]
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / steps
        
        y = y0
        t = t_start
        
        if self.solver == 'euler':
            for _ in range(steps):
                y = self._euler_step(y, t, dt)
                t += dt
        elif self.solver == 'midpoint':
            for _ in range(steps):
                y = self._midpoint_step(y, t, dt)
                t += dt
        elif self.solver == 'rk4':
            if not self.adaptive:
                for _ in range(steps):
                    y = self._rk4_step(y, t, dt)
                    t += dt
            else:
                # Simple adaptive stepping with RK4
                while t < t_end:
                    dt_try = min(dt, t_end - t)
                    
                    # Take a full step
                    y_new = self._rk4_step(y, t, dt_try)
                    
                    # Take two half steps
                    y_half = self._rk4_step(y, t, dt_try/2)
                    y_full = self._rk4_step(y_half, t + dt_try/2, dt_try/2)
                    
                    # Estimate error
                    error = torch.norm(y_new - y_full, dim=-1, keepdim=True)
                    tol = self.atol + self.rtol * torch.max(torch.norm(y_new, dim=-1, keepdim=True), 
                                                            torch.norm(y_full, dim=-1, keepdim=True))
                    
                    # Accept step if error is small enough
                    if (error <= tol).all():
                        y = y_full
                        t += dt_try
                        
                        # Increase step size for next step
                        dt = dt_try * 0.9 * torch.min(torch.sqrt(tol / error.clamp(min=1e-10)))
                    else:
                        # Decrease step size and retry
                        dt = dt_try * 0.9 * torch.min(torch.sqrt(tol / error.clamp(min=1e-10)))
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
            
        return y
    
    def forward(self, x: torch.Tensor, times: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            times: Optional tensor of time points [seq_len]
                   If not provided, uniform time steps are assumed
            
        Returns:
            Output tensor [batch_size, seq_len, input_size]
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate uniform time steps if not provided
        if times is None:
            times = torch.linspace(0, 1, seq_len, device=x.device)
        
        # Process each time step
        outputs = [x[:, 0]]  # Start with first time step
        
        for i in range(1, seq_len):
            # Integrate ODE from t_{i-1} to t_i
            t_span = (times[i-1].item(), times[i].item())
            
            # Number of steps proportional to time difference
            steps = max(1, int((t_span[1] - t_span[0]) / self.step_size))
            
            # Integrate to next time point
            y_next = self._odeint(outputs[-1], t_span, steps)
            outputs.append(y_next)
            
        return torch.stack(outputs, dim=1)

