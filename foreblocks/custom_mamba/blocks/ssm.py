from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import dt_prep, fused_out, selective_scan
from .conv import CausalDepthwiseConv1d
from .norms import RMSNormWeightOnly
from .utils import auto_dt_rank, conv_step, fused_out_2d, inverse_softplus


class HybridMambaBlock(nn.Module):
    """Mamba-style SSM block (selective scan, dense per-feature A/D matrices)."""

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_cuda_scan: bool = True,
        use_pre_norm: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or auto_dt_rank(d_model)
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_cuda_scan = use_cuda_scan

        self.pre_norm = nn.LayerNorm(d_model) if use_pre_norm else nn.Identity()

        total_out = 2 * self.d_inner + self.dt_rank + 2 * self.d_inner * self.d_state
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.d_inner, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.d_inner))
        self.dt_bias = nn.Parameter(torch.zeros(self.d_inner))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(
                1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype
            )
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.d_inner, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(
                self.d_inner, device=self.dt_bias.device, dtype=self.dt_bias.dtype
            )
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(
                self.dt_min
            )
            self.dt_bias.copy_(inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        D = self.d_inner
        N = self.d_state
        z, u, dt_hidden, Bflat, Cflat = torch.split(
            p, [D, D, self.dt_rank, D * N, D * N], dim=-1
        )
        Bpar = Bflat.reshape(batch_size, seqlen, D, N)
        Cpar = Cflat.reshape(batch_size, seqlen, D, N)
        return z, u, dt_hidden, Bpar, Cpar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        p = self.in_proj(self.pre_norm(x))
        z, u, dt_hidden, Bpar, Cpar = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        y = selective_scan(
            u=u,
            dt=dt,
            A=A,
            Bpar=Bpar,
            Cpar=Cpar,
            Dskip=self.Dskip,
            use_cuda_kernel=self.use_cuda_scan,
        )

        residual_inner = self.residual_proj(residual)
        y = fused_out(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y)

    def make_state(
        self, batch: int, device=None, dtype=None
    ) -> dict[str, torch.Tensor]:
        """Return a fresh recurrent state for *batch* sequences."""
        return {
            "conv": torch.zeros(
                batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype
            ),
            "ssm": torch.zeros(
                batch, self.d_inner, self.d_state, device=device, dtype=dtype
            ),
        }

    def step(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Single-token recurrent forward."""
        B = x.shape[0]
        D, N = self.d_inner, self.d_state

        p = self.in_proj(self.pre_norm(x))
        z, u_raw, dt_hidden, Bflat, Cflat = torch.split(
            p, [D, D, self.dt_rank, D * N, D * N], dim=-1
        )
        Bpar = Bflat.reshape(B, D, N)
        Cpar = Cflat.reshape(B, D, N)

        weight = self.conv.conv.weight.view(D, self.d_conv)
        u, state["conv"] = conv_step(u_raw, state["conv"], weight, self.conv.conv.bias)

        dt_raw = self.dt_proj(dt_hidden)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log)
        A_disc = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))

        h = state["ssm"].to(dtype=A_disc.dtype)
        h = A_disc * h + dt.unsqueeze(-1) * Bpar * u.unsqueeze(-1)
        y = (Cpar * h).sum(-1) + self.Dskip * u
        state["ssm"] = h.detach()

        residual_inner = self.residual_proj(x)
        y_normed = fused_out_2d(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y_normed)
