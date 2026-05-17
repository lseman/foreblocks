from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops import dt_prep, fused_out, grouped_ssd_scan
from .conv import CausalDepthwiseConv1d
from .norms import RMSNormWeightOnly
from .utils import auto_dt_rank, conv_step, fused_out_2d, inverse_softplus


class StructuredStateSpaceDualityBranch(nn.Module):
    """Multi-head Structured State Space Duality (SSD) branch."""

    def __init__(
        self,
        d_model: int,
        d_inner: int | None = None,
        d_state: int = 16,
        d_conv: int = 4,
        dt_rank: int | None = None,
        num_heads: int = 8,
        dt_min: float = 1e-4,
        dt_max: float = 1.0,
        use_gated_delta: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner or 2 * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or auto_dt_rank(d_model)
        self.num_heads = num_heads
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.use_gated_delta = use_gated_delta
        if self.d_inner % num_heads != 0:
            raise ValueError("d_inner must be divisible by num_heads")

        self.head_dim = self.d_inner // self.num_heads
        extra = self.num_heads if use_gated_delta else 0
        total_out = (
            2 * self.d_inner + self.dt_rank + 2 * self.num_heads * self.d_state + extra
        )
        self.in_proj = nn.Linear(d_model, total_out, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.num_heads, bias=False)
        self.conv = CausalDepthwiseConv1d(self.d_inner, kernel_size=d_conv)

        self.A_log = nn.Parameter(torch.empty(self.num_heads, self.d_state))
        self.Dskip = nn.Parameter(torch.ones(self.num_heads, self.head_dim))
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))

        self.norm = RMSNormWeightOnly(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.residual_proj = (
            nn.Identity()
            if self.d_model == self.d_inner
            else nn.Linear(self.d_model, self.d_inner, bias=False)
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        nn.init.xavier_uniform_(self.dt_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if isinstance(self.residual_proj, nn.Linear):
            nn.init.xavier_uniform_(self.residual_proj.weight)
        with torch.no_grad():
            base = torch.arange(
                1, self.d_state + 1, device=self.A_log.device, dtype=self.A_log.dtype
            )
            self.A_log.copy_(base.log().unsqueeze(0).expand(self.num_heads, -1))
            self.Dskip.fill_(1.0)
            dt = torch.rand(
                self.num_heads, device=self.dt_bias.device, dtype=self.dt_bias.dtype
            )
            dt = dt * (math.log(self.dt_max) - math.log(self.dt_min)) + math.log(
                self.dt_min
            )
            self.dt_bias.copy_(inverse_softplus(dt.exp()))

    def _split_proj(self, p: torch.Tensor) -> tuple[torch.Tensor, ...]:
        batch_size, seqlen, _ = p.shape
        h = self.num_heads
        n = self.d_state
        split_sizes = [self.d_inner, self.d_inner, self.dt_rank, h * n, h * n]
        if self.use_gated_delta:
            split_sizes.append(h)
        parts = torch.split(p, split_sizes, dim=-1)
        z, u, dt_hidden, Bflat, Cflat = parts[:5]
        delta_gate = parts[5] if self.use_gated_delta else None
        Bpar = Bflat.reshape(batch_size, seqlen, h, n)
        Cpar = Cflat.reshape(batch_size, seqlen, h, n)
        return z, u, dt_hidden, Bpar, Cpar, delta_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        batch_size, seqlen, _ = x.shape

        p = self.in_proj(x)
        z, u, dt_hidden, Bpar, Cpar, delta_gate = self._split_proj(p)
        u = self.conv(u)

        dt_raw = self.dt_proj(dt_hidden)
        dt = dt_prep(dt_raw, self.dt_bias, dt_min=self.dt_min, dt_max=self.dt_max)
        A = -torch.exp(self.A_log)

        u_heads = u.reshape(batch_size, seqlen, self.num_heads, self.head_dim)
        y = grouped_ssd_scan(
            u=u_heads,
            dt=dt,
            A=A,
            Bpar=Bpar,
            Cpar=Cpar,
            Dskip=self.Dskip,
            delta_gate=delta_gate,
        ).reshape(batch_size, seqlen, self.d_inner)

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
                batch,
                self.num_heads,
                self.head_dim,
                self.d_state,
                device=device,
                dtype=dtype,
            ),
        }

    def step(self, x: torch.Tensor, state: dict[str, torch.Tensor]) -> torch.Tensor:
        """Single-token recurrent forward."""
        B = x.shape[0]
        H, P, N = self.num_heads, self.head_dim, self.d_state
        D = self.d_inner

        split_sizes = [D, D, self.dt_rank, H * N, H * N]
        if self.use_gated_delta:
            split_sizes.append(H)
        parts = torch.split(self.in_proj(x), split_sizes, dim=-1)
        z, u_raw, dt_hidden, Bflat, Cflat = parts[:5]
        delta_gate_raw = parts[5] if self.use_gated_delta else None

        Bpar = Bflat.reshape(B, H, N)
        Cpar = Cflat.reshape(B, H, N)

        weight = self.conv.conv.weight.view(D, self.d_conv)
        u, state["conv"] = conv_step(u_raw, state["conv"], weight, self.conv.conv.bias)

        dt_raw = self.dt_proj(dt_hidden)
        dt = F.softplus(dt_raw + self.dt_bias).clamp(self.dt_min, self.dt_max)

        A = -torch.exp(self.A_log)
        h = state["ssm"].to(dtype=dt.dtype)

        decay = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0))
        decayed = decay.unsqueeze(-2) * h

        u_heads = u.reshape(B, H, P)
        delta = (
            dt.unsqueeze(-1).unsqueeze(-1) * Bpar.unsqueeze(-2) * u_heads.unsqueeze(-1)
        )

        if delta_gate_raw is not None:
            gate = torch.sigmoid(delta_gate_raw).unsqueeze(-1).unsqueeze(-1)
            h_new = decayed + gate * delta
        else:
            h_new = decayed + delta

        y_heads = (Cpar.unsqueeze(-2) * h_new).sum(-1) + self.Dskip.unsqueeze(
            0
        ) * u_heads
        y = y_heads.reshape(B, D)
        state["ssm"] = h_new.detach()

        residual_inner = self.residual_proj(x)
        y_normed = fused_out_2d(y, z, residual_inner, self.norm.weight)
        return self.out_proj(y_normed)
