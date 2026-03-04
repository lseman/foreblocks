"""Mamba SSM backbone — used by arch_mode="mamba"."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_primitives import RMSNorm

__all__ = ["MambaBranch"]


class _MambaBlock(nn.Module):
    """Single selective-SSM layer with pre-norm and residual (used in MambaBranch).

    Implements a vectorised closed-form causal scan. The causal weight matrix M
    is O(L² × inner), so for very long sequences (L > 256) prefer ``expand=1``
    or gradient checkpointing.
    """

    def __init__(self, dim: int, expand: int = 2):
        super().__init__()
        inner = dim * expand
        self.in_proj = nn.Linear(dim, inner * 2, bias=False)
        self._pad = 2
        self.conv = nn.Conv1d(
            inner, inner, kernel_size=3, padding=0, groups=inner, bias=False
        )
        self.state_decay = nn.Parameter(torch.zeros(inner))
        self.state_scale = nn.Parameter(torch.ones(inner))
        self.out_proj = nn.Linear(inner, dim, bias=False)
        self.norm = RMSNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        xp = self.in_proj(x)
        u, g = xp.chunk(2, dim=-1)

        u = self.conv(F.pad(u.transpose(1, 2), (self._pad, 0))).transpose(1, 2)

        d = torch.sigmoid(self.state_decay)
        inp = (1.0 - d) * self.state_scale * u
        log_d = d.clamp(min=1e-6).log()
        t_idx = torch.arange(L, device=x.device, dtype=x.dtype)
        diff = t_idx.unsqueeze(1) - t_idx.unsqueeze(0)
        causal_mask = (diff >= 0).to(x.dtype)
        M = (diff.clamp(min=0).unsqueeze(-1) * log_d).exp() * causal_mask.unsqueeze(-1)
        y = torch.einsum("tkd,bkd->btd", M, inp) * torch.sigmoid(g)
        return self.norm(self.out_proj(y))


class MambaBranch(nn.Module):
    """Multi-layer Mamba SSM encoder for ``arch_mode="mamba"``.

    Replaces the ``MixedEncoder`` + ``MixedDecoder`` pair with a purely
    recurrent SSM stack followed by a direct forecast head.

    Data flow::

        [B, L, hidden_dim]
            ──in_proj──▶  [B, L, latent_dim]
            ──N × (LayerNorm + _MambaBlock + residual)──▶
            ──mean+last-token pool──▶  [B, latent_dim]
            ──forecast head──▶  [B, forecast_horizon, input_dim]
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        forecast_horizon: int,
        input_dim: int,
        num_layers: int = 3,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim

        self.in_proj = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim, bias=False),
            nn.LayerNorm(latent_dim),
        )
        self.blocks = nn.ModuleList(
            [_MambaBlock(latent_dim, expand=expand) for _ in range(num_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(latent_dim) for _ in range(num_layers)]
        )
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, forecast_horizon * input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        h = self.in_proj(x)
        for block, norm in zip(self.blocks, self.norms):
            h = h + block(norm(h))
        pooled = h.mean(dim=1) * 0.5 + h[:, -1, :] * 0.5
        out = self.head(pooled)
        return out.view(B, self.forecast_horizon, self.input_dim)
