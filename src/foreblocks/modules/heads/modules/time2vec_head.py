"""foreblocks.modules.heads.modules.time2vec_head.

Time2Vec-style periodic temporal encoding projected back to feature dimensions.

Computes sinusoidal periodic features over normalized timestep positions, combines
them with the original input, and projects back to the original feature dimension.
Keeps sequence length and feature dimensions invariant. Use when you want to inject
temporal position information into a model without adding a separate positional
encoding module.

Core API:
- Time2Vec: periodic temporal encoding with learned frequencies and phases
- Time2VecHead: BaseHead wrapper

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from foreblocks.core.model import BaseHead


class Time2Vec(nn.Module):
    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        self.periodic = bool(periodic)
        self.freq = nn.Parameter(torch.randn(1, 1, self.k))
        self.phase = nn.Parameter(torch.zeros(1, 1, self.k))
        in_dim = self.feature_dim + 1 + (self.k if self.periodic else 0)
        self.proj = nn.Linear(in_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if self.feature_dim != F_:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1)
        parts = [x, t]
        if self.periodic and self.k > 0:
            z = torch.sin(2 * math.pi * (t * self.freq + self.phase))  # [1,T,k]
            parts.append(z.expand(B, -1, -1))
        h = torch.cat(parts, dim=-1)
        return self.proj(h)


class Time2VecHead(BaseHead):
    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__(module=Time2Vec(feature_dim, k, periodic), name="time2vec")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Differencing (reversible)
# ──────────────────────────────────────────────────────────────────────────────
