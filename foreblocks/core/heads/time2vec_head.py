from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class Time2Vec(nn.Module):
    """
    Time2Vec-style periodic features per timestep, projected back to feature_dim.
    Input/Output: [B,T,F]. Keeps dim invariant.
    """

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
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1)
        parts = [x, t]
        if self.periodic and self.k > 0:
            z = torch.sin(2 * math.pi * (t * self.freq + self.phase))  # [1,T,k]
            parts.append(z.expand(B, -1, -1))
        h = torch.cat(parts, dim=-1)
        return self.proj(h)


class Time2VecHead(BaseHead):
    """BaseHead wrapper for Time2Vec. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__(module=Time2Vec(feature_dim, k, periodic), name="time2vec")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Differencing (reversible)
# ──────────────────────────────────────────────────────────────────────────────
