from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class LearnableFourierSeasonal(nn.Module):
    """
    Learnable per-channel seasonal component via Fourier bases (sin/cos up to K).
    seasonal = B @ W, main = x - seasonal. Shapes [B,T,F].
    """

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.K = int(K)
        in_dim = 2 * self.K
        if share_weights:
            self.W = nn.Parameter(torch.randn(1, in_dim))
        else:
            self.W = nn.Parameter(torch.randn(self.feature_dim, in_dim))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def _bases(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(-1)  # [T,1]
        ks = torch.arange(1, self.K + 1, device=device, dtype=dtype).view(1, self.K)
        ang = 2 * math.pi * t * ks / float(T)
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        return torch.cat([sin, cos], dim=-1)  # [T,2K]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        Bx = self._bases(T, x.device, x.dtype)  # [T,2K]
        W = self.W.expand(F_, -1) if self.W.size(0) == 1 else self.W  # [F,2K]
        seasonal = (Bx @ W.t()).unsqueeze(0).expand(B, -1, -1)  # [B,T,F]
        main = x - seasonal
        return main, seasonal


class LearnableFourierSeasonalHead(BaseHead):
    """BaseHead wrapper for LearnableFourierSeasonal. Forward -> (main, seasonal)."""

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__(
            module=LearnableFourierSeasonal(feature_dim, K, share_weights),
            name="lfourier",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# DAIN
# ──────────────────────────────────────────────────────────────────────────────
