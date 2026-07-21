"""foreblocks.modules.heads.modules.learnable_fourier_seasonal_head.

Learnable per-channel seasonal decomposition via Fourier bases.

Projects input into a seasonal component (learnable linear combination of
sin/cos bases up to K frequencies) and a remainder main component. Use when
you need a parameterized, data-driven seasonal extractor that adapts during
training rather than relying on fixed sinusoidal bases.

Core API:
- LearnableFourierSeasonal: learnable seasonal via sin/cos Fourier bases
- LearnableFourierSeasonalHead: BaseHead wrapper returning (main, seasonal)

"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from foreblocks.core.model import BaseHead


class LearnableFourierSeasonal(nn.Module):
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        if self.feature_dim != F_:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        Bx = self._bases(T, x.device, x.dtype)  # [T,2K]
        W = self.W.expand(F_, -1) if self.W.size(0) == 1 else self.W  # [F,2K]
        seasonal = (Bx @ W.t()).unsqueeze(0).expand(B, -1, -1)  # [B,T,F]
        main = x - seasonal
        return main, seasonal


class LearnableFourierSeasonalHead(BaseHead):
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
