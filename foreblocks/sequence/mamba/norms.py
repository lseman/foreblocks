from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.ops.mamba import rms_norm


class RMSNormWeightOnly(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        raise RuntimeError("Use fused_out(...) instead.")


class RMSNorm(nn.Module):
    """Small RMSNorm helper used for per-head Q/K normalisation."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, eps=self.eps)
