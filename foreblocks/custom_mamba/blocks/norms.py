from __future__ import annotations

import torch
import torch.nn as nn


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
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x * rms * self.weight
