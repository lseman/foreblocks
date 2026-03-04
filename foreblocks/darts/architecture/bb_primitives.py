"""Primitive building blocks: RMSNorm, SwiGLUFFN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RMSNorm", "SwiGLUFFN"]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward block."""

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        mid = dim * expand
        self.w1 = nn.Linear(dim, mid, bias=False)
        self.w2 = nn.Linear(dim, mid, bias=False)
        self.w3 = nn.Linear(mid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))
