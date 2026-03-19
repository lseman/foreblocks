"""Primitive building blocks: RMSNorm, SwiGLUFFN, GeGLUFFN, ReluFFN."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RMSNorm", "SwiGLUFFN", "GeGLUFFN", "ReluFFN"]


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
    """SwiGLU feed-forward block.

    Hidden dimension is scaled by 2/3 so that the total parameter count
    (3 weight matrices) matches a standard ReLU FFN with the same ``expand``
    factor (2 weight matrices).  This follows the LLaMA / Mistral convention:
    ``mid = round_to(2/3 * expand * dim)``.
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        # Parameter-equivalent hidden size: 3*dim*mid == 2*dim*(expand*dim)
        # → mid = 2/3 * expand * dim, rounded up to nearest multiple of 64.
        mid_raw = max(dim, int(2 * dim * expand // 3))
        mid = ((mid_raw + 63) // 64) * 64
        self.w1 = nn.Linear(dim, mid, bias=False)
        self.w2 = nn.Linear(dim, mid, bias=False)
        self.w3 = nn.Linear(mid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class GeGLUFFN(nn.Module):
    """GeGLU feed-forward block (Shazeer 2020).

    Uses GELU gate instead of SiLU.  Same 2/3-scaled hidden dim as SwiGLUFFN
    to keep parameter counts comparable across FFN variants.
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        mid_raw = max(dim, int(2 * dim * expand // 3))
        mid = ((mid_raw + 63) // 64) * 64
        self.w1 = nn.Linear(dim, mid, bias=False)
        self.w2 = nn.Linear(dim, mid, bias=False)
        self.w3 = nn.Linear(mid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w3(F.gelu(self.w1(x)) * self.w2(x))


class ReluFFN(nn.Module):
    """Standard ReLU² feed-forward block (So et al. 2021 — Primer).

    ReLU² often matches SwiGLU quality with simpler gradient flow,
    and is the cheapest FFN variant — useful as a lightweight baseline
    in the FFN search.
    """

    def __init__(self, dim: int, expand: int = 4):
        super().__init__()
        mid = dim * expand
        self.w1 = nn.Linear(dim, mid, bias=False)
        self.w2 = nn.Linear(mid, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.relu(self.w1(x)).pow(2))
