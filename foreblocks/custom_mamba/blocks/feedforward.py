from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """Position-wise SwiGLU feed-forward block with pre-norm.

    Parameter count is set by ``expansion`` (default ``8/3``) rounded up to the
    nearest multiple of 64, matching the LLaMA / Mistral convention.

    Args:
        d_model: Input/output dimension.
        expansion: Inner dim multiplier (default ``8/3``).
        dropout: Dropout after the output projection.
    """

    def __init__(self, d_model: int, expansion: float = 8 / 3, dropout: float = 0.0):
        super().__init__()
        d_inner = max(int(d_model * expansion), 64)
        d_inner = (d_inner + 63) // 64 * 64
        self.norm = nn.LayerNorm(d_model)
        self.w1 = nn.Linear(d_model, d_inner, bias=False)
        self.w2 = nn.Linear(d_model, d_inner, bias=False)
        self.out = nn.Linear(d_inner, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return self.drop(self.out(F.silu(self.w1(h)) * self.w2(h)))
