from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class RevIN(nn.Module):
    """
    Reversible Instance Normalization (per-variable, over time).
    Forward: returns (x_norm, ctx) where ctx has {mu, sigma}.
    invert(x_hat, ctx) -> original scale.
    """

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.affine = bool(affine)
        self.eps = float(eps)
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_features))
        else:
            self.register_buffer("gamma", torch.ones(1, 1, self.num_features))
            self.register_buffer("beta", torch.zeros(1, 1, self.num_features))

    @torch.no_grad()
    def _stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=1, keepdim=True)  # [B,1,F]
        var = x.var(dim=1, unbiased=False, keepdim=True)  # [B,1,F]
        sigma = torch.sqrt(var + self.eps)
        return mu, sigma

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu, sigma = self._stats(x)
        x_hat = (x - mu) / sigma
        x_hat = x_hat * self.gamma + self.beta
        ctx = {"mu": mu, "sigma": sigma}
        return x_hat, ctx

    def invert(self, x_hat: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = (x_hat - self.beta) / (self.gamma + 1e-12)
        return x * ctx["sigma"] + ctx["mu"]


@node(
    type_id="revin_head",
    name="RevINHead",
    category="Preprocessing",
    outputs=["revin_head"],
    color="bg-gradient-to-r from-yellow-400 to-red-500",
)
class RevINHead(BaseHead):
    """BaseHead wrapper for RevIN. Forward -> (x_norm, ctx)."""

    def __init__(self, feature_dim: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(
            module=RevIN(feature_dim, affine=affine, eps=eps), name="revin"
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Conv
# ──────────────────────────────────────────────────────────────────────────────
