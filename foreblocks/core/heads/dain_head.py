from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class DAIN(nn.Module):
    """
    Deep Adaptive Input Normalization (DAIN).
    Applies adaptive shift, scale, and gating based on time summaries.
    Input/Output: [B,T,F]
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.W_a = nn.Parameter(torch.eye(self.feature_dim))
        self.W_b = nn.Parameter(torch.eye(self.feature_dim))
        self.W_c = nn.Parameter(torch.randn(self.feature_dim, self.feature_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(self.feature_dim))
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}.")

        a = x.mean(dim=1)  # [B,F]
        alpha = a @ self.W_a.t()
        shifted = x - alpha.unsqueeze(1)

        b = torch.sqrt(torch.mean(shifted**2, dim=1) + self.eps)
        beta = b @ self.W_b.t()
        scaled = shifted / (beta.unsqueeze(1) + self.eps)

        c = scaled.mean(dim=1)
        gate_input = (c @ self.W_c.t()) + self.d
        gamma = torch.sigmoid(gate_input)
        return scaled * gamma.unsqueeze(1)


class DAINHead(BaseHead):
    """BaseHead wrapper for DAIN. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int):
        super().__init__(module=DAIN(feature_dim=feature_dim), name="dain")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding (FIXED upsample)
# ──────────────────────────────────────────────────────────────────────────────
