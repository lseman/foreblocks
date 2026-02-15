from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

@node(
    type_id="decomposition_head",
    name="DecompositionHead",
    category="Preprocessing",
    outputs=["decomposition_head"],
    color="bg-gradient-to-r from-green-400 to-blue-500",
)
class DecompositionHead(nn.Module):
    """
    Series decomposition head for trend-seasonal separation.
    Inspired by Autoformer: learnable moving average (depthwise Conv1d) per feature.
    Forward: (seasonal, trend)  with shape [B,T,F].
    """

    def __init__(
        self,
        kernel_size: int = 25,
        feature_dim: int = None,  # Input features F (required)
        hidden_dim: Optional[int] = None,  # Optional projection on seasonal
        groups: Optional[int] = None,
    ):
        super().__init__()
        if feature_dim is None:
            raise ValueError("feature_dim must be provided (e.g., 32)")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.feature_dim
        self.kernel_size = int(kernel_size)

        if groups is None:
            groups = self.feature_dim  # per-channel filtering
        groups = int(groups)
        if self.feature_dim % groups != 0:
            raise ValueError(
                f"groups must divide feature_dim (feature_dim={self.feature_dim}, groups={groups})"
            )

        padding = self.kernel_size // 2
        self.decomp = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # Initialize as exact moving average (constant)
        with torch.no_grad():
            self.decomp.weight.fill_(1.0 / float(self.kernel_size))

        self.post_proj = (
            nn.Linear(self.feature_dim, self.hidden_dim)
            if self.hidden_dim != self.feature_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,F]
        F_in = x.size(-1)
        if F_in != self.feature_dim:
            raise RuntimeError(
                f"Input feature dim {F_in} != expected {self.feature_dim}."
            )

        xt = x.transpose(1, 2)  # [B,F,T] for Conv1d
        trend = self.decomp(xt)  # [B,F,T]
        seasonal = xt - trend  # [B,F,T]
        seasonal = seasonal.transpose(1, 2)  # [B,T,F]
        trend = trend.transpose(1, 2)  # [B,T,F]

        seasonal = self.post_proj(seasonal)  # optional projection
        return seasonal, trend


class DecompositionBlock(BaseHead):
    """BaseHead wrapper for DecompositionHead."""

    def __init__(
        self,
        kernel_size: int = 25,
        feature_dim: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        decomp_module = DecompositionHead(
            kernel_size=kernel_size, feature_dim=feature_dim, hidden_dim=hidden_dim
        )
        super().__init__(module=decomp_module, name="decomposition")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# RevIN (Reversible Instance Normalization)
# ──────────────────────────────────────────────────────────────────────────────
