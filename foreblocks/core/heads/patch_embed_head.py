from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class PatchEmbed(nn.Module):
    """
    Local patching via depthwise Conv1d, stride=patch_size (non-overlap), then upsample.
    Shape preserved: [B,T,F] -> [B,T,F] (residual added).
    """

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.patch_proj = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            groups=self.feature_dim,
            bias=True,
            padding=0,
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}.")
        if T % self.patch_size != 0:
            raise ValueError(f"T={T} must be divisible by patch_size={self.patch_size}.")

        xt = x.transpose(1, 2)  # [B,F,T]
        patches = self.patch_proj(xt)  # [B,F,T/patch]
        patches = self.dropout(patches)
        # Robust resize back to exactly T (no scale-factor drift)
        embedded = F.interpolate(patches, size=T, mode="linear", align_corners=False)  # [B,F,T]
        return embedded.transpose(1, 2) + x


@node(
    type_id="patchemb_head",
    name="PatchEmbedHead",
    category="Preprocessing",
    outputs=["patchemb_head"],
    color="bg-gradient-to-r from-blue-400 to-purple-500",
)
class PatchEmbedHead(BaseHead):
    """BaseHead wrapper for PatchEmbed. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__(module=PatchEmbed(feature_dim, patch_size, dropout), name="patchemb")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Haar Wavelet Top-K
# ──────────────────────────────────────────────────────────────────────────────
