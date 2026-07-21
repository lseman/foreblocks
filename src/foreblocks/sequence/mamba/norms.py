"""foreblocks.sequence.mamba.norms.

RMS normalization helpers for Mamba blocks.

Provides RMSNormWeightOnly (parameter-only, for use with fused_out) and
full RMSNorm wrapping the foreblocks rms_norm kernel. Used for per-head
normalization and the RMSNormGated pattern (rms_norm(y,w) * silu(z)).

Core API:
- RMSNormWeightOnly: parameter-only RMSNorm for fused_out usage
- RMSNorm: full RMSNorm with per-head weight and epsilon

"""

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
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, eps=self.eps)
