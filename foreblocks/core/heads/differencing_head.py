from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class Differencing(nn.Module):
    """
    First-order differencing along time with length preservation.
    Forward: (delta, ctx) where delta[:,0,:]=0 and ctx['x0']=first step.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        delta = x.clone()
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta[:, :1, :] = 0.0
        ctx = {"x0": x[:, :1, :]}
        return delta, ctx

    def invert(self, y_hat: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = ctx["x0"]
        rec = torch.cumsum(y_hat, dim=1)
        rec[:, :1, :] = 0.0
        return rec + x0


class DifferencingHead(BaseHead):
    """BaseHead wrapper for Differencing. Forward -> (delta, ctx)."""

    def __init__(self):
        super().__init__(module=Differencing(), name="diff")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Fourier Seasonal
# ──────────────────────────────────────────────────────────────────────────────
