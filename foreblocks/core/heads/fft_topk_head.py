from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class FFTTopK(nn.Module):
    """
    Keep top-K magnitudes in frequency domain as seasonal carry; residual is main.
    Input/Output: [B,T,F]. Returns (main, seasonal).
    """

    def __init__(self, topk: int = 8):
        super().__init__()
        self.topk = int(topk)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        Xf = fft.rfft(x, dim=1)  # [B, Tr, F]
        mag = Xf.abs()

        k = min(self.topk, mag.size(1))
        if k <= 0:
            seasonal = torch.zeros_like(x)
            return x, seasonal

        topk_idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_idx, value=True)

        Xf_seasonal = torch.where(mask, Xf, torch.zeros_like(Xf))
        Xf_residual = Xf - Xf_seasonal

        seasonal = fft.irfft(Xf_seasonal, n=T, dim=1)
        main = fft.irfft(Xf_residual, n=T, dim=1)
        return main, seasonal


class FFTTopKHead(BaseHead):
    """BaseHead wrapper for FFTTopK. Forward -> (main, seasonal)."""

    def __init__(self, topk: int = 8):
        super().__init__(module=FFTTopK(topk=topk), name="fft_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time2Vec
# ──────────────────────────────────────────────────────────────────────────────
