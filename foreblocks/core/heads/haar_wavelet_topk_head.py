from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

class HaarWaveletTopK(nn.Module):
    """
    1-level Haar wavelet analysis with Top-K keep on detail coefficients.
    Returns (main, detail_sparse) with shape [B,T,F].
    """

    def __init__(self, topk: int = 8):
        super().__init__()
        self.topk = int(topk)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        pad_added = False
        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1, 0, 0), mode="replicate")
            B, T, F_ = x.shape
            pad_added = True

        x_even = x[:, 0::2, :]
        x_odd = x[:, 1::2, :]
        x_low = (x_even + x_odd) / math.sqrt(2.0)   # [B,T/2,F]
        x_high = (x_even - x_odd) / math.sqrt(2.0)  # [B,T/2,F]

        k = min(self.topk, x_high.size(1))
        if k > 0:
            mag = x_high.abs()
            idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
            mask = torch.zeros_like(mag, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            xh_sparse = torch.where(mask, x_high, torch.zeros_like(x_high))
        else:
            xh_sparse = torch.zeros_like(x_high)

        out_len = T
        main = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)
        detail_sparse = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)

        # main = inverse with detail=0
        main[:, 0::2, :] = (x_low + 0.0) / math.sqrt(2.0)
        main[:, 1::2, :] = (x_low - 0.0) / math.sqrt(2.0)

        # sparse detail = inverse with low=0
        detail_sparse[:, 0::2, :] = (0.0 + xh_sparse) / math.sqrt(2.0)
        detail_sparse[:, 1::2, :] = (0.0 - xh_sparse) / math.sqrt(2.0)

        if pad_added:
            main = main[:, :-1, :]
            detail_sparse = detail_sparse[:, :-1, :]

        return main, detail_sparse


class HaarWaveletTopKHead(BaseHead):
    """BaseHead wrapper for HaarWaveletTopK. Forward -> (main, detail_sparse)."""

    def __init__(self, topk: int = 8):
        super().__init__(module=HaarWaveletTopK(topk=topk), name="haar_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time Attention (FIXED)
# Channel-independent self-attention over time for each feature stream.
# Input/Output: [B,T,F]
# ──────────────────────────────────────────────────────────────────────────────
