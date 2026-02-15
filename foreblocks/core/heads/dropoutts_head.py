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
    type_id="dropoutts_head",
    name="DropoutTSHead",
    category="Preprocessing",
    outputs=["dropoutts_head"],
    color="bg-gradient-to-r from-zinc-500 to-slate-700",
)
class DropoutTSHead(BaseHead):
    """
    Training-only DropoutTS head for time series.

    Modes:
      - "timestep": drop individual timesteps (token dropout)
      - "span":     drop contiguous spans along time
      - "feature":  drop whole features (channels)
      - "mixed":    timestep + feature (and optional spans)
    """

    def __init__(
        self,
        p_time: float = 0.1,
        p_feat: float = 0.0,
        mode: str = "span",
        span_len: int = 8,
        n_spans: int = 1,
        fill: str = "zero",
        scale_keep: bool = False,
    ):
        module = _DropoutTS(
            p_time=p_time,
            p_feat=p_feat,
            mode=mode,
            span_len=span_len,
            n_spans=n_spans,
            fill=fill,
            scale_keep=scale_keep,
        )
        super().__init__(module=module, name="dropoutts")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class _DropoutTS(nn.Module):
    def __init__(
        self,
        p_time: float,
        p_feat: float,
        mode: str,
        span_len: int,
        n_spans: int,
        fill: str,
        scale_keep: bool,
    ):
        super().__init__()
        if not (0.0 <= p_time <= 1.0):
            raise ValueError("p_time must be in [0,1]")
        if not (0.0 <= p_feat <= 1.0):
            raise ValueError("p_feat must be in [0,1]")
        if mode not in {"timestep", "span", "feature", "mixed"}:
            raise ValueError("mode must be one of: timestep, span, feature, mixed")
        if fill not in {"zero", "mean"}:
            raise ValueError("fill must be 'zero' or 'mean'")
        self.p_time = float(p_time)
        self.p_feat = float(p_feat)
        self.mode = mode
        self.span_len = int(max(1, span_len))
        self.n_spans = int(max(1, n_spans))
        self.fill = fill
        self.scale_keep = bool(scale_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.p_time == 0.0 and self.p_feat == 0.0):
            return x
        if x.dim() != 3:
            raise ValueError(f"DropoutTS expects [B,T,F], got {tuple(x.shape)}")

        B, T, F_ = x.shape
        device = x.device

        drop = torch.zeros(B, T, F_, dtype=torch.bool, device=device)

        if self.mode in {"timestep", "mixed"} and self.p_time > 0.0:
            dt = torch.rand(B, T, 1, device=device) < self.p_time
            drop |= dt.expand(B, T, F_)

        if self.mode in {"feature", "mixed"} and self.p_feat > 0.0:
            df = torch.rand(B, 1, F_, device=device) < self.p_feat
            drop |= df.expand(B, T, F_)

        if self.mode in {"span", "mixed"} and self.p_time > 0.0:
            L = min(self.span_len, T)
            for _ in range(self.n_spans):
                starts = torch.randint(0, max(1, T - L + 1), (B,), device=device)
                idx = starts.view(B, 1) + torch.arange(L, device=device).view(1, L)  # [B,L]
                idx = idx.clamp(max=T - 1)
                drop.scatter_(1, idx.unsqueeze(-1).expand(B, L, F_), True)

        if self.fill == "zero":
            y = x.masked_fill(drop, 0.0)
        else:
            mu = x.mean(dim=1, keepdim=True)  # [B,1,F]
            y = torch.where(drop, mu.expand_as(x), x)

        if self.scale_keep:
            keep_time = (1.0 - self.p_time) if self.mode in {"timestep", "mixed"} else 1.0
            keep_feat = (1.0 - self.p_feat) if self.mode in {"feature", "mixed"} else 1.0
            keep = max(1e-6, keep_time * keep_feat)
            y = y / keep

        return y
