"""CrossFormer-style time series forecasting head.

Cross-scale attention architecture that processes temporal and channel dimensions
with separate multi-head attention streams, enabling fine-grained feature learning
for multi-variate time series forecasting.

Based on: CrossFormer, a cross-scale attention architecture for time series.
Paper: https://openreview.net/pdf?id=vSVLM2j9eie

Core API:
- CrossFormer: lightweight cross-scale transformer head with temporal and channel attention

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.modules.attention.multi_att import MultiAttention


class CrossFormer(nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int,
        out_channels: int | None = None,
        d_model: int = 256,
        n_heads: int = 8,
        hidden: int = 512,
        dropout: float = 0.1,
        att_type: str = "standard",
        freq_modes: int = 32,
        use_swiglu: bool = True,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.in_channels
        )

        self.temporal_proj = nn.Linear(self.in_channels, d_model)
        self.channel_proj = nn.Conv1d(1, d_model, kernel_size=1)

        self.temporal_attn = MultiAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=True,
        )
        self.channel_attn = MultiAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            attention_type=att_type,
            freq_modes=freq_modes,
            cross_attention=True,
        )

        self.temporal_ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.channel_ff = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )

        self.horizon_mlp = nn.Sequential(
            nn.Linear(d_model * 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.pred_len * d_model),
        )
        self.output_proj = nn.Linear(d_model, self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        temp = self.temporal_proj(x)
        chan = x.permute(0, 2, 1).reshape(B * C, 1, L)
        chan = self.channel_proj(chan).mean(dim=-1).view(B, C, -1)

        temp_cross, _, _ = self.temporal_attn(temp, chan, chan)
        chan_cross, _, _ = self.channel_attn(chan, temp, temp)

        temp_out = temp + self.temporal_ff(temp_cross)
        chan_out = chan + self.channel_ff(chan_cross)

        pooled = torch.cat([temp_out.mean(dim=1), chan_out.mean(dim=1)], dim=-1)
        h = self.horizon_mlp(pooled).view(B, self.pred_len, -1)
        y = self.output_proj(h)
        return y
