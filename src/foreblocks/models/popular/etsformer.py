"""ETSformer-style time series forecasting head.

Combines exponential smoothing-based decomposition with a transformer encoder to
model seasonal and trend components separately. Decomposes input into trend and
seasonal parts, encodes seasonal via transformer, then projects both to forecast.

Based on: Wu et al., "ETSformer: Exponential Smoothing Transformer for Time Series Forecasting",
Paper: https://arxiv.org/abs/2306.04113

Core API:
- ETSformer: exponential smoothing transformer with seasonal-trend decomposition

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.embeddings import PositionalEncoding
from foreblocks.layers.norms import create_norm_layer
from foreblocks.models.transformer.transformer import TransformerEncoderLayer


class ETSformer(nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int,
        out_channels: int | None = None,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        att_type: str = "standard",
        freq_modes: int = 32,
        use_swiglu: bool = True,
        use_moe: bool = False,
        num_experts: int = 8,
        top_k: int = 2,
        layer_norm_eps: float = 1e-5,
        max_seq_len: int = 10000,
        ma_kernel: int = 25,
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.in_channels
        )
        self.ma_kernel = max(int(ma_kernel), 1)

        self.seasonal_in = nn.Linear(self.in_channels, d_model)
        self.pos_enc = PositionalEncoding(d_model=d_model, max_len=max_seq_len)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation=activation,
                    att_type=att_type,
                    freq_modes=freq_modes,
                    use_swiglu=use_swiglu,
                    norm_strategy="pre_norm",
                    custom_norm="rms",
                    layer_norm_eps=layer_norm_eps,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    top_k=top_k,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = create_norm_layer("rms", d_model, layer_norm_eps)

        self.seasonal_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pred_len * self.out_channels),
        )

        self.trend_proj = nn.Sequential(
            nn.Linear(self.in_channels, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.pred_len * self.out_channels),
        )

    @staticmethod
    def _moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1:
            return x
        B, L, C = x.shape
        x_n = x.permute(0, 2, 1)
        pad = (k - 1) // 2
        x_pad = F.pad(x_n, (pad, pad), mode="reflect")
        w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
        trend = F.conv1d(x_pad, w.expand(C, -1, -1), groups=C)
        return trend.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if self.in_channels != C:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        trend = self._moving_average(x, self.ma_kernel)
        seasonal = x - trend

        seasonal_tokens = self.seasonal_in(seasonal)
        seasonal_tokens = self.pos_enc(seasonal_tokens)
        h = seasonal_tokens
        for layer in self.layers:
            h, _ = layer(h, src_mask=None, src_key_padding_mask=None)
        h = self.final_norm(h)
        seasonal_pooled = h.mean(dim=1)
        seasonal_out = self.seasonal_proj(seasonal_pooled).view(
            B, self.pred_len, self.out_channels
        )

        trend_pooled = trend.mean(dim=1)
        trend_out = self.trend_proj(trend_pooled).view(
            B, self.pred_len, self.out_channels
        )

        return seasonal_out + trend_out
