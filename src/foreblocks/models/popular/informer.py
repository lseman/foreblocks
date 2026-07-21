"""Informer-style time series forecasting head.

Minimal Informer-style head using a full transformer encoder with time encoding,
pooled representation, and a projection head for horizon prediction. Suitable for
long-sequence time-series forecasting with efficient attention.

Based on: Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence
Time-Series Forecasting", AAAI 2021.
Paper: https://arxiv.org/abs/2012.07436

Core API:
- Informer: minimal Informer-style forecasting head with transformer encoder

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.models.transformer.transformer import TransformerEncoder


class Informer(nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int,
        out_channels: int | None = None,
        label_len: int = 0,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers_enc: int = 2,
        n_layers_dec: int = 2,
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
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.label_len = int(label_len)
        self.in_channels = int(in_channels)
        self.out_channels = (
            int(out_channels) if out_channels is not None else self.in_channels
        )

        self.encoder = TransformerEncoder(
            input_size=self.in_channels,
            d_model=d_model,
            nhead=n_heads,
            num_layers=n_layers_enc,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            att_type=att_type,
            freq_modes=freq_modes,
            use_swiglu=use_swiglu,
            norm_strategy="pre_norm",
            custom_norm="rms",
            layer_norm_eps=layer_norm_eps,
            max_seq_len=max_seq_len,
            use_final_norm=True,
            use_moe=use_moe,
            num_experts=num_experts,
            top_k=top_k,
            use_time_encoding=True,
            model_type="informer-like",
            patch_encoder=False,
        )

        self.output_proj = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, self.pred_len * self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels, got {C}")

        memory = self.encoder(x)
        if hasattr(memory, "last_hidden_state"):
            memory = memory.last_hidden_state
        pooled = memory.mean(dim=1)
        y = self.output_proj(pooled).view(B, self.pred_len, self.out_channels)
        return y
