"""foreblocks.anomaly.models.tranad.

This module implements the tranad pieces for its package.
It belongs to the forecasting, anomaly, and backbone model definitions area of Foreblocks.
It exposes classes such as TranAD.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from foreblocks.models.transformer.transformer import (
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)


class TranAD(nn.Module):
    def __init__(
        self,
        feats,
        window_size=24,
        d_model=None,
        n_heads=None,
        n_layers=1,
        dropout=0.1,
    ):
        super().__init__()
        self.n_feats = int(feats)
        self.n_window = int(window_size)
        self.d_model = int(d_model or max(2 * self.n_feats, 32))

        if n_heads is None:
            n_heads = min(max(1, self.n_feats), self.d_model)
            while self.d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
        self.n_heads = int(max(1, n_heads))

        self.input_projection = nn.Linear(2 * self.n_feats, self.d_model)
        self.pos_encoder = _TranADPositionalEncoding(
            self.d_model, dropout=dropout, max_len=max(512, self.n_window + 2)
        )

        layer_kwargs = {
            "d_model": self.d_model,
            "nhead": self.n_heads,
            "dim_feedforward": max(16, self.d_model),
            "dropout": dropout,
            "activation": "gelu",
            "pos_encoding_type": "sinusoidal",
        }
        self.transformer_encoder = nn.ModuleList(
            TransformerEncoderLayer(**layer_kwargs) for _ in range(n_layers)
        )
        self.transformer_decoder1 = nn.ModuleList(
            TransformerDecoderLayer(**layer_kwargs) for _ in range(n_layers)
        )
        self.transformer_decoder2 = nn.ModuleList(
            TransformerDecoderLayer(**layer_kwargs) for _ in range(n_layers)
        )
        self._materialize_attention_modules()
        self.output_projection = nn.Sequential(
            nn.Linear(self.d_model, self.n_feats),
            nn.Sigmoid(),
        )

    def _materialize_attention_modules(self) -> None:
        for layers in (
            self.transformer_encoder,
            self.transformer_decoder1,
            self.transformer_decoder2,
        ):
            for layer in layers:
                layer._self_attn()

    def _run_encoder(self, x: torch.Tensor) -> torch.Tensor:
        streams = None
        for layer in self.transformer_encoder:
            x, streams = layer(x, streams=streams)
        return x

    def _run_decoder(
        self, x: torch.Tensor, memory: torch.Tensor, layers: nn.ModuleList
    ) -> torch.Tensor:
        streams = None
        for layer in layers:
            x, _, streams = layer(x, memory, streams=streams)
        return x

    def encode(self, src, context, tgt):
        enc_in = torch.cat((src, context), dim=-1)
        enc_in = self.input_projection(enc_in)
        enc_in = self.pos_encoder(enc_in)
        memory = self._run_encoder(enc_in)

        dec_in = torch.cat((tgt, tgt), dim=-1)
        dec_in = self.input_projection(dec_in)
        dec_in = self.pos_encoder(dec_in)
        return dec_in, memory

    def forward(self, src, tgt=None):
        if tgt is None:
            tgt = src[:, -1:, :]

        zero_context = torch.zeros_like(src)
        dec1_in, memory1 = self.encode(src, zero_context, tgt)
        out1 = self.output_projection(
            self._run_decoder(dec1_in, memory1, self.transformer_decoder1)
        )

        focus = (out1 - src).pow(2)
        dec2_in, memory2 = self.encode(src, focus, tgt)
        out2 = self.output_projection(
            self._run_decoder(dec2_in, memory2, self.transformer_decoder2)
        )
        return out1, out2


class _TranADPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / max(1, d_model))
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


__all__ = ["TranAD"]
