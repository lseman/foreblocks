"""foreblocks.anomaly.models.base.

Base classes and utilities for anomaly detection models.

Provides the VAEForward dataclass for VAE outputs and the ForeblocksEncoderStack
for transformer encoder layers. Includes the choose_heads helper for determining
optimal attention head counts based on model dimension.

Core API:
- VAEForward: dataclass for VAE forward outputs (reconstruction, mu, logvar)
- ForeblocksEncoderStack: transformer encoder layer stack
- choose_heads: helper to determine optimal attention head count

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from foreblocks.models.transformer.transformer import TransformerEncoderLayer


@dataclass
class VAEForward:
    reconstruction: torch.Tensor
    mu: torch.Tensor
    logvar: torch.Tensor


def choose_heads(d_model: int, requested: int | None = None) -> int:
    if requested is not None:
        n_heads = max(1, int(requested))
    else:
        n_heads = min(8, max(1, int(d_model) // 32))
    while d_model % n_heads != 0 and n_heads > 1:
        n_heads -= 1
    return n_heads


class ForeblocksEncoderStack(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        layer_attention_type: str = "standard",
    ) -> None:
        super().__init__()
        kwargs = {
            "d_model": int(d_model),
            "nhead": int(n_heads),
            "dim_feedforward": int(dim_feedforward),
            "dropout": float(dropout),
            "activation": "gelu",
            "layer_attention_type": layer_attention_type,
            "pos_encoding_type": "sinusoidal",
        }
        self.layers = nn.ModuleList(
            TransformerEncoderLayer(**kwargs) for _ in range(int(n_layers))
        )
        for layer in self.layers:
            layer._self_attn()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        streams = None
        for layer in self.layers:
            x, streams = layer(x, streams=streams)
        return x
