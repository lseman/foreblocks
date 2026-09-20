"""Construction helpers shared by transformer encoder and decoder stacks."""

from __future__ import annotations

from collections.abc import Callable

import torch.nn as nn

from foreblocks.layers.embeddings import LearnablePositionalEncoding, PositionalEncoding


def build_positional_encoder(
    *,
    d_model: int,
    max_seq_len: int,
    dropout: float,
    encoding_type: str,
    attention_mode: str,
    scale: float,
    supplied: nn.Module | None,
) -> nn.Module | None:
    """Build the input-level positional encoder, if the backend needs one."""
    if supplied is not None:
        return supplied
    if encoding_type not in {"sinusoidal", "learnable"}:
        return None
    if attention_mode == "sype":
        return LearnablePositionalEncoding(
            d_model,
            max_len=max_seq_len,
            dropout=dropout,
            scale_strategy="fixed",
            scale_value=scale,
            use_layer_norm=False,
        )
    return PositionalEncoding(d_model, max_len=max_seq_len, scale=scale)


def build_layer_modules(
    *,
    num_layers: int,
    dropout: float,
    share_layers: bool,
    attention_type_for: Callable[[int], str],
    layer_factory: Callable[[str, float], nn.Module],
    dropout_for: Callable[[int], float] | None = None,
) -> tuple[nn.Module | None, nn.ModuleList | None]:
    """Construct shared or per-depth layers while preserving module layout."""
    if share_layers:
        shared = layer_factory(attention_type_for(0), dropout)
        return shared, None
    layers = nn.ModuleList(
        layer_factory(
            attention_type_for(index),
            dropout_for(index) if dropout_for is not None else dropout,
        )
        for index in range(num_layers)
    )
    return None, layers


__all__ = ["build_layer_modules", "build_positional_encoder"]
