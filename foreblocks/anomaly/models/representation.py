"""foreblocks.anomaly.models.representation.

This module implements the representation pieces for its package.
It belongs to the forecasting, anomaly, and backbone model definitions area of Foreblocks.
It exposes classes such as ContrastiveTransformerEncoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.anomaly.models.base import ForeblocksEncoderStack, choose_heads


class ContrastiveTransformerEncoder(nn.Module):
    """Representation model trained with simple augmented-window contrast."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        projection_size: int = 64,
        n_heads: int | None = None,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        layer_attention_type: str = "standard",
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.d_model = int(d_model)
        n_heads = choose_heads(self.d_model, n_heads)
        dim_feedforward = int(dim_feedforward or max(4 * self.d_model, 128))

        self.input_projection = nn.Linear(self.n_features, self.d_model)
        self.position = nn.Parameter(torch.zeros(1, self.window_size, self.d_model))
        self.encoder = ForeblocksEncoderStack(
            d_model=self.d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_attention_type=layer_attention_type,
        )
        self.projector = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, max(int(projection_size), 8)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x) + self.position[:, : x.shape[1]]
        h = self.encoder(h).mean(dim=1)
        return F.normalize(self.projector(h), dim=-1)
