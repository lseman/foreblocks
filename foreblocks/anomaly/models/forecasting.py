"""foreblocks.anomaly.models.forecasting.

Forecasting-based anomaly detection using transformer prediction error.

Models predict the final step of a time series window given prior timesteps.
Anomaly is detected by high prediction error — windows that are difficult to forecast
are anomalous. Use when anomalies manifest as unusual patterns that deviate from
learned temporal dynamics, complementing reconstruction-based methods.

Core API:
- TransformerForecaster: transformer forecaster for anomaly scoring

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.anomaly.models.base import ForeblocksEncoderStack, choose_heads


class TransformerForecaster(nn.Module):
    """Forecasting-based anomaly model that predicts the final window step."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
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
        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = x[:, :-1, :]
        h = self.input_projection(context) + self.position[:, : context.shape[1]]
        h = self.encoder(h)
        return self.head(h[:, -1:, :])
