"""foreblocks.anomaly.models.reconstruction.

Reconstruction-based anomaly detectors using autoencoders and VAEs.

Provides compact MLP-based and transformer-based VAEs that learn to reconstruct normal
windows. Anomaly is scored via reconstruction error — windows that the model cannot
reconstruct well are anomalous. The MLPVAE offers low-latency baselines; the
TransformerVAE captures temporal dependencies via encoder blocks. Use when you need
a reconstruction baseline that is fast to train and easy to interpret.

Core API:
- MLPVAE: compact window VAE baseline for low-latency scoring
- TransformerVAE: VAE with Foreblocks transformer encoder for temporal modeling

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.anomaly.models.base import (
    ForeblocksEncoderStack,
    VAEForward,
    choose_heads,
)


class MLPVAE(nn.Module):
    """Compact window VAE baseline for low-latency anomaly scoring."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 128,
        latent_size: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        input_size = self.n_features * self.window_size
        hidden_size = max(16, int(hidden_size))
        latent_size = max(2, int(latent_size))

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> VAEForward:
        flat = x.reshape(x.shape[0], -1)
        h = self.encoder(flat)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 8.0)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z).reshape_as(x)
        return VAEForward(recon, mu, logvar)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not mu.requires_grad:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        h = self.encoder(flat)
        return self.decoder(self.mu(h)).reshape_as(x)


class TransformerVAE(nn.Module):
    """Window VAE with Foreblocks transformer encoder blocks."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        latent_size: int = 32,
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
        latent_size = max(2, int(latent_size))
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
        pooled_size = self.window_size * self.d_model
        self.pre_latent = nn.Sequential(
            nn.LayerNorm(pooled_size),
            nn.Linear(pooled_size, self.d_model),
            nn.GELU(),
        )
        self.mu = nn.Linear(self.d_model, latent_size)
        self.logvar = nn.Linear(self.d_model, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, pooled_size),
            nn.GELU(),
            nn.Unflatten(1, (self.window_size, self.d_model)),
            nn.Linear(self.d_model, self.n_features),
        )

    def forward(self, x: torch.Tensor) -> VAEForward:
        h = self.input_projection(x) + self.position[:, : x.shape[1]]
        h = self.encoder(h)
        h = self.pre_latent(h.reshape(h.shape[0], -1))
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-10.0, 8.0)
        z = self.reparameterize(mu, logvar)
        return VAEForward(self.decoder(z), mu, logvar)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not mu.requires_grad:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_projection(x) + self.position[:, : x.shape[1]]
        h = self.encoder(h)
        h = self.pre_latent(h.reshape(h.shape[0], -1))
        return self.decoder(self.mu(h))
