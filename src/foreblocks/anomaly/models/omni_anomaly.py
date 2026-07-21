"""foreblocks.anomaly.models.omni_anomaly.

GRU-VAE model for multivariate window anomaly detection.

Uses a GRU-based variational autoencoder to learn the joint distribution of normal
multivariate windows in the latent space. Anomaly is detected via reconstruction error
and KL divergence from the prior. Inspired by OmniAnomaly, it captures both temporal
dependencies across timesteps and cross-feature correlations. Use when you need a
compact RNN-based model for multivariate anomaly detection without transformers.

Core API:
- OmniAnomaly: GRU-VAE backbone for multivariate anomaly detection

"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.anomaly.models.base import VAEForward


class OmniAnomaly(nn.Module):
    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 128,
        latent_size: int = 32,
        n_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        hidden_size = max(16, int(hidden_size))
        latent_size = max(2, int(latent_size))
        n_layers = max(1, int(n_layers))
        rnn_dropout = float(dropout) if n_layers > 1 else 0.0

        self.encoder = nn.GRU(
            self.n_features,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.latent_to_hidden = nn.Linear(latent_size, hidden_size)
        self.decoder = nn.GRU(
            latent_size,
            hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, self.n_features),
        )

    def forward(self, x: torch.Tensor) -> VAEForward:
        _, h = self.encoder(x)
        h_last = h[-1]
        mu = self.mu(h_last)
        logvar = self.logvar(h_last).clamp(-10.0, 8.0)
        z = self.reparameterize(mu, logvar)
        return VAEForward(self._decode(z), mu, logvar)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not mu.requires_grad:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        seq = z.unsqueeze(1).expand(-1, self.window_size, -1)
        h0 = torch.tanh(self.latent_to_hidden(z)).unsqueeze(0)
        out, _ = self.decoder(seq, h0)
        return self.output(out)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder(x)
        return self._decode(self.mu(h[-1]))


__all__ = ["OmniAnomaly"]
