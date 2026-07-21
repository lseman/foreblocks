"""foreblocks.ts_handler.auto_filter.filters.deep.

Denoising autoencoder and variational autoencoder filters.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from foreblocks.ts_handler.auto_filter.filters.utils import (
    _as_series,
    _valid_odd_window,
)
from foreblocks.ts_handler.auto_filter.registry import register_filter

_SEED = 42


class _DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64) -> None:
        super().__init__()
        hidden_size = max(16, int(hidden_size))
        bottleneck = max(8, hidden_size // 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, bottleneck),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class _VariationalAutoencoder(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int = 64, latent_size: int = 8
    ) -> None:
        super().__init__()
        hidden_size = max(16, int(hidden_size))
        latent_size = max(2, int(latent_size))
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
        )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(-8.0, 8.0)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        return self.decoder(z), mu, logvar

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.decoder(self.mu(h))


def _build_windows(values: np.ndarray, window: int) -> np.ndarray:
    half = window // 2
    padded = np.pad(values, (half, half), mode="reflect")
    return np.stack([padded[i : i + window] for i in range(len(values))], axis=0)


@register_filter("Denoising Autoencoder", slow=True)
def train_dae(
    ts: pd.Series,
    window: int = 21,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 1e-3,
    noise_std: float = 0.15,
) -> pd.Series:
    rng_state = torch.get_rng_state()
    torch.manual_seed(_SEED)

    values = ts.values.astype(np.float32)
    window = _valid_odd_window(len(values), window, minimum=5)
    if window < 5:
        return ts.copy()

    x_clean = _build_windows(values, window)
    x_clean_t = torch.tensor(x_clean, dtype=torch.float32)
    x_noisy_t = x_clean_t + noise_std * torch.randn_like(x_clean_t)

    model = _DenoisingAutoencoder(input_size=window, hidden_size=min(64, window * 3))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n = x_clean_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = criterion(model(x_noisy_t[idx]), x_clean_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        center = window // 2
        denoised = model(x_clean_t).numpy()[:, center]

    torch.set_rng_state(rng_state)
    return _as_series(denoised, ts.index, name="dae")


@register_filter("Variational Autoencoder", slow=True)
def train_vae(
    ts: pd.Series,
    window: int = 25,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    noise_std: float = 0.12,
    beta: float = 0.02,
    latent_size: int = 8,
) -> pd.Series:
    rng_state = torch.get_rng_state()
    torch.manual_seed(_SEED)

    values = ts.values.astype(np.float32)
    window = _valid_odd_window(len(values), window, minimum=7)
    if window < 7:
        return ts.copy()

    x_clean = _build_windows(values, window)
    x_clean_t = torch.tensor(x_clean, dtype=torch.float32)
    x_noisy_t = x_clean_t + noise_std * torch.randn_like(x_clean_t)

    model = _VariationalAutoencoder(
        input_size=window,
        hidden_size=min(96, window * 4),
        latent_size=latent_size,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    n = x_clean_t.shape[0]
    for _ in range(int(epochs)):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            recon, mu, logvar = model(x_noisy_t[idx])
            recon_loss = torch.mean((recon - x_clean_t[idx]) ** 2)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + float(beta) * kl
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        center = window // 2
        denoised = model.reconstruct_mean(x_clean_t).numpy()[:, center]

    torch.set_rng_state(rng_state)
    return _as_series(denoised, ts.index, name="vae")
