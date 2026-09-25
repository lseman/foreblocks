"""Small native neural architectures used by fitted tabular anomaly scorers."""

from itertools import pairwise

import torch
from torch import nn


def mlp(dimensions, *, bias=True):
    layers = []
    for i, (left, right) in enumerate(pairwise(dimensions)):
        layers.append(nn.Linear(left, right, bias=bias))
        if i < len(dimensions) - 2:
            layers.append(nn.LeakyReLU(0.1))
    return nn.Sequential(*layers)


class TabularAutoEncoder(nn.Module):
    def __init__(self, n_features, hidden_sizes, latent_dim):
        super().__init__()
        self.encoder = mlp((n_features, *hidden_sizes, latent_dim))
        self.decoder = mlp((latent_dim, *reversed(hidden_sizes), n_features))

    def forward(self, x):
        return self.decoder(self.encoder(x))


class TabularVAE(nn.Module):
    def __init__(self, n_features, hidden_sizes, latent_dim):
        super().__init__()
        self.encoder = mlp((n_features, *hidden_sizes))
        self.mu = nn.Linear(hidden_sizes[-1], latent_dim)
        self.logvar = nn.Linear(hidden_sizes[-1], latent_dim)
        self.decoder = mlp((latent_dim, *reversed(hidden_sizes), n_features))

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.mu(h), self.logvar(h).clamp(-20, 10)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        return self.decoder(z), mu, logvar


class SVDDEncoder(nn.Module):
    """Bias-free encoder, fixed nonzero center, optional soft-boundary radius."""

    def __init__(self, n_features, hidden_sizes, latent_dim):
        super().__init__()
        self.encoder = mlp((n_features, *hidden_sizes, latent_dim), bias=False)
        self.register_buffer("center", torch.zeros(latent_dim))
        self.register_buffer("radius", torch.zeros(()))

    def forward(self, x):
        return self.encoder(x)
