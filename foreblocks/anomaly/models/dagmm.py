"""foreblocks.anomaly.models.dagmm.

Deep autoencoding Gaussian mixture model for window-level density scoring.

DAGMM jointly trains an autoencoder and a Gaussian mixture model in the latent space.
Anomaly is scored via sampling energy — the negative log-likelihood under the learned
mixture distribution. Low-energy (high-density) windows are normal; high-energy windows
are anomalous. Use when you need a compact, end-to-end trainable model that scores
anomalies via both reconstruction error and latent-space density.

Core API:
- DAGMM: joint autoencoder + GMM anomaly detector
- DAGMMForward: forward output with reconstruction, latent, and assignment
- DAGMM.gmm_params: extract GMM parameters from latent/assignment
- DAGMM.sample_energy: compute density-based anomaly energy

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DAGMMForward:
    reconstruction: torch.Tensor
    latent: torch.Tensor
    gamma: torch.Tensor


class DAGMM(nn.Module):
    """Deep autoencoding Gaussian mixture model for window density scoring."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        hidden_size: int = 128,
        latent_size: int = 8,
        n_components: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.input_size = self.n_features * self.window_size
        hidden_size = max(16, int(hidden_size))
        latent_size = max(2, int(latent_size))
        self.n_components = max(1, int(n_components))

        self.encoder = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(latent_size * 2, 8)),
            nn.Tanh(),
            nn.Linear(max(latent_size * 2, 8), latent_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_size, max(latent_size * 2, 8)),
            nn.Tanh(),
            nn.Linear(max(latent_size * 2, 8), hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.input_size),
        )
        estimation_size = latent_size + 2
        self.estimation = nn.Sequential(
            nn.Linear(estimation_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.n_components),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> DAGMMForward:
        flat = x.reshape(x.shape[0], -1)
        latent = self.encoder(flat)
        recon_flat = self.decoder(latent)
        recon = recon_flat.reshape_as(x)
        z = self._estimation_features(flat, recon_flat, latent)
        gamma = self.estimation(z)
        return DAGMMForward(recon, z, gamma)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(x.shape[0], -1)
        return self.decoder(self.encoder(flat)).reshape_as(x)

    @staticmethod
    def _estimation_features(
        flat: torch.Tensor, recon_flat: torch.Tensor, latent: torch.Tensor
    ) -> torch.Tensor:
        rel_euclidean = torch.norm(flat - recon_flat, dim=1, keepdim=True) / (
            torch.norm(flat, dim=1, keepdim=True) + 1e-8
        )
        cosine = F.cosine_similarity(flat, recon_flat, dim=1).unsqueeze(1)
        return torch.cat([latent, rel_euclidean, cosine], dim=1)

    @staticmethod
    def gmm_params(
        z: torch.Tensor, gamma: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma_sum = gamma.sum(dim=0) + 1e-8
        phi = gamma_sum / z.shape[0]
        mu = gamma.T @ z / gamma_sum.unsqueeze(1)
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        var = (gamma.unsqueeze(2) * diff.pow(2)).sum(dim=0) / gamma_sum.unsqueeze(1)
        return phi.clamp_min(1e-8), mu, var.clamp_min(1e-6)

    @staticmethod
    def sample_energy(
        z: torch.Tensor, phi: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
    ) -> torch.Tensor:
        diff = z.unsqueeze(1) - mu.unsqueeze(0)
        quad = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)
        log_det = torch.log(var).sum(dim=1)
        log_prob = torch.log(phi).unsqueeze(0) - 0.5 * (
            quad
            + log_det.unsqueeze(0)
            + z.shape[1] * torch.log(z.new_tensor(2.0 * torch.pi))
        )
        return -torch.logsumexp(log_prob, dim=1)

    def loss(
        self,
        x: torch.Tensor,
        *,
        energy_weight: float = 0.1,
        covariance_weight: float = 0.005,
    ) -> torch.Tensor:
        out = self(x)
        phi, mu, var = self.gmm_params(out.latent, out.gamma)
        recon = F.mse_loss(out.reconstruction, x)
        energy = self.sample_energy(out.latent, phi, mu, var).mean()
        cov_penalty = (1.0 / var).mean()
        return (
            recon
            + float(energy_weight) * energy
            + float(covariance_weight) * cov_penalty
        )

    def fit_density(self, x: torch.Tensor, batch_size: int = 512) -> None:
        z_values: list[torch.Tensor] = []
        gamma_values: list[torch.Tensor] = []
        was_training = self.training
        self.eval()
        with torch.no_grad():
            for start in range(0, x.shape[0], int(batch_size)):
                out = self(x[start : start + int(batch_size)])
                z_values.append(out.latent.detach())
                gamma_values.append(out.gamma.detach())
        z = torch.cat(z_values, dim=0)
        gamma = torch.cat(gamma_values, dim=0)
        phi, mu, var = self.gmm_params(z, gamma)
        self.register_buffer("density_phi_", phi.detach())
        self.register_buffer("density_mu_", mu.detach())
        self.register_buffer("density_var_", var.detach())
        self.train(was_training)

    def energy_score(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        phi = getattr(self, "density_phi_", None)
        mu = getattr(self, "density_mu_", None)
        var = getattr(self, "density_var_", None)
        if phi is None or mu is None or var is None:
            phi, mu, var = self.gmm_params(out.latent, out.gamma)
        return self.sample_energy(
            out.latent, phi.to(x.device), mu.to(x.device), var.to(x.device)
        )


__all__ = ["DAGMM", "DAGMMForward"]
