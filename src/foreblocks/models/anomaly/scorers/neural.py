"""Native PyTorch fitting for denoising AE, deterministic-score VAE and Deep SVDD."""

import numpy as np
import torch
from torch import nn

from foreblocks.models.anomaly.backbones.tabular import (
    SVDDEncoder,
    TabularAutoEncoder,
    TabularVAE,
)
from foreblocks.models.anomaly.scorers.base import (
    FittedScorer,
    positive_float,
    positive_int,
)


class _NeuralScorer(FittedScorer):
    kind = "autoencoder"

    def __init__(
        self,
        hidden_sizes=(64, 32),
        latent_dim=8,
        epochs=20,
        batch_size=128,
        learning_rate=1e-3,
        weight_decay=1e-5,
        noise_std=0.05,
        beta=0.01,
        objective="one_class",
        nu=0.1,
        warmup_epochs=5,
        device=None,
        *,
        standardize=True,
        seed=42,
    ):
        super().__init__(standardize=standardize, seed=seed)
        self.hidden_sizes = tuple(
            positive_int("hidden_sizes entry", n) for n in hidden_sizes
        )
        if not self.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one layer.")
        self.latent_dim = positive_int("latent_dim", latent_dim)
        self.epochs = positive_int("epochs", epochs)
        self.batch_size = positive_int("batch_size", batch_size)
        self.learning_rate = positive_float("learning_rate", learning_rate)
        self.weight_decay = positive_float(
            "weight_decay", weight_decay, allow_zero=True
        )
        self.noise_std = positive_float("noise_std", noise_std, allow_zero=True)
        self.beta = positive_float("beta", beta, allow_zero=True)
        if objective not in {"one_class", "soft_boundary"}:
            raise ValueError("objective must be 'one_class' or 'soft_boundary'.")
        if not np.isfinite(nu) or not 0 < nu <= 1:
            raise ValueError("nu must be in (0, 1].")
        self.objective, self.nu = objective, float(nu)
        self.warmup_epochs = positive_int("warmup_epochs", warmup_epochs, 0)
        self.device = device

    def _tensor(self, x):
        data = torch.as_tensor(x, dtype=torch.float32, device=self.device_)
        if not torch.isfinite(data).all():
            raise ValueError("Input magnitude exceeds float32 range.")
        return data

    def _fit(self, x):
        self.device_ = torch.device(
            self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Restore the caller's RNG state, including CUDA states touched by manual_seed.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            if self.seed is not None:
                torch.manual_seed(self.seed)
            self._train(x)

    def _train(self, x):
        architecture = {
            "autoencoder": TabularAutoEncoder,
            "vae": TabularVAE,
            "deep_svdd": SVDDEncoder,
        }[self.kind]
        self.model_ = architecture(x.shape[1], self.hidden_sizes, self.latent_dim).to(
            self.device_
        )
        # Keep the full training set in host memory; move only minibatches to device.
        data = torch.from_numpy(x.astype(np.float32))
        if not torch.isfinite(data).all():
            raise ValueError("Input magnitude exceeds float32 range.")
        if self.kind == "deep_svdd":
            self.model_.eval()
            with torch.no_grad():
                total = torch.zeros(self.latent_dim, device=self.device_)
                for batch in data.split(self.batch_size):
                    total += self.model_(batch.to(self.device_)).sum(dim=0)
                center = total / len(data)
                # Nonzero fixed center and bias-free network prevent the trivial
                # zero map from being an exact optimum (Ruff et al., 2018).
                center = torch.where(
                    center.abs() < 0.1, torch.where(center < 0, -0.1, 0.1), center
                )
                self.model_.center.copy_(center)
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        self.loss_history_ = []
        for epoch in range(self.epochs):
            self.model_.train()
            total_loss = 0.0
            order = torch.randperm(len(data))
            for indices in order.split(self.batch_size):
                batch = data[indices].to(self.device_)
                if self.kind == "autoencoder":
                    reconstruction = self.model_(
                        batch + self.noise_std * torch.randn_like(batch)
                    )
                    loss = (reconstruction - batch).square().mean()
                elif self.kind == "vae":
                    reconstruction, mu, logvar = self.model_(batch)
                    reconstruction_loss = (
                        (reconstruction - batch).square().sum(dim=1).mean()
                    )
                    kl = (
                        -0.5
                        * (1 + logvar - mu.square() - logvar.exp()).sum(dim=1).mean()
                    )
                    loss = reconstruction_loss + self.beta * kl
                else:
                    distance = (
                        (self.model_(batch) - self.model_.center).square().sum(dim=1)
                    )
                    if (
                        self.objective == "soft_boundary"
                        and epoch >= self.warmup_epochs
                    ):
                        radius_squared = self.model_.radius.square()
                        loss = (
                            radius_squared
                            + torch.relu(distance - radius_squared).mean() / self.nu
                        )
                    else:
                        loss = distance.mean()
                if not torch.isfinite(loss):
                    raise ValueError("Neural training produced a nonfinite loss.")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 10.0)
                optimizer.step()
                total_loss += loss.item() * len(batch)
            self.loss_history_.append(total_loss / len(data))
            if (
                self.kind == "deep_svdd"
                and self.objective == "soft_boundary"
                and epoch >= self.warmup_epochs
            ):
                self.model_.eval()
                with torch.no_grad():
                    distances = torch.cat(
                        [
                            (self.model_(batch.to(self.device_)) - self.model_.center)
                            .square()
                            .sum(dim=1)
                            .cpu()
                            for batch in data.split(self.batch_size)
                        ]
                    )
                    self.model_.radius.copy_(
                        torch.quantile(distances.sqrt(), 1 - self.nu).to(self.device_)
                    )
        self.model_.eval()

    def _score(self, x):
        self.model_.eval()
        scores = []
        with torch.no_grad():
            for start in range(0, len(x), self.batch_size):
                batch = self._tensor(x[start : start + self.batch_size])
                output = self.model_(batch)
                if self.kind == "deep_svdd":
                    score = (output - self.model_.center).square().sum(dim=1)
                    if self.objective == "soft_boundary":
                        score -= self.model_.radius.square()
                else:
                    reconstruction = output[0] if self.kind == "vae" else output
                    score = (reconstruction - batch).square().mean(dim=1)
                scores.append(score.cpu().numpy())
        return np.concatenate(scores)


class AutoEncoderScorer(_NeuralScorer):
    """Denoising bottleneck MLP, scored by clean reconstruction mean squared error."""

    kind = "autoencoder"


class VAEScorer(_NeuralScorer):
    """Gaussian latent VAE; deterministic posterior-mean reconstruction at inference."""

    kind = "vae"


class DeepSVDD(_NeuralScorer):
    """Deep one-class or soft-boundary hypersphere objective (Ruff et al., 2018)."""

    kind = "deep_svdd"
