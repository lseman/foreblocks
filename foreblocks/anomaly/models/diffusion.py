"""Diffusion-based anomaly detection for time series windows.

Score-based diffusion (DDPM-style) trained on normal windows.
At inference, high reconstruction error = anomalous.
Supports classifier-free guidance and test-time adaptation.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F



@dataclass
class DiffusionAnomalyForward:
    reconstruction: torch.Tensor
    recon_error: torch.Tensor
    noise_level: float


class _SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, dtype=t.dtype, device=t.device)) / (
            half_dim - 1
        )
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DiffusionNetwork(nn.Module):
    """Time-conditional MLP for diffusion denoising."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.input_dim = self.n_features * self.window_size

        self.time_mlp = nn.Sequential(
            _SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.layers = nn.ModuleList(
            _ResidualBlock(d_model, dropout) for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(d_model, self.input_dim)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        time_emb = self.time_mlp(t)
        h = self.input_proj(x.reshape(x.shape[0], -1)) + time_emb
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h).reshape_as(x)


class DiffusionScheduler:
    """Linear noise schedule (DDPM-style)."""

    def __init__(self, num_steps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02) -> None:
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        alpha = self.alpha_cumprod[t].view(-1, 1, 1)
        return (
            torch.sqrt(alpha) * x + torch.sqrt(1 - alpha) * noise,
            noise,
        )

    @torch.no_grad()
    def denoise(
        self,
        model: DiffusionNetwork,
        noisy: torch.Tensor,
        *,
        num_steps: int | None = None,
        steps: torch.Tensor | None = None,
        guidance: float = 1.0,
    ) -> torch.Tensor:
        if num_steps is None:
            num_steps = self.num_steps
        if steps is None:
            steps = torch.arange(num_steps - 1, -1, device=noisy.device, dtype=torch.long)

        x = noisy
        for i in steps:
            t = torch.full((noisy.shape[0],), i.item(), device=noisy.device, dtype=torch.long)
            predicted = model(x, t)

            if guidance > 1.0:
                # Classifier-free guidance: unconditional prediction
                x_null = torch.zeros_like(noisy)
                t_null = torch.zeros_like(t)
                predicted_null = model(x_null, t_null)
                predicted = predicted_null + guidance * (predicted - predicted_null)

            alpha = self.alpha_cumprod[t]
            alpha_prev = torch.cat(
                [torch.tensor([1.0], device=noisy.device), self.alpha_cumprod[:num_steps - 1]]
            )[t]
            beta = self.betas[t]

            coeff1 = (1 - alpha_prev) / torch.sqrt(1 - alpha + 1e-8)
            coeff2 = torch.sqrt(alpha_prev) * beta / torch.sqrt(1 - alpha + 1e-8)
            coeff3 = torch.sqrt(alpha_prev) / torch.sqrt(1 - alpha + 1e-8)

            x = (
                torch.sqrt(alpha_prev) * predicted
                + coeff3 * (x - coeff1 * predicted - coeff2 * predicted)
            )
        return x


class DiffusionAnomaly(nn.Module):
    """Diffusion-based anomaly detector.

    Trains a diffusion model to reconstruct normal windows.
    Anomaly score = mean-squared reconstruction error after denoising.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        n_layers: int = 4,
        num_diffusion_steps: int = 1000,
        guidance: float = 1.5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.guidance = guidance
        self.model = DiffusionNetwork(
            n_features=n_features,
            window_size=window_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.scheduler = DiffusionScheduler(num_steps=num_diffusion_steps)

    def forward(
        self, x: torch.Tensor, *, noise: torch.Tensor | None = None
    ) -> DiffusionAnomalyForward:
        """Forward with optional pre-noised input (for testing)."""
        if noise is None:
            noisy, _ = self.scheduler.noise(x, torch.full(
                (x.shape[0],),
                self.scheduler.num_steps - 1,
                device=x.device,
                dtype=torch.long,
            ))
        else:
            noisy = noise

        recon = self.scheduler.denoise(
            self.model, noisy, guidance=self.guidance
        )
        recon_error = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
        noise_level = float(noise.std()) if noise is not None else 1.0
        return DiffusionAnomalyForward(recon, recon_error, noise_level)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).recon_error

    def train_step(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """Standard diffusion training: predict noise."""
        t = torch.randint(
            0, self.scheduler.num_steps, (x.shape[0],), device=x.device
        )
        noisy, target_noise = self.scheduler.noise(x, t)
        predicted = self.model(noisy, t)
        return F.mse_loss(predicted, target_noise)

    @torch.no_grad()
    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        """Full denoising for inference scoring."""
        t_max = torch.full((x.shape[0],), self.scheduler.num_steps - 1, device=x.device)
        noisy, _ = self.scheduler.noise(x, t_max)
        recon = self.scheduler.denoise(self.model, noisy, guidance=self.guidance)
        return F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
