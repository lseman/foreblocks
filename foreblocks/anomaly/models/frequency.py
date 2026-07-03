"""foreblocks.anomaly.models.frequency.

Frequency-domain anomaly detection for time series windows.

Detects anomalies that manifest in spectral space rather than time domain by learning
mappings between time and frequency representations. Scores combine time-domain
reconstruction error, frequency-domain spectral error, and deviation from normal
spectral profiles. Use when anomalies have distinctive frequency signatures (e.g.
sudden spectral shifts, harmonic distortions) invisible in raw time series.

Core API:
- FrequencyAnomaly: learns time↔frequency mapping with combined spectral scoring
- LogFreqAnomaly: log-frequency variant robust to scale differences

"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FrequencyAnomalyForward:
    time_error: torch.Tensor
    freq_error: torch.Tensor
    spectral_ratio: torch.Tensor
    reconstruction: torch.Tensor


class FourierEncoder(nn.Module):
    """Learned frequency encoder using complex-valued projections."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        d_model = int(d_model)
        self.d_model = d_model

        self.input_proj = nn.Linear(self.n_features, d_model)
        self.freq_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(d_model // 2, 16)),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(max(d_model // 2, 16), d_model),
        )
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, d_model // 16),
                dim_feedforward=max(d_model * 4, 64),
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        )
        self.output_proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, feat = x.shape
        h = self.input_proj(x)
        # Add frequency-domain bias
        mag = torch.abs(fft.rfft(x, dim=1))
        phase = torch.angle(fft.rfft(x, dim=1))
        freq_enc = torch.cat([mag, phase], dim=-1)
        freq_enc = self.freq_proj(freq_enc[:, :seq, :])
        h = h + freq_enc
        for layer in self.layers:
            h = layer(h)
        return self.output_proj(h)


class InverseTransformer(nn.Module):
    """Inverse: learned frequency → time reconstruction."""

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.d_model = int(d_model)

        self.freq_proj = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model, self.d_model),
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.n_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(x)
        phase = torch.angle(x)
        combined = self.freq_proj(torch.cat([mag, phase], dim=-1))
        return self.output_proj(combined)


class FrequencyAnomaly(nn.Module):
    """Frequency-domain anomaly detector.

    Learns mapping between time and frequency domains.
    Anomaly score combines:
    - Time-domain reconstruction error
    - Frequency-domain spectral error
    - Spectral ratio (deviation from normal frequency profile)
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        use_wavelet: bool = False,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)
        self.use_wavelet = bool(use_wavelet)

        self.freq_encoder = FourierEncoder(
            n_features=n_features,
            window_size=window_size,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        self.inverse = InverseTransformer(
            n_features=n_features,
            window_size=window_size,
            d_model=d_model,
            dropout=dropout,
        )

        # Normal spectral profile (running statistics)
        self.register_buffer("normal_spectral_mean_", None)
        self.register_buffer("normal_spectral_std_", None)
        self.spectral_momentum = 0.99
        self.num_spectral_updates = 0

    def time_to_freq(self, x: torch.Tensor) -> torch.Tensor:
        return fft.rfft(x, dim=1)

    def freq_to_time(self, X: torch.Tensor) -> torch.Tensor:
        return fft.irfft(X, dim=1, n=self.window_size)

    def _wavelet_transform(self, x: torch.Tensor) -> torch.Tensor:
        """Simple Haar-like wavelet approximation via downsampling."""
        low = (x[:, ::2] + x[:, 1::2]) / 2.0
        high = (x[:, ::2] - x[:, 1::2]) / 2.0
        return torch.cat([low, high], dim=-1)

    def forward(self, x: torch.Tensor) -> FrequencyAnomalyForward:
        # Time → freq → encode
        X = self.time_to_freq(x)
        encoded = self.freq_encoder(X)
        # Reconstruct in freq domain
        X_recon = self.inverse(encoded)
        # Back to time
        x_recon = self.freq_to_time(X_recon)

        # Time-domain error
        time_error = F.mse_loss(x_recon, x, reduction="none").mean(dim=(1, 2))

        # Frequency-domain error
        freq_error = F.mse_loss(X_recon, X, reduction="none").mean(dim=(1, 2))

        # Update running spectral statistics
        spectral_mag = torch.abs(X).mean(dim=0)  # [seq, freq]
        self._update_spectral_stats(spectral_mag)
        spectral_ratio = self._spectral_deviation(x)

        return FrequencyAnomalyForward(
            time_error,
            freq_error,
            spectral_ratio,
            x_recon,
        )

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        # Combined score: weighted average of time + freq errors + spectral
        return 0.5 * out.time_error + 0.3 * out.freq_error + 0.2 * out.spectral_ratio

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        """Train: minimize time + freq reconstruction error."""
        out = self(x)
        return out.time_error.mean() + 0.5 * out.freq_error.mean()

    def _update_spectral_stats(self, spectral_mag: torch.Tensor) -> None:
        if self.normal_spectral_mean_ is None:
            self.normal_spectral_mean_ = spectral_mag
            self.normal_spectral_std_ = torch.ones_like(spectral_mag) * 0.5
            self.num_spectral_updates = 1
            return

        with torch.no_grad():
            self.num_spectral_updates += 1
            self.normal_spectral_mean_ = (
                self.spectral_momentum * self.normal_spectral_mean_
                + (1 - self.spectral_momentum) * spectral_mag
            )
            self.normal_spectral_std_ = (
                self.spectral_momentum * self.normal_spectral_std_
                + (1 - self.spectral_momentum)
                * (
                    (spectral_mag - self.normal_spectral_mean_).pow(2)
                    + self.normal_spectral_std_
                ).sqrt()
            )

    def _spectral_deviation(self, x: torch.Tensor) -> torch.Tensor:
        """Maha-like distance of spectral profile from normal."""
        if self.normal_spectral_mean_ is None:
            return torch.zeros(x.shape[0], device=x.device)

        X = self.time_to_freq(x)
        mag = torch.abs(X).mean(dim=0)
        diff = mag - self.normal_spectral_mean_
        std = self.normal_spectral_std_ + 1e-8
        dev = (diff / std).pow(2).mean(dim=1)  # [batch]
        return dev

    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)


class LogFreqAnomaly(nn.Module):
    """Log-frequency variant: focuses on log-scaled spectral features.

    More robust to scale differences. Inspired by LogFT paper.
    Uses log-frequency spectrogram for better resolution at low frequencies.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.window_size = int(window_size)

        # Log-frequency grid
        n_log_bins = max(int(n_features * 0.5), 16)
        log_freqs = self._log_freq_grid(window_size, n_log_bins)
        self.register_buffer("log_freqs", log_freqs)

        self.encoder = nn.Sequential(
            nn.Linear(int(n_log_bins), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, int(n_log_bins)),
        )

    @staticmethod
    def _log_freq_grid(window_size: int, n_bins: int) -> torch.Tensor:
        """Create log-spaced frequency-to-bin interpolation weights."""
        fft_bins = window_size // 2 + 1
        freqs = torch.linspace(0.0, 1.0, fft_bins)
        log_freqs = torch.log1p(10.0 * freqs) / torch.log1p(torch.tensor(10.0))
        log_grid = torch.linspace(0.0, 1.0, n_bins)
        lo = (log_grid * (fft_bins - 1)).long().clamp(0, fft_bins - 2)
        hi = lo + 1
        f = (log_grid * (fft_bins - 1) - lo.float()).unsqueeze(1)
        return (1 - f) * freqs[lo].unsqueeze(0) + f * freqs[hi].unsqueeze(0)

    def forward(self, x: torch.Tensor) -> FrequencyAnomalyForward:
        X = self.time_to_freq(x)
        mag = torch.abs(X) + 1e-8
        log_mag = torch.log(mag)

        # Project to log-frequency grid: [batch, seq, fft_bins] → [batch, seq, n_bins]
        bsz, seq, _ = log_mag.shape
        weights: torch.Tensor = self.log_freqs  # type: ignore[assignment]
        log_freq_proj = log_mag.view(bsz * seq, -1) @ weights.T
        log_freq_proj = log_freq_proj.view(bsz, seq, -1)

        encoded = self.encoder(log_freq_proj)
        recon = self.decoder(encoded)
        log_mag_recon = recon

        # Log-domain error
        log_error = F.mse_loss(log_mag_recon, log_freq_proj, reduction="none").mean(
            dim=(1, 2)
        )

        # Also reconstruct in time domain
        log_mag_back = log_mag_recon.exp()
        X_recon = log_mag_back * torch.exp(1j * torch.angle(X))
        x_recon = fft.irfft(X_recon, dim=1, n=self.window_size)

        time_error = F.mse_loss(x_recon, x, reduction="none").mean(dim=(1, 2))
        spectral_ratio = log_error

        return FrequencyAnomalyForward(
            time_error,
            log_error,
            spectral_ratio,
            x_recon,
        )

    @staticmethod
    def time_to_freq(x: torch.Tensor) -> torch.Tensor:
        return fft.rfft(x, dim=1)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return 0.6 * out.time_error + 0.4 * out.freq_error

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return out.time_error.mean() + 0.3 * out.freq_error.mean()

    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)
