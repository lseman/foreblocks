import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from foreblocks.transformer.norms.triton_backend import (
        TRITON_AVAILABLE,
        RMSNormTritonFunction,
        _should_use_triton,
    )
except Exception:  # pragma: no cover - foreblocks namespace may exclude transformer
    TRITON_AVAILABLE = False
    RMSNormTritonFunction = None

    def _should_use_triton(x, min_numel: int = 2048) -> bool:
        return False

from .norms import RMSNorm


class DLinearOp(nn.Module):
    """DLinear-style trend/seasonal decomposition followed by linear projection."""

    def __init__(self, input_dim: int, latent_dim: int, trend_kernel: int = 25):
        super().__init__()
        self.trend_kernel = max(3, int(trend_kernel) | 1)  # enforce odd kernel

        self.seasonal_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.trend_proj = nn.Linear(input_dim, latent_dim, bias=False)
        self.output_norm = RMSNorm(latent_dim)

        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        x_t = x.transpose(1, 2)

        trend = F.avg_pool1d(
            x_t,
            kernel_size=self.trend_kernel,
            stride=1,
            padding=self.trend_kernel // 2,
        ).transpose(1, 2)
        seasonal = x - trend

        y = self.seasonal_proj(seasonal) + self.trend_proj(trend)
        return self.output_norm(y + residual)


class NBeatsOp(nn.Module):
    """N-BEATS-inspired doubly-residual block for time-series backbone cells.

    Each block learns to split the input into a *backcast* (the portion that
    can be explained and should be removed) and a *forecast contribution*
    (what is added forward).  The cell output is

        out = norm(x_proj − backcast + forecast)

    which is the doubly-residual structure of Oreshkin et al. (ICLR 2020)
    adapted for use as a backbone operation rather than as a full stack.
    """

    def __init__(self, input_dim: int, latent_dim: int, expansion: int = 4):
        super().__init__()
        hidden = latent_dim * expansion
        # Shared 3-layer FC stack (generic basis, no inductive bias)
        self.fc_stack = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.backcast_proj = nn.Linear(hidden, latent_dim, bias=False)
        self.forecast_proj = nn.Linear(hidden, latent_dim, bias=False)
        self.input_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )
        self.norm = RMSNorm(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D_in]
        h = self.fc_stack(x)  # [B, L, hidden]
        backcast = self.backcast_proj(h)  # [B, L, D_out]
        forecast = self.forecast_proj(h)  # [B, L, D_out]
        residual = self.input_proj(x)  # [B, L, D_out]
        return self.norm(residual - backcast + forecast)


class TimesNetOp(nn.Module):
    """TimesNet: 2-D temporal variation modelling via FFT period detection.

    Treats 1-D time-series of length T as a 2-D signal by detecting the
    top-k dominant periods through FFT, reshaping each period window into a
    2-D tensor, applying a 2-D convolutional block, and averaging the
    period-wise outputs back to [B, T, D].

    Reference: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for
    General Time Series Analysis", ICLR 2023.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        top_k: int = 3,
        num_kernels: int = 6,
    ):
        super().__init__()
        self.top_k = max(1, top_k)
        self.input_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )
        # Inception-style 2-D convolutional block
        self.conv2d = nn.Sequential(
            nn.Conv2d(latent_dim, num_kernels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(num_kernels, latent_dim, kernel_size=3, padding=1),
        )
        self.norm = RMSNorm(latent_dim)
        self.residual_proj = (
            nn.Linear(input_dim, latent_dim, bias=False)
            if input_dim != latent_dim
            else nn.Identity()
        )

    @staticmethod
    def _top_k_periods(x: torch.Tensor, k: int) -> list[int]:
        """Return the top-k dominant periods inferred from FFT amplitudes."""
        T = x.size(1)
        # Average over B and C so the period is a dataset-level statistic
        xf = torch.fft.rfft(x.detach().mean(dim=(0, 2)), norm="ortho")  # [T//2+1]
        amplitudes = xf.abs()
        amplitudes[0] = 0.0  # suppress DC component
        n_freqs = amplitudes.shape[0]
        topk = torch.topk(amplitudes[1:n_freqs], min(k, n_freqs - 1))
        # freq index → period length (floor); ensure period >= 2
        periods = [max(2, T // int(idx.item() + 1)) for idx in topk.indices]
        return periods

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        B, T, _ = x.shape
        residual = self.residual_proj(x)  # [B, T, D]
        x_proj = self.input_proj(x)  # [B, T, D]
        D = x_proj.size(-1)

        # Detect dominant periods without accumulating graph
        with torch.no_grad():
            periods = self._top_k_periods(x_proj, self.top_k)

        period_outputs: list[torch.Tensor] = []
        for period in periods:
            # Pad T to the next multiple of `period`
            pad_len = (period - T % period) % period
            x_pad = F.pad(x_proj, (0, 0, 0, pad_len))  # [B, T+pad, D]
            T_pad = x_pad.shape[1]
            rows = T_pad // period
            # Reshape to 2-D spatial representation: [B, D, rows, period]
            x_2d = x_pad.reshape(B, rows, period, D).permute(0, 3, 1, 2)
            y_2d = self.conv2d(x_2d)  # [B, D, rows, period]
            y_1d = y_2d.permute(0, 2, 3, 1).reshape(B, T_pad, D)
            period_outputs.append(y_1d[:, :T, :])  # trim padding

        if period_outputs:
            out = torch.stack(period_outputs, dim=0).mean(dim=0)  # [B, T, D]
        else:
            out = x_proj

        return self.norm(out + residual)


