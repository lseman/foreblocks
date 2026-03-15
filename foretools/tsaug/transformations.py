"""
Time series augmentation transformations for AutoDA-Timeseries.

Implements transformation set T:
  Raw, Jittering, Scaling, Resample, TimeWarp, FreqWarp, MagWarp, TimeMask, Drift

Each transformation takes:
  - x: (batch, length, channels) tensor
  - intensity: scalar or (batch,) tensor controlling augmentation strength
and returns the augmented tensor of the same shape.
"""

import torch
import torch.nn.functional as F
import numpy as np


def _to_batch_intensity(intensity: torch.Tensor, batch_size: int, device, dtype):
    """Normalize intensity input to shape (B,) on the target device/dtype."""
    if not torch.is_tensor(intensity):
        intensity = torch.tensor(float(intensity), device=device, dtype=dtype)
    if intensity.dim() == 0:
        intensity = intensity.expand(batch_size)
    return intensity.to(device=device, dtype=dtype)


def raw(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Identity transformation — returns input unchanged."""
    return x


def jittering(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise scaled by intensity.

    Y(c) = c + n, where n ~ N(0, intensity^2)
    """
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    std = intensity.abs().view(-1, 1, 1)  # (B, 1, 1)
    noise = torch.randn_like(x) * std
    return x + noise


def scaling(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Multiply by a random scaling factor centered at 1.

    Y(c) = c * s, where s ~ U[1 - intensity, 1 + intensity]
    """
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    half_range = intensity.abs().clamp(max=0.99).view(-1, 1, 1)
    # Sample uniform scale per batch element
    u = torch.rand(x.size(0), 1, 1, device=x.device)
    scale = 1.0 - half_range + 2.0 * half_range * u
    return x * scale


def resample(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Resample the time series by interpolating to a randomly shorter/longer
    length and then back to the original length.
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)

    # Per-sample ratio (instead of previous batch-averaged ratio).
    ratio = 1.0 + intensity.abs().clamp(max=0.5) * (
        2.0 * torch.rand(B, device=x.device, dtype=x.dtype) - 1.0
    )
    ratio = ratio.clamp(min=0.6, max=1.4)

    out = torch.empty_like(x)
    x_t = x.permute(0, 2, 1)  # (B, C, L)
    for b in range(B):
        new_l = max(2, int(L * ratio[b].item()))
        x_res = F.interpolate(
            x_t[b : b + 1], size=new_l, mode="linear", align_corners=False
        )
        x_back = F.interpolate(x_res, size=L, mode="linear", align_corners=False)
        out[b] = x_back.squeeze(0).transpose(0, 1)
    return out


def time_warp(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Warp the time axis using a smooth random curve.

    Generates a monotonically increasing warping path via cumulative sum
    of perturbed uniform steps.
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    mag = intensity.abs().clamp(max=2.0).view(-1, 1)

    # Create smooth warp path: start with uniform steps, add perturbation
    steps = torch.ones(B, L, device=x.device)
    # Add smooth noise via a few low-frequency sinusoids
    num_knots = 4
    t = torch.linspace(0, 1, L, device=x.device).unsqueeze(0).expand(B, -1)
    for _ in range(num_knots):
        freq = torch.rand(B, 1, device=x.device) * 3.0 + 1.0
        phase = torch.rand(B, 1, device=x.device) * 2 * np.pi
        steps = steps + mag * 0.1 * torch.sin(freq * t * 2 * np.pi + phase)

    steps = steps.clamp(min=0.1)
    warp_path = torch.cumsum(steps, dim=1)
    # Normalize to [0, L-1]
    warp_path = (warp_path - warp_path[:, :1]) / (
        warp_path[:, -1:] - warp_path[:, :1] + 1e-8
    ) * (L - 1)

    # Interpolate using the warped indices
    # x: (B, L, C) -> gather along time dim
    warp_path = warp_path.unsqueeze(-1).expand(-1, -1, C)
    idx_floor = warp_path.long().clamp(0, L - 2)
    idx_ceil = (idx_floor + 1).clamp(max=L - 1)
    frac = warp_path - idx_floor.float()

    x_floor = torch.gather(x, 1, idx_floor)
    x_ceil = torch.gather(x, 1, idx_ceil)
    return x_floor + frac * (x_ceil - x_floor)


def freq_warp(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Frequency-domain warping via perturbation in the Fourier domain.

    Applies random phase shifts scaled by intensity to the frequency
    components, preserving magnitude.
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    mag = intensity.abs().clamp(max=1.0).view(-1, 1, 1)

    # FFT along time axis
    x_t = x.permute(0, 2, 1)  # (B, C, L)
    X_freq = torch.fft.rfft(x_t, dim=-1)

    # Random phase perturbation
    n_freq = X_freq.shape[-1]
    phase_noise = (
        torch.randn(B, C, n_freq, device=x.device, dtype=x.dtype) * mag * np.pi * 0.1
    )
    perturbation = torch.exp(1j * phase_noise)
    X_freq_warped = X_freq * perturbation

    # IFFT back
    x_warped = torch.fft.irfft(X_freq_warped, n=L, dim=-1)
    return x_warped.permute(0, 2, 1)


def mag_warp(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Magnitude warping — multiply by a smooth random curve along time.

    Generates a smooth curve via cubic spline interpolation of random knots.
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    mag = intensity.abs().clamp(max=1.0).view(-1, 1)

    # Generate smooth warping curve from random knots
    num_knots = 4
    knot_values = 1.0 + mag * (torch.rand(B, num_knots, device=x.device) * 2 - 1) * 0.5
    # Add boundary values
    knot_values = torch.cat(
        [knot_values[:, :1], knot_values, knot_values[:, -1:]], dim=1
    )
    # Interpolate to length L
    curve = F.interpolate(
        knot_values.unsqueeze(1), size=L, mode="linear", align_corners=False
    ).squeeze(1)  # (B, L)

    return x * curve.unsqueeze(-1)


def time_mask(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Mask a contiguous time window and fill it with local mean."""
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    frac = intensity.abs().clamp(max=1.0) * 0.35  # up to 35% masking
    out = x.clone()

    for b in range(B):
        win = int(max(1, round(float(frac[b].item()) * L)))
        if win >= L:
            win = L - 1
        if win <= 0:
            continue
        start = torch.randint(0, L - win + 1, (1,), device=x.device).item()
        end = start + win
        fill = x[b].mean(dim=0, keepdim=True)  # (1, C)
        out[b, start:end, :] = fill
    return out


def drift(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Add smooth low-frequency drift (trend) scaled by signal std."""
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)

    t = torch.linspace(-1.0, 1.0, L, device=x.device, dtype=x.dtype).view(1, L, 1)
    lin = t
    quad = t * t - t.mean()
    coeff_lin = (
        torch.randn(B, 1, 1, device=x.device, dtype=x.dtype)
        * intensity.view(B, 1, 1)
        * 0.3
    )
    coeff_quad = (
        torch.randn(B, 1, 1, device=x.device, dtype=x.dtype)
        * intensity.view(B, 1, 1)
        * 0.15
    )
    base = coeff_lin * lin + coeff_quad * quad
    scale = x.std(dim=1, keepdim=True).clamp(min=1e-6)
    return x + base * scale


# Registry mapping indices to transformation functions
TRANSFORMATIONS = [
    raw,         # T1: Raw (index 0)
    jittering,   # T2: Jittering
    scaling,     # T3: Scaling
    resample,    # T4: Resample
    time_warp,   # T5: TimeWarp
    freq_warp,   # T6: FreqWarp
    mag_warp,    # T7: MagWarp
    time_mask,   # T8: TimeMask
    drift,       # T9: Drift
]

TRANSFORM_NAMES = [
    "Raw", "Jittering", "Scaling", "Resample",
    "TimeWarp", "FreqWarp", "MagWarp", "TimeMask", "Drift",
]

NUM_TRANSFORMS = len(TRANSFORMATIONS)
