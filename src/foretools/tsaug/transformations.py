"""Time series augmentation transformations for AutoDA-Timeseries.

Implements 12 transformation functions forming the set T = {T_1, ..., T_12}:

| Index | Name         | Description                                    |
|-------|--------------|------------------------------------------------|
| T1    | Raw          | Identity — returns input unchanged             |
| T2    | Jittering    | Add Gaussian noise scaled by intensity         |
| T3    | Scaling      | Multiply by random scaling factor              |
| T4    | Resample     | Interpolate to shorter/longer length           |
| T5    | TimeWarp     | Warp time axis using smooth random curve       |
| T6    | FreqWarp     | Perturb phase in Fourier domain                |
| T7    | MagWarp      | Multiply by smooth random curve along time     |
| T8    | TimeMask     | Mask contiguous window, fill with local mean   |
| T9    | Drift        | Add smooth low-frequency trend                 |
| T10   | Permutation  | Randomly permute temporal segments             |
| T11   | WindowSlice  | Crop a window and resize back to original      |
| T12   | TimeMix      | Mix segments between paired samples            |

Each transformation has the signature::

    def transform_fn(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor

where:
  - ``x``: (batch_size, length, channels) tensor
  - ``intensity``: scalar or (batch_size,) tensor in [0, 1] controlling strength
  - Returns: augmented tensor of shape (batch_size, length, channels)

Usage
-----
>>> import torch
>>> from foretools.tsaug.transformations import jittering, scaling
>>> x = torch.randn(4, 100, 1)  # (B, L, C)
>>> intensity = torch.tensor([0.5])  # scalar broadcast to batch
>>> x_jittered = jittering(x, intensity)
>>> x_scaled = scaling(x, intensity)
"""

import numpy as np
import torch
import torch.nn.functional as F


def _to_batch_intensity(intensity: torch.Tensor, batch_size: int, device, dtype):
    """Normalize intensity input to shape (B,) on the target device/dtype."""
    if not torch.is_tensor(intensity):
        intensity = torch.tensor(float(intensity), device=device, dtype=dtype)
    if intensity.dim() == 0:
        intensity = intensity.expand(batch_size)
    return intensity.to(device=device, dtype=dtype)


def raw(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Identity transformation — returns input unchanged.

    Used as a baseline; the framework learns to assign probability mass
    to Raw when augmentation is not beneficial for a given sample.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Ignored by this transformation.

    Returns
    -------
    torch.Tensor
        Copy of *x* unchanged.
    """
    return x


def jittering(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Add Gaussian noise scaled by intensity.

    Each element is perturbed as::

        Y(c) = c + n,  where  n ~ N(0, intensity^2)

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Noise standard deviation. Scalar or (batch_size,) tensor.

    Returns
    -------
    torch.Tensor
        Noisy tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import jittering
    >>> x = torch.randn(2, 50, 1)
    >>> y = jittering(x, intensity=torch.tensor([0.1]))
    >>> assert y.shape == x.shape
    """
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    std = intensity.abs().view(-1, 1, 1)  # (B, 1, 1)
    noise = torch.randn_like(x) * std
    return x + noise
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    std = intensity.abs().view(-1, 1, 1)  # (B, 1, 1)
    noise = torch.randn_like(x) * std
    return x + noise


def scaling(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Multiply by a random scaling factor centered at 1.

    Each sample is scaled as::

        Y(c) = c * s,  where  s ~ U[1 - intensity, 1 + intensity]

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Scaling range half-width. Scalar or (batch_size,) tensor.
            Values > 0.99 are clamped to avoid excessive scaling.

    Returns
    -------
    torch.Tensor
        Scaled tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import scaling
    >>> x = torch.randn(2, 50, 1)
    >>> y = scaling(x, intensity=torch.tensor([0.2]))
    >>> assert y.shape == x.shape
    """
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    half_range = intensity.abs().clamp(max=0.99).view(-1, 1, 1)
    # Sample uniform scale per batch element
    u = torch.rand(x.size(0), 1, 1, device=x.device)
    scale = 1.0 - half_range + 2.0 * half_range * u
    return x * scale
    intensity = _to_batch_intensity(intensity, x.size(0), x.device, x.dtype)
    half_range = intensity.abs().clamp(max=0.99).view(-1, 1, 1)
    # Sample uniform scale per batch element
    u = torch.rand(x.size(0), 1, 1, device=x.device)
    scale = 1.0 - half_range + 2.0 * half_range * u
    return x * scale


def resample(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Resample the time series by interpolating to a randomly shorter/longer
    length and then back to the original length.

    Each sample is resampled with ratio::

        r = 1 + |intensity| * (2*u - 1),  where u ~ U[0, 1]

    The ratio is clamped to [0.6, 1.4], then the series is interpolated
    to ``round(L * r)`` points and back to *L*.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Resampling range. Scalar or (batch_size,) tensor.

    Returns
    -------
    torch.Tensor
        Resampled tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import resample
    >>> x = torch.randn(2, 50, 1)
    >>> y = resample(x, intensity=torch.tensor([0.3]))
    >>> assert y.shape == x.shape
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

    Generates a monotonically increasing warping path by adding low-frequency
    sinusoidal perturbations to uniform steps, then normalizing to [0, L-1].
    The original series is interpolated along this warped path.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Warping magnitude. Scalar or (batch_size,) tensor. Values > 2.0 are clamped.

    Returns
    -------
    torch.Tensor
        Time-warped tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import time_warp
    >>> x = torch.randn(2, 50, 1)
    >>> y = time_warp(x, intensity=torch.tensor([0.5]))
    >>> assert y.shape == x.shape
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
    warp_path = (
        (warp_path - warp_path[:, :1])
        / (warp_path[:, -1:] - warp_path[:, :1] + 1e-8)
        * (L - 1)
    )

    # Interpolate using the warped indices
    # x: (B, L, C) -> gather along time dim
    warp_path = warp_path.unsqueeze(-1).expand(-1, -1, C)
    idx_floor = warp_path.long().clamp(0, L - 2)
    idx_ceil = (idx_floor + 1).clamp(max=L - 1)
    frac = warp_path - idx_floor.float()

    x_floor = torch.gather(x, 1, idx_floor)
    x_ceil = torch.gather(x, 1, idx_ceil)
    return x_floor + frac * (x_ceil - x_floor)
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
    warp_path = (
        (warp_path - warp_path[:, :1])
        / (warp_path[:, -1:] - warp_path[:, :1] + 1e-8)
        * (L - 1)
    )

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

    Applies random Gaussian phase shifts scaled by intensity to each
    frequency component, preserving the magnitude spectrum.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Phase perturbation magnitude. Scalar or (batch_size,) tensor. Values > 1.0 are clamped.

    Returns
    -------
    torch.Tensor
        Frequency-warped tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import freq_warp
    >>> x = torch.randn(2, 50, 1)
    >>> y = freq_warp(x, intensity=torch.tensor([0.3]))
    >>> assert y.shape == x.shape
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

    Generates a smooth multiplicative curve from 4 random knots (clamped to
    [1 - 0.5*|intensity|, 1 + 0.5*|intensity|]) and linearly interpolated
    to length *L*.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Curve deviation magnitude. Scalar or (batch_size,) tensor. Values > 1.0 are clamped.

    Returns
    -------
    torch.Tensor
        Magnitude-warped tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import mag_warp
    >>> x = torch.randn(2, 50, 1)
    >>> y = mag_warp(x, intensity=torch.tensor([0.5]))
    >>> assert y.shape == x.shape
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
    """Mask a contiguous time window and fill it with local mean.

    A random window of length ``round(intensity * 0.35 * L)`` is selected
    and replaced with the channel-wise mean of the sample.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Fraction of time to mask (0–1). Scalar or (batch_size,) tensor.
            Window size is clamped to at most 35% of *L*.

    Returns
    -------
    torch.Tensor
        Time-masked tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import time_mask
    >>> x = torch.randn(2, 50, 1)
    >>> y = time_mask(x, intensity=torch.tensor([0.5]))
    >>> assert y.shape == x.shape
    """
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
    """Add smooth low-frequency drift (trend) scaled by signal std.

    A quadratic trend ``a*t + b*(t^2 - mean(t))`` is generated with random
    coefficients proportional to *intensity*, then scaled by the per-sample
    standard deviation.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Drift magnitude. Scalar or (batch_size,) tensor.

    Returns
    -------
    torch.Tensor
        Drifted tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import drift
    >>> x = torch.randn(2, 50, 1)
    >>> y = drift(x, intensity=torch.tensor([0.3]))
    >>> assert y.shape == x.shape
    """
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


def permutation(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Randomly permute a small number of temporal segments.

    Splits the time series into 2–5 segments (determined by *intensity*)
    and applies a random permutation to the segment order.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Number of segments: ``clamp(2 + intensity * 3, 2, 5)`` rounded. Scalar or (batch_size,) tensor.

    Returns
    -------
    torch.Tensor
        Permutated tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import permutation
    >>> x = torch.randn(2, 50, 1)
    >>> y = permutation(x, intensity=torch.tensor([0.5]))
    >>> assert y.shape == x.shape
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()

    for b in range(B):
        n_segments = int(
            torch.clamp(2 + intensity[b] * 3.0, min=2.0, max=5.0).round().item()
        )
        # deterministic segment split on sorted random knots
        if n_segments <= 1:
            continue
        cut_points = torch.rand(n_segments - 1, device=x.device).sort().values
        cut_points = (cut_points * (L - 1)).long().unique()
        splits = [0] + cut_points.tolist() + [L]
        segments = [x[b, splits[i] : splits[i + 1], :] for i in range(len(splits) - 1)]
        perm = torch.randperm(len(segments), device=x.device)
        out[b] = torch.cat([segments[i] for i in perm], dim=0)

    return out
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()

    for b in range(B):
        n_segments = int(
            torch.clamp(2 + intensity[b] * 3.0, min=2.0, max=5.0).round().item()
        )
        # deterministic segment split on sorted random knots
        if n_segments <= 1:
            continue
        cut_points = torch.rand(n_segments - 1, device=x.device).sort().values
        cut_points = (cut_points * (L - 1)).long().unique()
        splits = [0] + cut_points.tolist() + [L]
        segments = [x[b, splits[i] : splits[i + 1], :] for i in range(len(splits) - 1)]
        perm = torch.randperm(len(segments), device=x.device)
        out[b] = torch.cat([segments[i] for i in perm], dim=0)

    return out


def window_slice(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Randomly crop a window and resize it back to original length.

    A contiguous window of length ``round(L * (1 - 0.5 * intensity))`` is
    randomly selected, then linearly interpolated back to length *L*.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels).
    intensity : torch.Tensor
        Crop fraction: ``intensity * 0.5`` determines the fraction removed.
            Scalar or (batch_size,) tensor. Values > 0.8 are clamped.

    Returns
    -------
    torch.Tensor
        Window-sliced tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import window_slice
    >>> x = torch.randn(2, 50, 1)
    >>> y = window_slice(x, intensity=torch.tensor([0.4]))
    >>> assert y.shape == x.shape
    """
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()
    x_t = x.permute(0, 2, 1)  # (B, C, L)

    for b in range(B):
        frac = intensity[b].abs().clamp(max=0.8).item() * 0.5
        win = max(2, int(round(L * (1.0 - frac))))
        if win >= L:
            continue
        start = torch.randint(0, L - win + 1, (1,), device=x.device).item()
        segment = x_t[b : b + 1, :, start : start + win]
        resized = F.interpolate(segment, size=L, mode="linear", align_corners=False)
        out[b] = resized.squeeze(0).permute(1, 0)
    return out
    B, L, C = x.shape
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()
    x_t = x.permute(0, 2, 1)  # (B, C, L)

    for b in range(B):
        frac = intensity[b].abs().clamp(max=0.8).item() * 0.5
        win = max(2, int(round(L * (1.0 - frac))))
        if win >= L:
            continue
        start = torch.randint(0, L - win + 1, (1,), device=x.device).item()
        segment = x_t[b : b + 1, :, start : start + win]
        resized = F.interpolate(segment, size=L, mode="linear", align_corners=False)
        out[b] = resized.squeeze(0).permute(1, 0)
    return out


def time_mix(x: torch.Tensor, intensity: torch.Tensor) -> torch.Tensor:
    """Mix random temporally-aligned segments between paired samples.

    For each sample *i*, a random partner *j* is selected. A contiguous
    window of length ``round(L * (0.1 + 0.3 * intensity))`` is then mixed
    with alpha = 0.5 + 0.5 * intensity.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor of shape (batch_size, length, channels). Requires B >= 2.
    intensity : torch.Tensor
        Mixing fraction (0–1). Scalar or (batch_size,) tensor.

    Returns
    -------
    torch.Tensor
        Time-mixed tensor of the same shape as *x*.

    Examples
    --------
    >>> import torch
    >>> from foretools.tsaug.transformations import time_mix
    >>> x = torch.randn(4, 50, 1)  # need batch >= 2
    >>> y = time_mix(x, intensity=torch.tensor([0.5]))
    >>> assert y.shape == x.shape
    """
    B, L, C = x.shape
    if B < 2:
        return x
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()
    indices = torch.randperm(B, device=x.device)

    for b in range(B):
        partner = x[indices[b]]
        mix = intensity[b].clamp(max=1.0).item()
        if mix <= 0.0:
            continue
        win = max(1, int(round(L * (0.1 + 0.3 * mix))))
        start = torch.randint(0, L - win + 1, (1,), device=x.device).item()
        end = start + win
        alpha = 0.5 + 0.5 * mix
        out[b, start:end] = (1.0 - alpha) * x[b, start:end] + alpha * partner[start:end]
    return out
    B, L, C = x.shape
    if B < 2:
        return x
    intensity = _to_batch_intensity(intensity, B, x.device, x.dtype)
    out = x.clone()
    indices = torch.randperm(B, device=x.device)

    for b in range(B):
        partner = x[indices[b]]
        mix = intensity[b].clamp(max=1.0).item()
        if mix <= 0.0:
            continue
        win = max(1, int(round(L * (0.1 + 0.3 * mix))))
        start = torch.randint(0, L - win + 1, (1,), device=x.device).item()
        end = start + win
        alpha = 0.5 + 0.5 * mix
        out[b, start:end] = (1.0 - alpha) * x[b, start:end] + alpha * partner[start:end]
    return out


# Registry mapping indices to transformation functions
TRANSFORMATIONS = [
    raw,  # T1: Raw (index 0)
    jittering,  # T2: Jittering
    scaling,  # T3: Scaling
    resample,  # T4: Resample
    time_warp,  # T5: TimeWarp
    freq_warp,  # T6: FreqWarp
    mag_warp,  # T7: MagWarp
    time_mask,  # T8: TimeMask
    drift,  # T9: Drift
    permutation,  # T10: Permutation
    window_slice,  # T11: WindowSlice
    time_mix,  # T12: TimeMix
]

TRANSFORM_NAMES = [
    "Raw",
    "Jittering",
    "Scaling",
    "Resample",
    "TimeWarp",
    "FreqWarp",
    "MagWarp",
    "TimeMask",
    "Drift",
    "Permutation",
    "WindowSlice",
    "TimeMix",
]

NUM_TRANSFORMS = len(TRANSFORMATIONS)
