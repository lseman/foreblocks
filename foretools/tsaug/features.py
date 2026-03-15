"""
Time series feature extraction for AutoDA-Timeseries.

Extracts 24 descriptive statistics from each time series, forming a
feature vector F_i = fe(D_i) that captures autocorrelation, distribution,
and higher-order properties. These features remain static across augmentation
layers to preserve global context (Section 3.3).

Inspired by catch22 (Lubba et al., 2019) and tsfresh feature sets.
"""

import torch
import numpy as np
from typing import Optional


def extract_features(x: torch.Tensor) -> torch.Tensor:
    """Extract 24 descriptive statistics from a batch of time series.

    Args:
        x: (batch, length, channels) time series tensor.

    Returns:
        features: (batch, 24) feature vector.
    """
    B, L, C = x.shape
    # Flatten channels by computing features per channel then averaging
    features_list = []

    for c in range(C):
        xc = x[:, :, c]  # (B, L)
        feats = _compute_features(xc)  # (B, 24)
        features_list.append(feats)

    # Average features across channels
    features = torch.stack(features_list, dim=0).mean(dim=0)  # (B, 24)
    return features


def _compute_features(x: torch.Tensor) -> torch.Tensor:
    """Compute 24 features for a single-channel time series.

    Args:
        x: (batch, length) tensor.

    Returns:
        (batch, 24) feature tensor.
    """
    B, L = x.shape
    eps = 1e-8
    feats = []

    # --- Basic statistics (4 features) ---
    mean = x.mean(dim=1)
    std = x.std(dim=1) + eps
    feats.append(mean)                                          # 1: mean
    feats.append(std)                                           # 2: std
    feats.append(((x - mean.unsqueeze(1)) / std.unsqueeze(1)).pow(3).mean(dim=1))  # 3: skewness
    feats.append(((x - mean.unsqueeze(1)) / std.unsqueeze(1)).pow(4).mean(dim=1))  # 4: kurtosis

    # --- Distribution features (4 features) ---
    feats.append(x.median(dim=1).values)                        # 5: median
    feats.append(x.max(dim=1).values - x.min(dim=1).values)     # 6: range
    q25 = torch.quantile(x, 0.25, dim=1)
    q75 = torch.quantile(x, 0.75, dim=1)
    feats.append(q75 - q25)                                     # 7: IQR
    # Mode approximation via histogram: bin with highest count
    feats.append(_histogram_mode(x, num_bins=10))               # 8: histogram mode (10 bins)

    # --- Autocorrelation features (4 features) ---
    x_centered = x - mean.unsqueeze(1)
    var = (x_centered ** 2).mean(dim=1) + eps
    # lag-1 autocorrelation
    ac1 = (x_centered[:, :-1] * x_centered[:, 1:]).mean(dim=1) / var
    feats.append(ac1)                                           # 9: lag-1 autocorrelation
    # lag-2 autocorrelation
    if L > 2:
        ac2 = (x_centered[:, :-2] * x_centered[:, 2:]).mean(dim=1) / var
    else:
        ac2 = torch.zeros(B, device=x.device)
    feats.append(ac2)                                           # 10: lag-2 autocorrelation
    # First 1/e crossing of autocorrelation function
    feats.append(_first_ac_crossing(x_centered, var))           # 11: first 1/e crossing
    # Partial autocorrelation (approx via lag-1 and lag-2)
    pac = (ac2 - ac1 ** 2) / (1 - ac1 ** 2 + eps)
    feats.append(pac)                                           # 12: partial autocorrelation

    # --- Differencing / local variation features (4 features) ---
    dx = x[:, 1:] - x[:, :-1]
    feats.append(dx.mean(dim=1))                                # 13: mean of first differences
    feats.append(dx.std(dim=1))                                 # 14: std of first differences
    # Longest stretch of values above/below mean
    above = (x > mean.unsqueeze(1)).float()
    feats.append(_longest_stretch(above))                       # 15: longest stretch above mean
    below = 1.0 - above
    feats.append(_longest_stretch(below))                       # 16: longest stretch below mean

    # --- Entropy & complexity features (4 features) ---
    feats.append(_sample_entropy_approx(x))                     # 17: sample entropy approx
    feats.append(_spectral_entropy(x))                          # 18: spectral entropy
    # Number of mean crossings
    sign_changes = ((x[:, 1:] - mean.unsqueeze(1)) *
                    (x[:, :-1] - mean.unsqueeze(1))) < 0
    feats.append(sign_changes.float().sum(dim=1) / L)           # 19: mean crossing rate
    # Number of zero crossings (of differenced series)
    zc = (dx[:, 1:] * dx[:, :-1]) < 0
    feats.append(zc.float().sum(dim=1) / max(1, L - 2))        # 20: zero crossing rate of diff

    # --- Trend & higher-order features (4 features) ---
    t = torch.arange(L, dtype=x.dtype, device=x.device).unsqueeze(0).expand(B, -1)
    t_mean = t.mean(dim=1, keepdim=True)
    slope = ((t - t_mean) * (x - mean.unsqueeze(1))).sum(dim=1) / (
        (t - t_mean) ** 2
    ).sum(dim=1).clamp(min=eps)
    feats.append(slope)                                         # 21: linear trend slope
    # Fraction of values within 1 std of mean
    within_1std = ((x - mean.unsqueeze(1)).abs() < std.unsqueeze(1)).float().mean(dim=1)
    feats.append(within_1std)                                   # 22: fraction within 1 std
    # Peak-to-peak ratio (max of abs first diff / range)
    peak_ratio = dx.abs().max(dim=1).values / (
        x.max(dim=1).values - x.min(dim=1).values + eps
    )
    feats.append(peak_ratio)                                    # 23: peak-to-peak ratio
    # Energy ratio: ratio of energy in first vs second half
    half = L // 2
    e1 = (x[:, :half] ** 2).sum(dim=1)
    e2 = (x[:, half:] ** 2).sum(dim=1) + eps
    feats.append(e1 / e2)                                       # 24: energy ratio

    features = torch.stack(feats, dim=1)  # (B, 24)
    # Replace NaN/Inf with 0
    features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def _histogram_mode(x: torch.Tensor, num_bins: int = 10) -> torch.Tensor:
    """Approximate mode via histogram bin centers."""
    B, L = x.shape
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    x_range = (x_max - x_min).clamp(min=1e-8)
    x_norm = (x - x_min) / x_range * (num_bins - 1)
    bins = x_norm.long().clamp(0, num_bins - 1)

    modes = []
    for b in range(B):
        counts = torch.bincount(bins[b], minlength=num_bins)
        mode_bin = counts.argmax().float()
        mode_val = x_min[b, 0] + (mode_bin + 0.5) / num_bins * x_range[b, 0]
        modes.append(mode_val)
    return torch.stack(modes)


def _first_ac_crossing(x_centered: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Find the first lag where autocorrelation drops below 1/e."""
    B, L = x_centered.shape
    threshold = 1.0 / np.e
    max_lag = min(L // 2, 50)

    result = torch.full((B,), float(max_lag), device=x_centered.device)
    for lag in range(1, max_lag):
        ac = (x_centered[:, :-lag] * x_centered[:, lag:]).mean(dim=1) / (var + 1e-8)
        crossed = ac < threshold
        result = torch.where(crossed & (result == max_lag), torch.tensor(float(lag), device=result.device), result)

    return result / max_lag  # Normalize


def _longest_stretch(binary: torch.Tensor) -> torch.Tensor:
    """Find the longest consecutive stretch of 1s in a binary tensor."""
    B, L = binary.shape
    result = torch.zeros(B, device=binary.device)
    current = torch.zeros(B, device=binary.device)
    for i in range(L):
        current = (current + binary[:, i]) * binary[:, i]
        result = torch.maximum(result, current)
    return result / L  # Normalize by length


def _sample_entropy_approx(x: torch.Tensor) -> torch.Tensor:
    """Quick approximation of sample entropy via binned distribution entropy."""
    B, L = x.shape
    num_bins = 20
    x_min = x.min(dim=1, keepdim=True).values
    x_max = x.max(dim=1, keepdim=True).values
    x_norm = (x - x_min) / (x_max - x_min + 1e-8)
    bins = (x_norm * (num_bins - 1)).long().clamp(0, num_bins - 1)

    entropies = []
    for b in range(B):
        counts = torch.bincount(bins[b], minlength=num_bins).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        ent = -(probs * probs.log()).sum()
        entropies.append(ent)
    return torch.stack(entropies)


def _spectral_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute spectral entropy from the power spectral density."""
    B, L = x.shape
    X = torch.fft.rfft(x, dim=1)
    psd = (X.abs() ** 2) / L
    psd_norm = psd / (psd.sum(dim=1, keepdim=True) + 1e-8)
    psd_norm = psd_norm.clamp(min=1e-10)
    entropy = -(psd_norm * psd_norm.log()).sum(dim=1)
    # Normalize by log of number of frequency bins
    n_freq = psd.shape[1]
    entropy = entropy / (np.log(n_freq) + 1e-8)
    return entropy


# Feature dimension constant
FEATURE_DIM = 24
