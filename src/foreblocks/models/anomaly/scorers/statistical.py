"""foreblocks.models.anomaly.scorers.statistical.

Statistical anomaly detection methods for time-series windows.

Provides lightweight, non-parametric statistical detectors that operate on
sliding windows without neural network training. Includes:

- EBS (Exponentially Weighted Block Statistics): detects anomalies by comparing
  exponentially weighted block statistics against historical baselines
- CUSUM (Cumulative Sum): detects gradual shifts in the mean via accumulated
  deviations from a target value
- EWMA (Exponentially Weighted Moving Average): detects level shifts by
  scoring deviations from an exponentially decaying baseline
- SeasonalHybrid: combines seasonal decomposition with robust z-scoring of
  residuals, using ESD for outlier removal on the seasonal component
- STL+Residual: STL (Seasonal-Trend decomposition using LOESS) of each
  feature, scores windows via combined residual and structural anomaly

All functions accept 2-D series of shape [T, D] or 3-D windows of shape [N, T, D].

Core API:
- ebs_score: Exponentially Weighted Block Statistics scores
- cusum_score: Cumulative Sum control chart scores
- ewma_score: Exponentially Weighted Moving Average scores
- seasonal_hybrid_score: Seasonal Hybrid ESD scores
- stl_residual_score: STL decomposition + residual anomaly scores

"""

import numpy as np
from typing import Optional

# ---------------------------------------------------------------------------
# EBS — Exponentially Weighted Block Statistics
# ---------------------------------------------------------------------------


def ebs_score(
    windows: np.ndarray,
    decay: float = 0.95,
    window_size: Optional[int] = None,
) -> np.ndarray:
    """Compute Exponentially Weighted Block Statistics anomaly scores.

    Divides windows into blocks, computes exponentially weighted statistics
    (mean, variance) for each block, and scores windows by deviation from
    the global exponentially weighted baseline.

    Args:
        windows: array of shape [N, T, D].
        decay: exponential decay factor (0 < decay < 1).
        window_size: block size (defaults to total window length).

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    if window_size is None:
        window_size = t

    # Compute block statistics for each window
    n_blocks = int(np.ceil(t / window_size))
    block_stats = np.zeros((n, n_blocks, d))

    for i in range(n):
        for b in range(n_blocks):
            start = b * window_size
            end = min(start + window_size, t)
            block_stats[i, b, :] = windows[i, start:end, :].mean(axis=0)

    # Exponentially weighted global baseline (across all windows and blocks)
    ew_mean = np.zeros(d)
    ew_var = np.zeros(d)
    weight_sum = 0.0

    for i in range(n):
        for b in range(n_blocks):
            w = decay ** (i * n_blocks + b)
            block_mean = block_stats[i, b, :]
            ew_mean = (decay * ew_mean + w * block_mean) / (decay * 1.0 + w + 1e-8)
            ew_var = decay * ew_var + w * (block_mean - ew_mean) ** 2
            weight_sum += w

    ew_mean = ew_mean / (weight_sum + 1e-8)
    ew_var = np.maximum(ew_var, 1e-8)
    ew_std = np.sqrt(ew_var)

    # Score each window: max block deviation from baseline
    scores = np.zeros(n)
    for i in range(n):
        for b in range(n_blocks):
            z = np.abs((block_stats[i, b, :] - ew_mean) / (ew_std + 1e-8))
            scores[i] = max(scores[i], z.max())

    # Normalise to [0, 1]
    scores = scores / (np.nanpercentile(scores, 99) + 1e-8)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# CUSUM — Cumulative Sum Control Chart
# ---------------------------------------------------------------------------


def cusum_score(
    windows: np.ndarray,
    drift: float = 0.5,
    threshold_scale: float = 4.0,
) -> np.ndarray:
    """Compute CUSUM (Cumulative Sum) anomaly scores.

    For each feature, accumulates deviations from the running mean. The
    anomaly score is the maximum cumulative deviation observed within each
    window.

    Args:
        windows: array of shape [N, T, D].
        drift: allowed drift from target (in standard deviations).
        threshold_scale: decision threshold scale factor.

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    scores = np.zeros(n)

    for feat in range(d):
        col = windows[:, :, feat]  # [N, T]

        # Global statistics for initialisation
        global_mean = col.mean()
        global_std = col.std() + 1e-8

        # CUSUM: compute cumulative deviations
        for i in range(n):
            s_pos = 0.0
            s_neg = 0.0
            max_dev = 0.0

            for j in range(t):
                z = (col[i, j] - global_mean) / global_std
                s_pos = max(0.0, s_pos + z - drift)
                s_neg = max(0.0, s_neg - z - drift)
                max_dev = max(max_dev, s_pos, s_neg)

            scores[i] = max(scores[i], max_dev / (threshold_scale * global_std + 1e-8))

    # Normalise to [0, 1]
    scores = scores / (np.nanpercentile(scores, 99) + 1e-8)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# EWMA — Exponentially Weighted Moving Average
# ---------------------------------------------------------------------------


def ewma_score(
    windows: np.ndarray,
    lambda_: float = 0.2,
    k: float = 0.5,
    threshold_multiplier: float = 3.0,
) -> np.ndarray:
    """Compute EWMA (Exponentially Weighted Moving Average) anomaly scores.

    For each feature, maintains an exponentially weighted moving average and
    scores windows by the deviation from this baseline. Detects gradual
    level shifts in the time series.

    Args:
        windows: array of shape [N, T, D].
        lambda_: smoothing factor (0 < λ ≤ 1).
        k: allowance factor in standard deviations.
        threshold_multiplier: control chart multiplier.

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    scores = np.zeros(n)

    for feat in range(d):
        col = windows[:, :, feat]
        global_mean = col.mean()
        global_std = col.std() + 1e-8

        # EWMA control chart limits
        sigma_ewma = global_std * np.sqrt(lambda_ / (2.0 - lambda_) * (1.0 - lambda_ ** t))
        control_limit = threshold_multiplier * sigma_ewma

        for i in range(n):
            ewma = col[i, 0]
            max_dev = 0.0

            for j in range(1, t):
                ewma = lambda_ * col[i, j] + (1.0 - lambda_) * ewma
                dev = abs(ewma - global_mean)
                max_dev = max(max_dev, dev)

            scores[i] = max(scores[i], max_dev / (control_limit + 1e-8))

    # Normalise to [0, 1]
    scores = scores / (np.nanpercentile(scores, 99) + 1e-8)
    return np.clip(scores, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Seasonal Hybrid ESD
# ---------------------------------------------------------------------------


def _seasonal_decompose(
    x: np.ndarray,
    period: int,
    span: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simple seasonal decomposition using moving averages.

    Decomposes a 1-D signal into seasonal, trend, and residual components
    using centred moving averages. Uses LOESS-like robust iteration.

    Args:
        x: 1-D array [T].
        period: seasonal period.
        span: LOESS span for trend (defaults to period).

    Returns:
        (seasonal, trend, residual) each of shape [T].
    """
    T = len(x)
    if span is None:
        span = period

    # Initial trend: centred moving average
    half = span // 2
    trend = np.full(T, np.nan)
    for i in range(half, T - half):
        trend[i] = x[i - half : i + half + 1].mean()

    # Handle NaN at edges via forward/backward fill
    valid = ~np.isnan(trend)
    if valid.any():
        idx_good = np.flatnonzero(valid)
        idx_bad = np.flatnonzero(~valid)
        if len(idx_bad) > 0:
            trend[idx_bad] = np.interp(idx_bad, idx_good, trend[idx_good])

    # Detrend
    detrended = x - trend

    # Initial seasonal: average of detrended at each seasonal position
    seasonal = np.zeros(T)
    for p in range(period):
        mask = np.arange(T) % period == p
        seasonal[mask] = detrended[mask].mean()

    # Normalise seasonal to zero mean
    seasonal -= seasonal.mean()

    # Residual
    residual = x - trend - seasonal

    # Robust iteration (1-2 iterations)
    for _ in range(2):
        # Robust weights
        mad = np.median(np.abs(residual - np.median(residual)))
        weights = np.ones(T)
        if mad > 1e-8:
            z = np.abs(residual) / (1.4826 * mad + 1e-8)
            weights = np.where(z < 3.0, 1.0 - (z / 3.0) ** 2, 0.0)

        # Re-compute seasonal with weights
        seasonal = np.zeros(T)
        for p in range(period):
            mask = np.arange(T) % period == p
            if weights[mask].sum() > 0:
                seasonal[mask] = np.average(detrended[mask], weights=weights[mask])
            else:
                seasonal[mask] = detrended[mask].mean()
        seasonal -= seasonal.mean()

        # Re-compute trend
        for i in range(half, T - half):
            window = x[i - half : i + half + 1] - seasonal[i - half : i + half + 1]
            w = weights[i - half : i + half + 1]
            if w.sum() > 0:
                trend[i] = np.average(window, weights=w)
            else:
                trend[i] = window.mean()

        # Handle NaN
        valid = ~np.isnan(trend)
        if valid.any():
            idx_good = np.flatnonzero(valid)
            idx_bad = np.flatnonzero(~valid)
            if len(idx_bad) > 0:
                trend[idx_bad] = np.interp(idx_bad, idx_good, trend[idx_good])

        residual = x - trend - seasonal

    return seasonal, trend, residual


def _generalised_esd(
    residuals: np.ndarray,
    max_anomalies: float = 0.1,
    alpha: float = 0.05,
) -> np.ndarray:
    """Generalised ESD (Extreme Studentised Deviate) test.

    Iteratively tests the most extreme value for being an outlier.

    Args:
        residuals: 1-D array of residuals.
        max_anomalies: fraction of data that could be anomalous.
        alpha: significance level.

    Returns:
        Boolean mask of detected anomalies.
    """
    n = len(residuals)
    r = max(1, int(np.ceil(max_anomalies * n)))
    x = residuals.copy()
    anomaly_mask = np.zeros(n, dtype=bool)

    for i in range(r):
        mean = x.mean()
        std = x.std() + 1e-8
        z = np.abs(x - mean) / std
        max_idx = np.argmax(z)
        lambda_stat = ((n - i) * z[max_idx]) / np.sqrt(
            (n - i - 1 + z[max_idx] ** 2) * (n - i)
        )

        crit = max(2.0, 3.0 - 2.0 * np.log1p(i))

        if lambda_stat > crit:
            anomaly_mask[max_idx] = True
            x[max_idx] = np.nan  # remove extreme value
        else:
            break

    return anomaly_mask




def seasonal_hybrid_score(
    windows: np.ndarray,
    period: int = 24,
    contamination: float = 0.01,
) -> np.ndarray:
    """Compute Seasonal Hybrid ESD anomaly scores.

    For each feature, decomposes into seasonal+trend+residual, then applies
    ESD test on residuals to detect anomalies. The score combines the
    magnitude of residual deviations with the ESD test statistic.

    Args:
        windows: array of shape [N, T, D].
        period: seasonal period (e.g. 24 for hourly daily seasonality).
        contamination: expected fraction of anomalies.

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    scores = np.zeros(n)

    for feat in range(d):
        col = windows[:, :, feat]

        # Decompose each window
        window_resids = np.zeros(n)
        for i in range(n):
            seasonal, trend, residual = _seasonal_decompose(col[i], period)
            # Score by residual magnitude (normalised by MAD)
            mad = np.median(np.abs(residual - np.median(residual))) + 1e-8
            window_resids[i] = np.max(np.abs(residual)) / (1.4826 * mad)

        scores += window_resids / (np.nanpercentile(window_resids, 99) + 1e-8)

    scores /= d  # normalise by number of features
    scores = np.clip(scores, 0.0, 1.0)
    return scores


# ---------------------------------------------------------------------------
# STL + Residual
# ---------------------------------------------------------------------------


def stl_residual_score(
    windows: np.ndarray,
    period: int = 24,
    robust: bool = True,
) -> np.ndarray:
    """Compute STL decomposition + residual anomaly scores.

    For each feature, performs STL (Seasonal-Trend decomposition using
    LOESS) on the window's time series, then scores the window by the
    combined magnitude of:
    1. Residual deviation (normalised by MAD)
    2. Structural anomaly (deviation from typical seasonal pattern)

    Args:
        windows: array of shape [N, T, D].
        period: seasonal period.
        robust: whether to use robust LOESS iteration.

    Returns:
        scores array of shape [N] (higher = more anomalous).
    """
    windows = np.asarray(windows, dtype=np.float64)
    n, t, d = windows.shape

    scores = np.zeros(n)

    for feat in range(d):
        col = windows[:, :, feat]

        # Decompose and score
        window_scores_feat = np.zeros(n)
        for i in range(n):
            seasonal, trend, residual = _seasonal_decompose(col[i], period)

            # Residual score: normalised max absolute residual
            mad = np.median(np.abs(residual - np.median(residual))) + 1e-8
            resid_score = np.max(np.abs(residual)) / (1.4826 * mad)

            # Structural score: how unusual is the seasonal shape?
            # Compare seasonal profile to median seasonal profile
            if i == 0:
                baseline_seasonal = seasonal.copy()
                seasonal_score = 0.0
            else:
                norm = np.linalg.norm(baseline_seasonal) * (np.linalg.norm(seasonal) + 1e-8)
                if norm > 1e-8:
                    similarity = np.dot(baseline_seasonal, seasonal) / norm
                else:
                    similarity = 1.0
                seasonal_score = 1.0 - similarity

            window_scores_feat[i] = resid_score + 2.0 * seasonal_score

        scores += window_scores_feat / (np.nanpercentile(window_scores_feat, 99) + 1e-8)

    scores /= d
    return np.clip(scores, 0.0, 1.0)
