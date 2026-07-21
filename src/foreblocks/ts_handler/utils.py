"""foreblocks.ts_handler.utils.

Basic utility functions for time-series preprocessing.

"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from scipy.stats import kurtosis, skew


def _as_2d(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [T,D], got {x.shape}")
    return x


def _prepare_series_for_diagnostics(
    values: np.ndarray, *, max_points: int = 4096
) -> tuple[np.ndarray, int]:
    clean = np.asarray(values, dtype=float)
    clean = clean[~np.isnan(clean)]
    if clean.size == 0:
        return clean, 1
    stride = max(1, int(np.ceil(clean.size / max_points)))
    if stride > 1:
        clean = clean[::stride]
    return clean, stride


def _select_diagnostic_features(n_features: int, max_features: int = 32) -> np.ndarray:
    if n_features <= max_features:
        return np.arange(n_features, dtype=int)
    return np.unique(np.linspace(0, n_features - 1, num=max_features, dtype=int))


def _longest_nan_run(data: np.ndarray) -> int:
    x = _as_2d(data)
    best = 0
    for j in range(x.shape[1]):
        run = 0
        for is_nan in np.isnan(x[:, j]):
            if is_nan:
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best


def _linear_interpolate_2d(data: np.ndarray) -> np.ndarray:
    x = _as_2d(data).copy()
    idx = np.arange(x.shape[0], dtype=float)

    for j in range(x.shape[1]):
        col = x[:, j]
        mask = ~np.isnan(col)
        if mask.all():
            continue
        valid_count = int(mask.sum())
        if valid_count == 0:
            continue
        if valid_count == 1:
            col[~mask] = col[mask][0]
            continue
        col[~mask] = np.interp(idx[~mask], idx[mask], col[mask])

    return x


def _mean_fill_2d(data: np.ndarray) -> np.ndarray:
    x = _as_2d(data).copy()
    means = np.nanmean(x, axis=0)
    means = np.where(np.isfinite(means), means, 0.0)
    nan_rows, nan_cols = np.where(np.isnan(x))
    if nan_rows.size:
        x[nan_rows, nan_cols] = means[nan_cols]
    return x


def _lag_fill_2d(data: np.ndarray, lag: int) -> np.ndarray:
    x = _as_2d(data).copy()
    if lag <= 0:
        return x

    nan_rows, nan_cols = np.where(np.isnan(x))
    for row, col in zip(nan_rows, nan_cols):
        donor = row - lag
        if donor >= 0 and np.isfinite(x[donor, col]):
            x[row, col] = x[donor, col]
    return x


def _hybrid_impute(x: np.ndarray) -> np.ndarray:
    filled = _linear_interpolate_2d(x)
    filled = _lag_fill_2d(filled, 24)
    return _mean_fill_2d(filled)


def _safe_corr_1d(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or y.size < 2:
        return 0.0
    sx = float(np.std(x))
    sy = float(np.std(y))
    if np.isclose(sx, 0.0) or np.isclose(sy, 0.0):
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return 0.0 if not np.isfinite(corr) else float(corr)


def _mean_abs_autocorr(x: np.ndarray, max_lag: int = 20) -> float:
    vals = np.asarray(x, dtype=float)
    if vals.size < 3:
        return 0.0
    upper = min(int(max_lag), vals.size - 1)
    if upper < 1:
        return 0.0
    scores = []
    for lag in range(1, upper + 1):
        scores.append(abs(_safe_corr_1d(vals[:-lag], vals[lag:])))
    return float(np.mean(scores)) if scores else 0.0


def _rank_normalize(values: np.ndarray, invert: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    n = arr.size
    if n == 1:
        return np.array([0.5], dtype=float)

    finite = np.isfinite(arr)
    out = np.ones(n, dtype=float)
    if finite.sum() < 2:
        out[:] = 0.5
        return 1.0 - out if invert else out

    order = np.argsort(arr[finite])
    finite_idx = np.where(finite)[0]
    ranks = np.empty(finite.sum(), dtype=float)
    ranks[order] = np.arange(finite.sum(), dtype=float) / max(1, finite.sum() - 1)
    out[finite_idx] = ranks
    return 1.0 - out if invert else out


def _cyclical_encode(values: np.ndarray, period: float) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float32)
    angle = (2.0 * np.pi * vals) / max(float(period), 1.0)
    return np.column_stack((np.sin(angle), np.cos(angle))).astype(
        np.float32, copy=False
    )


def apply_log_transform(
    data: np.ndarray,
    log_flags: list[bool],
    offsets: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"apply_log_transform expects 2D array, got shape {x.shape}")

    D = x.shape[1]
    if len(log_flags) != D:
        raise ValueError(f"log_flags length {len(log_flags)} != D {D}")

    flags = np.asarray(log_flags, dtype=bool)

    if offsets is None:
        offsets = np.zeros(D, dtype=float)
        if np.any(flags):
            mins = np.nanmin(x[:, flags], axis=0)
            mins = np.where(np.isfinite(mins), mins, 0.0)
            offsets[flags] = np.maximum(0.0, -mins + 1.0)
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (D,):
            raise ValueError(f"offsets shape {offsets.shape} != ({D},)")

    out = x.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.any(flags):
            out[:, flags] = np.log(out[:, flags] + offsets[flags])
    return out, offsets


def compute_basic_stats(
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    valid_mask = ~np.isnan(x)
    coverage = np.mean(valid_mask, axis=0)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    skews = skew(x, nan_policy="omit")
    kurts = kurtosis(x, nan_policy="omit")
    return coverage, means, stds, skews, kurts
