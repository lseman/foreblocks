"""foreblocks.ts_handler.filters.utils.

Helper functions for time-series filters.

"""

from __future__ import annotations

import numpy as np


def _as_2d(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"data must be 2D [T,F], got shape={x.shape}")
    return x


def _odd_at_least(n: int, min_odd: int) -> int:
    n = int(max(n, min_odd))
    if n % 2 == 0:
        n += 1
    return n


def _nan_interp_1d(x: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs (1D). If too few points, returns copy."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_nan_interp_1d expects 1D")
    if not np.isnan(x).any():
        return x.copy()

    idx = np.arange(x.size)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return x.copy()

    out = x.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return out


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)
