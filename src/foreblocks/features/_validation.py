"""Input contracts shared by the feature transforms."""

from __future__ import annotations

import numpy as np


def as_panel(x, *, min_length: int = 2, allow_empty: bool = False) -> np.ndarray:
    """Coerce `[N, L]` or `[N, C, L]` series to a contiguous float32 `[N, C, L]`."""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    if x.ndim != 3:
        raise ValueError("Expected series with shape [samples, time] or [samples, channels, time].")
    if x.shape[1] < 1 or x.shape[2] < min_length:
        raise ValueError(f"Expected at least one channel and time length >= {min_length}.")
    if not allow_empty and len(x) == 0:
        raise ValueError("At least one series is required.")
    if not np.isfinite(x).all():
        raise ValueError("Series must contain only finite values.")
    return np.ascontiguousarray(x)


def input_length(x) -> int:
    """Series length from an int, or from the last axis of an array."""
    if isinstance(x, (int, np.integer)):
        return int(x)
    return int(np.shape(x)[-1])


def normalize_series(x, eps: float = 1e-6) -> np.ndarray:
    """Per-series (last-axis) z-score with a numerical floor on the scale."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    std = np.maximum(x.std(axis=-1, keepdims=True), eps)
    return ((x - mean) / std).astype(np.float32)
