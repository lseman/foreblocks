"""foreblocks.ts_handler.auto_filter.filters.utils.

Utility functions for auto-filter filters.

"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _as_series(
    values: np.ndarray, index: pd.Index, name: str | None = None
) -> pd.Series:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if len(arr) != len(index):
        raise ValueError(
            f"Filtered output length {len(arr)} != input length {len(index)}."
        )
    return pd.Series(arr, index=index, name=name)


def _resize_to_match_length(values: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == target_len:
        return arr
    if arr.size == 0:
        raise ValueError("Cannot resize an empty reconstruction.")
    if target_len <= 0:
        raise ValueError(f"Target length must be positive, got {target_len}.")
    if arr.size == 1:
        return np.full(target_len, float(arr[0]), dtype=float)

    source_grid = np.linspace(0.0, 1.0, num=arr.size)
    target_grid = np.linspace(0.0, 1.0, num=target_len)
    return np.interp(target_grid, source_grid, arr)


def _valid_odd_window(n: int, preferred: int, minimum: int = 3) -> int:
    if n <= 1:
        return 1
    w = max(int(preferred), minimum)
    if w % 2 == 0:
        w += 1
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(w, 1)


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.size < 2:
        return 0.0
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return 0.0
    c = np.corrcoef(x, y)[0, 1]
    return 0.0 if np.isnan(c) else float(c)


def _autocorr(x: np.ndarray, lag: int) -> float:
    if lag <= 0 or lag >= len(x):
        return 0.0
    return _safe_corr(x[:-lag], x[lag:])
