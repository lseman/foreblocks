"""Safe statistical helpers that guard against overflow in scipy.stats.

Provides:
  - ``safe_skew``, ``safe_kurtosis`` — scalar functions on 1-D arrays
  - ``safe_row_skew``, ``safe_row_kurtosis`` — row-wise operations on DataFrames
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import warnings

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    pass


def _clip_for_moments(x: np.ndarray, clip_std: int = 50) -> np.ndarray:
    """Clip to prevent overflow in moment-based stats.

    Two-stage guard:
    1. If data spans >100 orders of magnitude → return zeros (moments meaningless).
    2. Otherwise clip to ±clip_std * MAD around median.
    """
    x = np.asarray(x, dtype=np.float64)
    finite = x[np.isfinite(x)]
    if finite.size < 5:
        return x

    lo, hi = float(np.min(finite)), float(np.max(finite))
    # Log-range guard
    if hi > 0 and lo > 0:
        if np.log10(hi) - np.log10(lo) > 100:
            return np.full_like(finite, 0.0, dtype=np.float64)
    elif hi < 0 and lo < 0:
        if np.log10(-lo) - np.log10(-hi) > 100:
            return np.full_like(finite, 0.0, dtype=np.float64)
    elif abs(hi) > 1e100 or abs(lo) > 1e100:
        return np.full_like(finite, 0.0, dtype=np.float64)

    median = np.median(finite)
    mad = np.median(np.abs(finite - median))
    bound = clip_std * max(mad, 1e-15)
    if bound < 1e-12:
        return np.full_like(x, 0.0, dtype=np.float64)
    return np.clip(x, median - bound, median + bound)


# ── scalar functions ────────────────────────────────────────────────────


def safe_skew(x: np.ndarray, bias: bool = False) -> float:
    """Compute skewness with overflow guard.  Returns NaN on failure."""
    finite = x[np.isfinite(x)]
    if finite.size < 8:
        return float("nan")
    clipped = _clip_for_moments(finite, clip_std=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(stats.skew(clipped, bias=bias))
        except (FloatingPointError, ValueError):
            return float("nan")


def safe_kurtosis(x: np.ndarray, fisher: bool = True, bias: bool = False) -> float:
    """Compute kurtosis with overflow guard.  Returns NaN on failure."""
    finite = x[np.isfinite(x)]
    if finite.size < 10:
        return float("nan")
    clipped = _clip_for_moments(finite, clip_std=50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return float(stats.kurtosis(clipped, fisher=fisher, bias=bias))
        except (FloatingPointError, ValueError):
            return float("nan")


# ── row-wise functions ──────────────────────────────────────────────────


def safe_row_skew(df: pd.DataFrame) -> pd.Series:
    """Row-wise skew with overflow guard.  Returns a Series."""

    def _row_skew(row: pd.Series) -> float:
        finite = row.dropna().to_numpy(dtype=float)
        if finite.size < 8:
            return float("nan")
        clipped = _clip_for_moments(finite, clip_std=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(stats.skew(clipped))
            except Exception:
                return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = df.skew(axis=1)  # type: ignore[call-arg]
        except (RuntimeError, ValueError):
            result = df.apply(_row_skew)

    # Recompute any NaN rows safely
    if result.isna().any():  # type: ignore[union-attr]
        for idx in result.index[result.isna()]:  # type: ignore[union-attr]
            result.loc[idx] = _row_skew(df.loc[idx])  # type: ignore[index]
    return result


def safe_row_kurtosis(df: pd.DataFrame) -> pd.Series:
    """Row-wise kurtosis with overflow guard.  Returns a Series."""

    def _row_kurtosis(row: pd.Series) -> float:
        finite = row.dropna().to_numpy(dtype=float)
        if finite.size < 10:
            return float("nan")
        clipped = _clip_for_moments(finite, clip_std=50)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return float(stats.kurtosis(clipped))
            except Exception:
                return float("nan")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = df.kurtosis(axis=1)  # type: ignore[call-arg]
        except (RuntimeError, ValueError):
            result = df.apply(_row_kurtosis)

    if result.isna().any():  # type: ignore[union-attr]
        for idx in result.index[result.isna()]:  # type: ignore[union-attr]
            result.loc[idx] = _row_kurtosis(df.loc[idx])  # type: ignore[index]
    return result
