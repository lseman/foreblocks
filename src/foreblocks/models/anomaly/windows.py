"""Anomaly preprocessing, window-score alignment, and robust thresholds.

Generic window construction is provided by ``foreblocks.data.windowing`` and
re-exported here for compatibility with existing anomaly callers.
"""

from __future__ import annotations

import numpy as np

from foreblocks.data.windowing import (
    as_2d_array as as_2d_array,
    build_sliding_windows as build_sliding_windows,
)


def fill_nan_forward(values: np.ndarray) -> np.ndarray:
    x = as_2d_array(values).copy()
    for j in range(x.shape[1]):
        col = x[:, j]
        finite = np.isfinite(col)
        if not finite.any():
            x[:, j] = 0.0
            continue
        first = np.flatnonzero(finite)[0]
        col[:first] = col[first]
        for i in range(first + 1, len(col)):
            if not np.isfinite(col[i]):
                col[i] = col[i - 1]
    return x


def map_window_scores(
    scores: np.ndarray,
    series_length: int,
    window_size: int,
    *,
    align: str = "end",
    reduce: str = "max",
) -> np.ndarray:
    raw = np.asarray(scores, dtype=np.float32)
    if raw.ndim == 2:
        raw = raw.mean(axis=1)
    if raw.ndim != 1:
        raise ValueError(f"Expected [N] or [N,D] scores, got shape {raw.shape}")

    out = np.full(int(series_length), np.nan, dtype=np.float32)
    counts = np.zeros(int(series_length), dtype=np.float32)

    for i, score in enumerate(raw):
        if align == "end":
            idx = i + window_size - 1
        elif align == "center":
            idx = i + window_size // 2
        elif align == "all":
            start = i
            stop = min(i + window_size, series_length)
            segment = out[start:stop]
            if reduce == "mean":
                finite = np.nan_to_num(segment, nan=0.0)
                out[start:stop] = finite + float(score)
                counts[start:stop] += 1.0
            elif reduce == "max":
                out[start:stop] = np.fmax(segment, float(score))
            else:
                raise ValueError("reduce must be 'max' or 'mean'")
            continue
        else:
            raise ValueError("align must be 'end', 'center', or 'all'")

        if 0 <= idx < series_length:
            out[idx] = float(score)

    if align == "all" and reduce == "mean":
        mask = counts > 0
        out[mask] = out[mask] / counts[mask]
    return out


def robust_threshold(
    scores: np.ndarray,
    *,
    contamination: float = 0.01,
    min_z: float = 3.5,
) -> float:
    finite = np.asarray(scores, dtype=np.float32)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("inf")

    contamination = float(np.clip(contamination, 0.0, 0.5))
    percentile = 100.0 * (1.0 - contamination)
    q_cut = float(np.nanpercentile(finite, percentile))

    median = float(np.nanmedian(finite))
    mad = float(np.nanmedian(np.abs(finite - median)))
    if mad <= 1e-8:
        return q_cut
    z_cut = median + float(min_z) * 1.4826 * mad
    return max(q_cut, z_cut)
