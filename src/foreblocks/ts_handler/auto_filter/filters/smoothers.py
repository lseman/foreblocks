"""foreblocks.ts_handler.auto_filter.filters.smoothers.

Bayesian and patch-based smoothers: Gaussian Process, Non-local Means.

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


@register_filter("Gaussian Process")
def gaussian_process_smoother(
    ts: pd.Series,
    length_scale: float = 12.0,
    noise: float = 0.08,
    max_inducing: int = 256,
) -> pd.Series:
    y = ts.values.astype(float)
    n = len(y)
    if n < 4:
        return ts.copy()

    m = min(max(8, int(max_inducing)), n)
    inducing_idx = np.unique(np.linspace(0, n - 1, m).round().astype(int))
    x_train = inducing_idx.astype(float)[:, None]
    y_train = y[inducing_idx]
    x_all = np.arange(n, dtype=float)[:, None]

    length_scale = max(float(length_scale), 1e-3)
    signal_var = max(float(np.var(y_train)), 1e-8)
    noise_var = max(float(noise), 1e-6) ** 2 * signal_var

    def rbf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        dist2 = (a - b.T) ** 2
        return signal_var * np.exp(-0.5 * dist2 / (length_scale**2))

    K = rbf(x_train, x_train)
    K[np.diag_indices_from(K)] += noise_var + 1e-8
    Ks = rbf(x_all, x_train)
    centered = y_train - float(np.mean(y_train))
    try:
        alpha = np.linalg.solve(K, centered)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K, centered, rcond=None)[0]
    pred = float(np.mean(y_train)) + Ks @ alpha
    return _as_series(pred, ts.index, name="gp")


@register_filter("Non-local Means 1D")
def non_local_means_filter(
    ts: pd.Series,
    patch_radius: int = 3,
    search_radius: int = 24,
    h: float | None = None,
) -> pd.Series:
    y = ts.values.astype(float)
    n = len(y)
    if n < 4:
        return ts.copy()

    patch_radius = max(1, int(patch_radius))
    search_radius = max(patch_radius + 1, int(search_radius))
    padded = np.pad(y, (patch_radius, patch_radius), mode="reflect")
    patches = np.stack([padded[i : i + 2 * patch_radius + 1] for i in range(n)])
    if h is None:
        h = float(np.median(np.abs(np.diff(y))) / 0.6745) + 1e-6
    h = max(float(h), 1e-6)

    out = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - search_radius)
        hi = min(n, i + search_radius + 1)
        d2 = np.mean((patches[lo:hi] - patches[i]) ** 2, axis=1)
        weights = np.exp(-d2 / (h * h))
        out[i] = np.dot(weights, y[lo:hi]) / max(float(weights.sum()), 1e-12)
    return _as_series(out, ts.index, name="nlm")
