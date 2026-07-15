"""foreblocks.ts_handler.auto_filter.filters.bilateral.

Bilateral filter (1-D, edge-preserving).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


@register_filter("Bilateral")
def bilateral_filter(
    ts: pd.Series,
    sigma_t: float = 5.0,
    sigma_v: float | None = None,
) -> pd.Series:
    """Edge-preserving 1-D bilateral filter.

    Combines a Gaussian spatial (time-domain) kernel with a Gaussian range
    (value-domain) kernel.  Unlike a pure Gaussian, it suppresses noise while
    preserving abrupt level shifts and local extrema.

    Parameters
    ----------
    sigma_t:
        Temporal kernel width in samples.
    sigma_v:
        Value-range kernel width.  Defaults to the MAD-based noise estimate
        of the series (auto-scaled per signal).
    """
    y = ts.values.astype(float)
    N = len(y)

    if sigma_v is None:
        sigma_v = float(np.median(np.abs(np.diff(y))) / 0.6745) + 1e-6
    sigma_v = max(float(sigma_v), 1e-6)
    sigma_t = max(float(sigma_t), 0.5)

    half_win = int(np.ceil(3 * sigma_t))
    result = np.empty(N, dtype=float)

    for i in range(N):
        lo = max(0, i - half_win)
        hi = min(N, i + half_win + 1)
        idx = np.arange(lo, hi)
        w_t = np.exp(-0.5 * ((idx - i) / sigma_t) ** 2)
        w_v = np.exp(-0.5 * ((y[lo:hi] - y[i]) / sigma_v) ** 2)
        w = w_t * w_v
        result[i] = np.dot(w, y[lo:hi]) / (w.sum() + 1e-12)

    return _as_series(result, ts.index, name="bilateral")
