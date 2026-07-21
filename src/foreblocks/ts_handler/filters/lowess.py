"""foreblocks.ts_handler.filters.lowess.

LOWESS (locally weighted scatterplot smoothing) filter.

"""

from __future__ import annotations

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

from foreblocks.ts_handler.filters.utils import _as_2d


def lowess_filter(
    data: np.ndarray,
    frac: float = 0.05,
    *,
    it: int = 0,
    delta: float = 0.0,
    min_points: int = 10,
) -> np.ndarray:
    x = _as_2d(data)
    T, F = x.shape
    out = np.full_like(x, np.nan)

    t = np.arange(T, dtype=float)
    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)
        if mask.sum() < min_points:
            continue
        try:
            sm = lowess(
                col[mask],
                t[mask],
                frac=frac,
                it=it,
                delta=delta,
                return_sorted=False,
            )
            out[mask, i] = sm
        except Exception:
            continue
    return out
