"""foreblocks.ts_handler.auto_filter.filters.tv.

Total-variation denoising (1-D Chambolle projection).

"""

from __future__ import annotations

import numpy as np
import pandas as pd

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


@register_filter("TV Denoising")
def tv_denoise(
    ts: pd.Series, weight: float = 0.35, max_iter: int = 300, eps: float = 1e-5
) -> pd.Series:
    y = ts.values.astype(float)
    n = len(y)
    if n < 3:
        return ts.copy()

    p = np.zeros(n - 1, dtype=float)
    out = y.copy()
    tau = 0.25

    for _ in range(max_iter):
        out_prev = out.copy()
        p = p + (tau / max(weight, 1e-6)) * np.diff(out)
        p /= np.maximum(1.0, np.abs(p))

        div = np.empty(n, dtype=float)
        div[0], div[1:-1], div[-1] = p[0], p[1:] - p[:-1], -p[-1]

        out = y + weight * div
        if np.linalg.norm(out - out_prev) / np.sqrt(n) < eps:
            break

    return _as_series(out, ts.index, name="tv")
