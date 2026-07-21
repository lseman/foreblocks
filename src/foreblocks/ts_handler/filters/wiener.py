"""foreblocks.ts_handler.filters.wiener.

Wiener filter for time-series denoising.

"""

from __future__ import annotations

import numpy as np
from scipy.signal import wiener as _wiener

from foreblocks.ts_handler.filters.utils import _as_2d, _nan_interp_1d


def wiener_filter(
    data: np.ndarray,
    mysize: int = 15,
    *,
    noise: float | None = None,
    fill_nans_for_filter: bool = True,
) -> np.ndarray:
    x = _as_2d(data)
    T, F = x.shape
    out = np.full_like(x, np.nan)

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)
        if mask.sum() == 0:
            continue

        col_work = col
        if fill_nans_for_filter and np.isnan(col_work).any():
            col_work = _nan_interp_1d(col_work)

        try:
            y = _wiener(col_work, mysize=mysize, noise=noise)
            out[mask, i] = y[mask]
        except Exception:
            out[mask, i] = col[mask]

    return out
