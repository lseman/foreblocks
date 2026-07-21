"""foreblocks.ts_handler.filters.stl.

STL (Seasonal and Trend decomposition using Loess) filter.

"""

from __future__ import annotations

import numpy as np

from foreblocks.ts_handler.filters.utils import _as_2d, _nan_interp_1d, _odd_at_least

# Optional import
try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    STL = None


def stl_filter(
    data: np.ndarray,
    period: int,
    *,
    robust: bool = True,
    seasonal: int = 7,
    trend: int | None = None,
    return_component: str = "trend_seasonal",  # "trend", "seasonal", or "trend_seasonal"
    fill_nans_for_filter: bool = True,
) -> np.ndarray:
    if STL is None:
        raise ImportError("statsmodels is required for STL")

    x = _as_2d(data)
    T, F = x.shape
    out = np.full_like(x, np.nan)

    # STL requires odd seasonal window >= 7
    seasonal_w = max(7, _odd_at_least(seasonal, 7))

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)

        # STL needs at least two periods
        if mask.sum() < 2 * period:
            out[mask, i] = col[mask]
            continue

        col_work = col.copy()
        if fill_nans_for_filter and np.isnan(col_work).any():
            col_work = _nan_interp_1d(col_work)

        try:
            stl_model = STL(
                col_work, period=period, seasonal=seasonal_w, trend=trend, robust=robust
            )
            res = stl_model.fit()

            if return_component == "trend":
                filtered = res.trend
            elif return_component == "seasonal":
                filtered = res.seasonal
            else:
                filtered = res.trend + res.seasonal

            out[mask, i] = filtered[mask]
        except Exception:
            out[mask, i] = col[mask]

    return out
