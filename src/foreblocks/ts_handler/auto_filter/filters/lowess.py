"""foreblocks.ts_handler.auto_filter.filters.lowess.

LOWESS (locally weighted scatterplot smoothing) filters.

"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess

from foreblocks.ts_handler.auto_filter.filters.utils import _as_series
from foreblocks.ts_handler.auto_filter.registry import register_filter


@register_filter("LOWESS")
def lowess_filter(ts: pd.Series, frac: float = 0.08, it: int = 2) -> pd.Series:
    """Locally weighted regression smoother (Cleveland 1979).

    Parameters
    ----------
    frac:
        Fraction of data used in each local regression window.
        Smaller → less smoothing, more local detail.
    it:
        Number of robustifying iterations (down-weights outliers).
    """
    x = np.arange(len(ts), dtype=float)
    y = ts.values.astype(float)
    smoothed = sm_lowess(y, x, frac=frac, it=it, return_sorted=False)
    return _as_series(smoothed, ts.index, name="lowess")


@register_filter("Robust LOESS")
def robust_loess_filter(ts: pd.Series, frac: float = 0.10, it: int = 4) -> pd.Series:
    """Outlier-robust local regression (Cleveland's robust LOESS).

    Same local-regression machinery as :func:`lowess_filter` but with more
    robustifying iterations and a slightly wider default window, so bilinear
    weights down-weight outliers/spikes via Tukey's biweight. Prefer this over
    plain LOWESS when the series has heavy-tailed noise or transient spikes.
    """
    x = np.arange(len(ts), dtype=float)
    y = ts.values.astype(float)
    smoothed = sm_lowess(y, x, frac=frac, it=max(3, int(it)), return_sorted=False)
    return _as_series(smoothed, ts.index, name="robust_loess")
