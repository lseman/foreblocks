"""foreblocks.processing.auto_filter.filters.kalman_rts.

Kalman RTS smoother (1-D local-linear-trend model, pure NumPy).

"""

from __future__ import annotations

import pandas as pd

from foreblocks.processing.auto_filter.filters.utils import _as_series
from foreblocks.processing.auto_filter.registry import register_filter
from foreblocks.processing.filters._kalman import (
    _estimate_noise_variances as _estimate_noise_variances,
    _local_linear_trend_smoother,
)


@register_filter("Kalman RTS")
def kalman_rts_smoother(
    ts: pd.Series,
    q: float | None = None,
    r: float | None = None,
) -> pd.Series:
    y = ts.values.astype(float)
    N = len(y)
    if N < 4:
        return ts.copy()

    q_auto, r_auto = _estimate_noise_variances(y)
    q = float(q) if q is not None else q_auto
    r = float(r) if r is not None else r_auto

    smoothed = _local_linear_trend_smoother(y, q=q, r=r)
    return _as_series(smoothed, ts.index, name="kalman_rts")
