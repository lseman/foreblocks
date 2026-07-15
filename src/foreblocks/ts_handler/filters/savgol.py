"""foreblocks.ts_handler.filters.savgol.

Adaptive Savitzky-Golay filter with robust NaN handling and parallel processing.

"""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from scipy.signal import savgol_filter

from foreblocks.ts_handler.filters.utils import _as_2d, _mad, _nan_interp_1d, _odd_at_least


def adaptive_savgol_filter(
    data: np.ndarray,
    window: int = 15,
    polyorder: int = 2,
    n_jobs: int = -1,
    *,
    robust_center: bool = True,
    fill_nans_for_filter: bool = True,
    backend: str = "loky",
) -> np.ndarray:
    """
    Parallelized and numerically robust adaptive Savitzky-Golay filter.

    Args:
        data: [T, F] input time series
        window: base window size (will be adapted per feature)
        polyorder: polynomial order for filtering
        n_jobs: parallel jobs (default: all cores)

    Keyword Args:
        robust_center: subtract median before filtering and add back after (stability)
        fill_nans_for_filter: interpolate NaNs before filtering, then restore NaNs
        backend: joblib backend ("loky" | "threading" | "multiprocessing")

    Returns:
        Filtered time series of same shape as input (NaN positions preserved).
    """
    x = _as_2d(data)
    T, F = x.shape

    if window <= 0:
        raise ValueError("window must be > 0")
    if polyorder < 0:
        raise ValueError("polyorder must be >= 0")

    results = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
        delayed(_adaptive_savgol_column)(
            i=i,
            x=x[:, i],
            base_window=window,
            polyorder=polyorder,
            robust_center=robust_center,
            fill_nans_for_filter=fill_nans_for_filter,
        )
        for i in range(F)
    )

    results.sort(key=lambda tup: tup[0])
    return np.column_stack([col for _, col in results])


def _adaptive_savgol_column(
    i: int,
    x: np.ndarray,
    base_window: int,
    polyorder: int,
    *,
    robust_center: bool,
    fill_nans_for_filter: bool,
) -> tuple[int, np.ndarray]:
    """
    Adaptive SavGol smoothing to a single column.
    Preserves NaN mask.
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)

    mask = ~np.isnan(x)
    if mask.sum() < max(polyorder + 2, 5):
        return i, out

    x_work = x.copy()

    # (optional) fill NaNs for the convolution-like filter, then restore NaNs
    if fill_nans_for_filter and np.isnan(x_work).any():
        x_work = _nan_interp_1d(x_work)

    # adaptive factor based on robust dispersion
    x_valid = x_work[mask]
    # robust scale to avoid being dominated by spikes
    disp = _mad(x_valid)
    mag = float(np.median(np.abs(x_valid)) + 1e-8)
    factor = float(np.clip(disp / mag, 0.5, 2.0))

    w = int(round(base_window * factor))
    w = _odd_at_least(w, min_odd=max(polyorder + 2, 5))

    # SavGol requires window <= len
    w = min(w, _odd_at_least(mask.sum(), min_odd=max(polyorder + 2, 5)))
    if w < polyorder + 2:
        return i, out

    # robust centering helps numerical stability on large-magnitude series
    if robust_center:
        c = float(np.median(x_valid))
        x_centered = x_work - c
    else:
        c = 0.0
        x_centered = x_work

    try:
        y = savgol_filter(
            x_centered, window_length=w, polyorder=polyorder, mode="interp"
        )
        y = y + c
        out[mask] = y[mask]  # preserve NaN positions
    except Exception:
        # safe fallback: keep NaNs
        return i, out

    return i, out
