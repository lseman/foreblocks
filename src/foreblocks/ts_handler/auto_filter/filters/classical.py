"""foreblocks.ts_handler.auto_filter.filters.classical.

Classical filters: moving average, Gaussian, Savitzky-Golay, Butterworth lowpass.

"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from foreblocks.ts_handler.auto_filter.filters.utils import (
    _as_series,
    _valid_odd_window,
)
from foreblocks.ts_handler.auto_filter.registry import register_filter


def moving_average(ts: pd.Series, window: int = 7) -> pd.Series:
    window = _valid_odd_window(len(ts), window, minimum=3)
    return ts.rolling(window=window, center=True, min_periods=1).mean()


@register_filter("Gaussian")
def gaussian_filter(ts: pd.Series, sigma: float = 2.0) -> pd.Series:
    sigma = max(float(sigma), 0.1)
    values = gaussian_filter1d(ts.values.astype(float), sigma=sigma)
    return _as_series(values, ts.index, name="gaussian")


@register_filter("Savitzky-Golay")
def savgol_filter(ts: pd.Series, window: int = 11, polyorder: int = 2) -> pd.Series:
    n = len(ts)
    if n < 5:
        return ts.copy()
    window = _valid_odd_window(n, window, minimum=max(3, polyorder + 2))
    polyorder = min(polyorder, window - 1)
    values = signal.savgol_filter(ts.values, window_length=window, polyorder=polyorder)
    return _as_series(values, ts.index, name="savgol")


@register_filter("Butterworth Lowpass")
def butter_lowpass(ts: pd.Series, cutoff: float = 0.15, order: int = 3) -> pd.Series:
    cutoff = float(np.clip(cutoff, 1e-4, 0.99))
    b, a = signal.butter(order, cutoff, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))
    if len(ts) <= padlen:
        warnings.warn(
            f"butter_lowpass: series too short for filtfilt (need >{padlen}), "
            "falling back to moving average.",
            stacklevel=2,
        )
        return moving_average(ts, window=min(9, len(ts)))
    values = signal.filtfilt(b, a, ts.values)
    return _as_series(values, ts.index, name="butter")
