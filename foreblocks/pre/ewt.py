# ============================
# Standard Library
# ============================
import copy
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================
# External Libraries - Core
# ============================
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from numba import njit, prange

# ============================
# Scientific Computing & ML
# ============================
from scipy.signal import savgol_filter, wiener
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.stattools import pacf
import statsmodels.api as sm

# ============================
# Visualization
# ============================
import matplotlib.pyplot as plt

# --- Parallel EWT + Detrend ---
def _select_best_imf_by_aic(signal: np.ndarray, imfs: np.ndarray) -> int:
    from statsmodels.tsa.stattools import arma_order_select_ic
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
    import warnings

    best_score = np.inf
    best_idx = 0

    for i in range(imfs.shape[1]):
        residual = signal - imfs[:, i]
        residual = residual[~np.isnan(residual)]

        if len(residual) < 20 or np.std(residual) < 1e-5:
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", category=ConvergenceWarning)

                order = arma_order_select_ic(residual, ic="aic", max_ar=2, max_ma=2)[
                    "aic_min_order"
                ]
                model = sm.tsa.ARIMA(residual, order=order).fit()
                aic = model.aic

                if np.isfinite(aic) and aic < best_score:
                    best_score = aic
                    best_idx = i
        except Exception:
            continue

    return best_idx


def _ewt_detrend_column(i, signal, ewt_bands, detrend, trend_idx, auto_trend):
    from ewtpy import EWT1D
    import warnings

    if np.isnan(signal).any():
        return i, signal, None, None, None

    try:
        ewt, _, bounds = EWT1D(signal, N=ewt_bands)
        if auto_trend:
            best_idx = _select_best_imf_by_aic(signal, ewt)
        else:
            best_idx = trend_idx
        trend = ewt[:, best_idx] if detrend else None
        result = signal - trend if trend is not None else signal
        return i, result, ewt, bounds, trend
    except Exception as e:
        warnings.warn(f"EWT failed on feature {i}: {e}")
        return i, signal, None, None, None


def apply_ewt_and_detrend_parallel(
    data: np.ndarray,
    ewt_bands: int,
    detrend: bool,
    trend_idx: int,
    auto_trend: bool = True,
):
    results = Parallel(n_jobs=-1)(
        delayed(_ewt_detrend_column)(
            i, data[:, i], ewt_bands, detrend, trend_idx, auto_trend
        )
        for i in range(data.shape[1])
    )

    output = np.copy(data)
    ewt_components = [None] * data.shape[1]
    ewt_boundaries = [None] * data.shape[1]
    trend_components = np.zeros_like(data) if detrend else None

    for i, result, ewt, bounds, trend in results:
        output[:, i] = result
        ewt_components[i] = ewt
        ewt_boundaries[i] = bounds
        if detrend and trend is not None:
            trend_components[:, i] = trend

    return output, ewt_components, ewt_boundaries, trend_components

