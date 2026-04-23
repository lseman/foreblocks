from __future__ import annotations

import numpy as np
from joblib import Parallel
from joblib import delayed
from scipy.signal import savgol_filter
from scipy.signal import wiener as _wiener
from statsmodels.nonparametric.smoothers_lowess import lowess


# =============================================================================
# filters.py (modernized / SOTA)
# - keeps same public API:
#     adaptive_savgol_filter, kalman_filter, lowess_filter, wiener_filter, emd_filter
# - newly added SOTA methods:
#     ssa_filter, stl_filter
# - more robust NaN handling, edge cases, and numeric safety
# - deterministic + configurable parallel backend
# - adds optional robust pre-centering for SavGol (improves stability)
# - adds safer Wiener behavior (preserves NaNs, uses mysize)
# - EMD: optional parallel + robust IMF selection
# =============================================================================


# Optional imports
try:
    from statsmodels.tsa.seasonal import STL
except Exception:
    STL = None
try:
    from pykalman import KalmanFilter  # type: ignore
except Exception:
    KalmanFilter = None

try:
    from PyEMD import EMD  # type: ignore
except Exception:
    EMD = None


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------
def _as_2d(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"data must be 2D [T,F], got shape={x.shape}")
    return x


def _odd_at_least(n: int, min_odd: int) -> int:
    n = int(max(n, min_odd))
    if n % 2 == 0:
        n += 1
    return n


def _nan_interp_1d(x: np.ndarray) -> np.ndarray:
    """Linear interpolate NaNs (1D). If too few points, returns copy."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_nan_interp_1d expects 1D")
    if not np.isnan(x).any():
        return x.copy()

    idx = np.arange(x.size)
    mask = ~np.isnan(x)
    if mask.sum() < 2:
        return x.copy()

    out = x.copy()
    out[~mask] = np.interp(idx[~mask], idx[mask], x[mask])
    return out


def _mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)) + 1e-12)


# =============================================================================
# SavGol (adaptive, parallel)
# =============================================================================
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
        y = savgol_filter(x_centered, window_length=w, polyorder=polyorder, mode="interp")
        y = y + c
        out[mask] = y[mask]  # preserve NaN positions
    except Exception:
        # safe fallback: keep NaNs
        return i, out

    return i, out


# =============================================================================
# Kalman filter
# =============================================================================
def kalman_filter(
    data: np.ndarray,
    *,
    n_iter: int = 5,
    min_points: int = 10,
    em_on_valid_only: bool = True,
) -> np.ndarray:
    """
    Apply a per-feature Kalman smoother (pykalman).

    Improvements:
      - preserves NaNs
      - fits parameters via EM on valid points (default)
      - configurable iterations and min points
    """
    if KalmanFilter is None:
        raise ImportError("pykalman not installed")

    x = _as_2d(data)
    T, F = x.shape
    out = x.copy()

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)
        if mask.sum() < min_points:
            continue

        kf = KalmanFilter(initial_state_mean=0.0, n_dim_obs=1)
        try:
            obs = col[mask]
            if em_on_valid_only:
                kf = kf.em(obs, n_iter=n_iter)
                smoothed, _ = kf.smooth(obs)
                out[mask, i] = smoothed.reshape(-1)
            else:
                # if you want to run on full length, fill NaNs temporarily
                filled = _nan_interp_1d(col)
                kf = kf.em(filled, n_iter=n_iter)
                smoothed, _ = kf.smooth(filled)
                out[:, i] = smoothed.reshape(-1)
                out[~mask, i] = np.nan
        except Exception:
            continue

    return out


# =============================================================================
# LOWESS
# =============================================================================
def lowess_filter(
    data: np.ndarray,
    frac: float = 0.05,
    *,
    it: int = 0,
    delta: float = 0.0,
    min_points: int = 10,
) -> np.ndarray:
    """
    Apply LOWESS smoother to each feature.

    Improvements:
      - preserves NaNs
      - exposes robust iterations (it) + delta
    """
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


# =============================================================================
# Wiener
# =============================================================================
def wiener_filter(
    data: np.ndarray,
    mysize: int = 15,
    *,
    noise: float | None = None,
    fill_nans_for_filter: bool = True,
) -> np.ndarray:
    """
    Apply Wiener filter column-wise.

    Improvements:
      - uses mysize parameter (previous version ignored it)
      - preserves NaNs by default (fills, filters, restores)
      - optionally passes noise
    """
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


# =============================================================================
# EMD
# =============================================================================
def emd_filter(
    data: np.ndarray,
    keep_ratio: float = 0.5,
    *,
    n_jobs: int = 1,
    backend: str = "loky",
    min_imfs: int = 1,
    min_points: int = 32,
) -> np.ndarray:
    """
    Empirical Mode Decomposition filter.

    Behavior:
      - skips columns with NaNs (same as your original)
      - keeps a fraction of IMFs (low-index IMFs tend to be higher frequency in PyEMD)
      - optional parallelism across features

    NOTE:
      IMFs ordering can vary by implementation; keeping "first k" is a heuristic.
      If you want "denoise", often you keep *higher* index IMFs (trend-like). Your
      original kept first k; we keep that default for backward compatibility.
    """
    if EMD is None:
        raise ImportError("PyEMD not installed")

    x = _as_2d(data)
    T, F = x.shape
    out = x.copy()

    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))

    def _emd_one(j: int) -> tuple[int, np.ndarray]:
        col = x[:, j]
        if np.isnan(col).any() or col.size < min_points:
            return j, col

        try:
            imfs = EMD().emd(col)
            if imfs is None or len(imfs) == 0:
                return j, col

            k = int(np.floor(len(imfs) * keep_ratio))
            k = int(np.clip(k, min_imfs, len(imfs)))
            rec = np.sum(imfs[:k], axis=0)
            return j, rec
        except Exception:
            return j, col

    if n_jobs == 1:
        cols = [_emd_one(j) for j in range(F)]
    else:
        cols = Parallel(n_jobs=n_jobs, backend=backend, prefer="processes")(
            delayed(_emd_one)(j) for j in range(F)
        )

    cols.sort(key=lambda t: t[0])
    for j, colf in cols:
        out[:, j] = colf

    return out


# =============================================================================
# SSA (Singular Spectrum Analysis)
# =============================================================================
def ssa_filter(
    data: np.ndarray,
    window_length: int | None = None,
    n_components: int = 2,
    *,
    fill_nans_for_filter: bool = True,
) -> np.ndarray:
    """
    SOTA Singular Spectrum Analysis (SSA) filter.
    Decomposes the time series into a trajectory matrix, performs SVD, 
    and reconstructs the signal using only the top `n_components` 
    (assumed to be trend + dominant oscillations, discarding noise).
    """
    x = _as_2d(data)
    T, F = x.shape
    out = np.full_like(x, np.nan)

    # Heuristic for embedding window length if not provided
    if window_length is None:
        window_length = min(T // 2, 50)
    
    L = int(max(2, window_length))
    K = T - L + 1
    
    if K < 1:
        # Time series too short for the window
        return x.copy()

    for i in range(F):
        col = x[:, i]
        mask = ~np.isnan(col)
        
        if mask.sum() < L + 2:
            out[mask, i] = col[mask]
            continue
            
        col_work = col.copy()
        if fill_nans_for_filter and np.isnan(col_work).any():
            col_work = _nan_interp_1d(col_work)

        # 1. Embedding
        X = np.column_stack([col_work[j : j + L] for j in range(K)])
        
        # 2. SVD
        try:
            U, s, Vh = np.linalg.svd(X, full_matrices=False)
            
            # 3. Grouping (select top components)
            d = min(n_components, len(s))
            X_elem = np.zeros_like(X)
            for j in range(d):
                X_elem += s[j] * np.outer(U[:, j], Vh[j, :])
            
            # 4. Diagonal Averaging (Reconstruction)
            rec = np.zeros(T)
            counts = np.zeros(T)
            
            for c in range(K):
                for r in range(L):
                    rec[r + c] += X_elem[r, c]
                    counts[r + c] += 1
                    
            rec = rec / np.maximum(counts, 1)
            
            out[mask, i] = rec[mask]
        except Exception:
            out[mask, i] = col[mask]

    return out


# =============================================================================
# STL (Seasonal and Trend decomposition using Loess)
# =============================================================================
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
    """
    STL (Seasonal and Trend decomposition using Loess).
    Filters the signal by isolating the Trend and/or Seasonal components,
    effectively removing the 'Residual' noise.
    """
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
                col_work, 
                period=period, 
                seasonal=seasonal_w, 
                trend=trend, 
                robust=robust
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
