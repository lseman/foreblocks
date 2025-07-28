# Standard Library
from typing import Tuple

# Scientific Computing and Visualization
import numpy as np
from joblib import Parallel, delayed
from scipy.signal import savgol_filter, wiener
from statsmodels.nonparametric.smoothers_lowess import lowess

# Optional imports
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None


def adaptive_savgol_filter(
    data: np.ndarray, window: int = 15, polyorder: int = 2, n_jobs: int = -1
) -> np.ndarray:
    """
    Parallelized and numerically robust adaptive Savitzky-Golay filter.

    Args:
        data: [T, F] input time series
        window: base window size
        polyorder: polynomial order for filtering
        n_jobs: parallel jobs (default: all cores)

    Returns:
        Filtered time series of same shape as input.
    """
    T, F = data.shape
    results = Parallel(n_jobs=n_jobs)(
        delayed(_adaptive_savgol_column)(i, data[:, i], window, polyorder)
        for i in range(F)
    )

    # Reconstruct output in correct order
    results.sort(key=lambda tup: tup[0])
    return np.column_stack([col for _, col in results])


def kalman_filter(data: np.ndarray) -> np.ndarray:
    """Apply Kalman filter to the data."""
    if KalmanFilter is None:
        raise ImportError("pykalman not installed")

    filtered = np.copy(data)
    T, F = data.shape
    for i in range(F):
        x = data[:, i]
        mask = ~np.isnan(x)
        if np.sum(mask) < 10:
            continue

        kf = KalmanFilter(initial_state_mean=0, n_dim_obs=1)
        try:
            kf = kf.em(x[mask], n_iter=5)
            smoothed, _ = kf.smooth(x[mask])
            filtered[mask, i] = smoothed.flatten()
        except Exception:
            continue

    return filtered


def lowess_filter(data: np.ndarray, frac: float = 0.05) -> np.ndarray:
    """Apply LOWESS filter to the data."""
    T, F = data.shape
    filtered = np.full_like(data, np.nan)
    for i in range(F):
        x = data[:, i]
        mask = ~np.isnan(x)
        if np.sum(mask) > 10:
            smoothed = lowess(
                x[mask], np.arange(T)[mask], frac=frac, return_sorted=False
            )
            filtered[mask, i] = smoothed
    return filtered


def wiener_filter(data: np.ndarray, mysize: int = 15) -> np.ndarray:
    """Apply Wiener filter to the data."""
    return np.column_stack(
        [
            wiener(data[:, i]) if not np.isnan(data[:, i]).all() else data[:, i]
            for i in range(data.shape[1])
        ]
    )


def emd_filter(data: np.ndarray, keep_ratio: float = 0.5) -> np.ndarray:
    """Apply Empirical Mode Decomposition filter to the data."""
    if EMD is None:
        raise ImportError("PyEMD not installed")

    T, F = data.shape
    filtered = np.copy(data)
    for i in range(F):
        x = data[:, i]
        if np.isnan(x).any():
            continue
        imfs = EMD().emd(x)
        keep = int(len(imfs) * keep_ratio)
        filtered[:, i] = np.sum(imfs[:keep], axis=0)
    return filtered




def _adaptive_savgol_column(
    i: int, x: np.ndarray, base_window: int, polyorder: int
) -> Tuple[int, np.ndarray]:
    """
    Apply adaptive Savitzky-Golay smoothing to a single column.
    Returns the smoothed column with original NaN mask.
    """
    result = np.full_like(x, np.nan)
    mask = ~np.isnan(x)
    x_valid = x[mask]

    if len(x_valid) < polyorder + 2:
        return i, result

    volatility = np.std(x_valid)
    avg_magnitude = np.mean(np.abs(x_valid)) + 1e-8
    factor = np.clip(volatility / avg_magnitude, 0.5, 2.0)

    adaptive_window = int(base_window * factor)
    adaptive_window = max(polyorder + 2, adaptive_window)
    if adaptive_window % 2 == 0:
        adaptive_window += 1

    try:
        smoothed = savgol_filter(
            x_valid, window_length=adaptive_window, polyorder=polyorder
        )
        result[mask] = smoothed
    except Exception:
        pass

    return i, result
