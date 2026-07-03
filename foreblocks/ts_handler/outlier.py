"""foreblocks.ts_handler.outlier.

This module implements the outlier pieces for its package.
It belongs to the time-series preprocessing, filtering, imputation, and analysis area of Foreblocks.
It exposes functions such as fast_mad_outlier_removal, fast_quantile_outlier_removal, fast_zscore_outlier_removal, fast_iqr_outlier_removal.
"""

# Standard Library
import warnings

# Scientific Computing and Visualization
import numpy as np
import torch
from numba import njit, prange
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from foreblocks.anomaly.tranad import (
    TranAD,
    TranADDataset,
    TranADDetector,
    create_sequences_vectorized,
)


# Optional imports
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None


def _remove_outliers_parallel(index, col, method, threshold):
    cleaned = _remove_outliers_wrapper((index, col, method, threshold))
    return cleaned


@njit
def fast_mad_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    valid = ~np.isnan(x)
    if np.sum(valid) < 5:
        return x  # not enough data

    med = np.nanmedian(x)
    deviations = np.abs(x - med)
    mad = np.nanmedian(deviations) + 1e-8

    # Optional robustness clamp
    if mad < 1e-6:
        mad = np.nanmean(deviations) + 1e-8

    # Correct modified Z-score (Iglewicz and Hoaglin)
    mod_z = 0.6745 * np.abs(x - med) / mad

    # Apply adaptive threshold (optional nonlinear taper)
    adapt_thresh = threshold + 0.5 * (np.std(mod_z[valid]) > 3.5)

    return np.where(mod_z > adapt_thresh, np.nan, x)


@njit
def fast_quantile_outlier_removal(
    x: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    return np.where((x < lower) | (x > upper), np.nan, x)


@njit(parallel=True)
def fast_zscore_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated Z-score outlier removal.
    """
    mean = np.nanmean(x)
    std = np.nanstd(x) + 1e-8
    n = x.shape[0]
    result = np.copy(x)
    for i in prange(n):
        if not np.isnan(x[i]):
            z = abs((x[i] - mean) / std)
            if z > threshold:
                result[i] = np.nan
    return result


@njit(parallel=True)
def fast_iqr_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """
    Numba-accelerated IQR outlier removal.
    """
    q1 = np.percentile(x[~np.isnan(x)], 25)
    q3 = np.percentile(x[~np.isnan(x)], 75)
    iqr = q3 - q1 + 1e-8
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    n = x.shape[0]
    result = np.copy(x)
    for i in prange(n):
        if not np.isnan(x[i]) and (x[i] < lower or x[i] > upper):
            result[i] = np.nan
    return result


def _hbos_outlier_removal(x: np.ndarray, threshold: float) -> np.ndarray:
    """Histogram-based Outlier Scoring for univariate or multivariate series."""
    valid = ~np.isnan(x)
    if np.sum(valid) < 5:
        return x

    if x.ndim == 1:
        values = x[valid]
    else:
        values = x[valid].reshape(-1)

    n = values.shape[0]
    bins = min(max(int(np.sqrt(n)), 10), 50)

    if x.ndim == 1:
        col = x[valid]
        hist, edges = np.histogram(col, bins=bins)
        hist = hist.astype(np.float64) + 1e-8
        dens = hist / np.sum(hist)
        idx = np.clip(np.searchsorted(edges, col, side="right") - 1, 0, len(hist) - 1)
        scores = np.zeros_like(x, dtype=np.float64)
        scores[valid] = -np.log(dens[idx])
    else:
        nrows, ncols = x.shape
        scores = np.zeros(nrows, dtype=np.float64)
        for j in range(ncols):
            col = x[:, j]
            valid_col = ~np.isnan(col)
            if np.sum(valid_col) < 5:
                continue
            col_values = col[valid_col]
            hist, edges = np.histogram(col_values, bins=bins)
            hist = hist.astype(np.float64) + 1e-8
            dens = hist / np.sum(hist)
            idx = np.clip(
                np.searchsorted(edges, col_values, side="right") - 1, 0, len(hist) - 1
            )
            temp_scores = -np.log(dens[idx])
            scores[valid_col] += temp_scores

    threshold_value = np.percentile(scores[valid], 100.0 * (1.0 - float(threshold)))
    result = x.copy()
    if x.ndim == 1:
        result[valid & (scores > threshold_value)] = np.nan
    else:
        result[scores > threshold_value] = np.nan
    return result


def _remove_outliers(
    data_col: np.ndarray, method: str, threshold: float, **kwargs
) -> np.ndarray:
    """
    Remove outliers from a univariate or multivariate time series using the specified method.
    Replaces detected outliers with np.nan.

    Parameters:
        data_col: np.ndarray of shape (T,) or (T, D)
        method: One of ["zscore", "iqr", "mad", "hbos", "quantile", "isolation_forest", "lof", "ecod", "tranad"]
        threshold: method-dependent threshold (e.g. contamination fraction for HBOS and ECOD, 0.95 for percentile methods)
        **kwargs: Optional method-specific config (e.g. seq_len, epochs for tranad)

    Returns:
        np.ndarray of same shape as input, with outliers replaced by np.nan
    """
    data_col = np.asarray(data_col)
    is_multivariate = data_col.ndim == 2
    x = data_col.copy().astype(np.float64)

    if x.size == 0 or np.isnan(x).all():
        return x

    def mask_to_nan(mask: np.ndarray) -> np.ndarray:
        if is_multivariate:
            return np.where(mask[:, None], np.nan, x)
        else:
            return np.where(mask, np.nan, x)

    # === Univariate-only methods ===
    if not is_multivariate:
        if method == "zscore":
            return fast_zscore_outlier_removal(x, threshold)
        elif method == "iqr":
            return fast_iqr_outlier_removal(x, threshold)
        elif method == "mad":
            return fast_mad_outlier_removal(x, threshold)
        elif method == "hbos":
            return _hbos_outlier_removal(x, threshold)
        elif method == "quantile":
            q1, q3 = np.nanpercentile(x, [threshold * 100, 100 - threshold * 100])
            return fast_quantile_outlier_removal(x, q1, q3)

    # === Multivariate-aware methods ===
    if method == "hbos":
        return _hbos_outlier_removal(x, threshold)
    if method == "isolation_forest":
        model = IsolationForest(contamination=threshold, random_state=42)
        pred = model.fit_predict(x if is_multivariate else x.reshape(-1, 1))
        return mask_to_nan(pred != 1)

    elif method == "lof":
        model = LocalOutlierFactor(n_neighbors=20, contamination=threshold)
        pred = model.fit_predict(x if is_multivariate else x.reshape(-1, 1))
        return mask_to_nan(pred != 1)

    elif method == "ecod":
        try:
            from pyod.models.ecod import ECOD

            model = ECOD()
            pred = model.fit(x if is_multivariate else x.reshape(-1, 1)).predict(
                x if is_multivariate else x.reshape(-1, 1)
            )
            return mask_to_nan(pred == 1)
        except ImportError:
            warnings.warn("pyod not installed. Falling back to IQR.")
            if not is_multivariate:
                Q1, Q3 = np.percentile(x, [25, 75])
                IQR = Q3 - Q1 + 1e-8
                return mask_to_nan(
                    (x < Q1 - threshold * IQR) | (x > Q3 + threshold * IQR)
                )
            else:
                raise ValueError("ECOD fallback does not support multivariate input.")

    elif method == "tranad":
        seq_len = kwargs.get("seq_len", 24)
        epochs = kwargs.get("epochs", 10)
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        adaptive = kwargs.get("adaptive", True)
        min_z = kwargs.get(
            "z_threshold", float(threshold) if float(threshold) > 1.0 else 3.0
        )

        # Ensure 2D format
        if data_col.ndim == 1:
            data_col = data_col.reshape(-1, 1)
        x = data_col.astype(np.float64)
        T, D = x.shape

        if T < seq_len + 5:
            return x if D > 1 else x.flatten()

        detector = TranADDetector(
            seq_len=seq_len,
            epochs=epochs,
            device=device,
            scaler_type=kwargs.get("scaler_type", "minmax"),
        )
        scores = detector.fit_predict(x)  # shape (T - seq_len + 1, D)
        if scores.ndim == 1:
            scores = scores[:, None]

        anomaly_mask = np.full(T, False)
        if adaptive:
            score_mean = np.mean(scores, axis=0, keepdims=True)
            score_std = np.std(scores, axis=0, keepdims=True) + 1e-8
            score_z = (scores - score_mean) / score_std
            anomaly_mask[seq_len - 1 : seq_len - 1 + len(score_z)] = np.any(
                score_z > float(min_z), axis=1
            )
        else:
            if threshold > 1.0:
                cutoff = np.mean(scores, axis=0, keepdims=True) + float(threshold) * (
                    np.std(scores, axis=0, keepdims=True) + 1e-8
                )
            else:
                percentile = float(np.clip(threshold * 100.0, 0.0, 100.0))
                cutoff = np.nanpercentile(scores, percentile, axis=0, keepdims=True)
            anomaly_mask[seq_len - 1 : seq_len - 1 + len(scores)] = np.any(
                scores > cutoff, axis=1
            )

        x_cleaned = x.copy()
        x_cleaned[anomaly_mask] = np.nan

        return x_cleaned if D > 1 else x_cleaned.flatten()

    else:
        raise ValueError(f"Unsupported outlier method: {method}")


def _remove_outliers_wrapper(args):
    """Wrapper function for parallel outlier removal."""
    i, col, method, threshold = args
    cleaned = _remove_outliers(col, method, threshold)
    return i, cleaned
