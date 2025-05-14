# Standard Library
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
# Scientific Computing and Visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, wiener
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels as sm
# Optional imports
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None

from statsmodels.tsa.stattools import pacf


from numba import njit, prange

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


def pacf_decay(series: np.ndarray, threshold: float = 0.2, max_lag: int = 20) -> int:
    """
    Returns number of lags where PACF exceeds the threshold.
    """
    x = series[~np.isnan(series)]
    if len(x) < max_lag:
        return 0
    pacf_vals = pacf(x, nlags=max_lag, method="yw")[1:]  # 'yw' is valid
    return np.sum(np.abs(pacf_vals) > threshold)


# Set consistent matplotlib styling
def set_plot_style():
    plt.rcParams.update({
        "figure.figsize": (18, 9),
        "figure.facecolor": "white",
        "figure.dpi": 100,
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#333333",
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "grid.color": "#dddddd",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.facecolor": "white",
        "legend.edgecolor": "#cccccc",
        "legend.fontsize": 12,
        "legend.loc": "upper right",
        "lines.linewidth": 1.8,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "savefig.dpi": 150,
    })


# Filter functions
def adaptive_savgol_filter(data: np.ndarray, window: int = 15, polyorder: int = 2) -> np.ndarray:
    """Apply adaptive Savitzky-Golay filter to the data."""
    filtered = np.copy(data)
    T, F = data.shape

    for i in range(F):
        x = data[:, i]
        mask = ~np.isnan(x)
        if np.sum(mask) < polyorder + 2:
            continue

        x_valid = x[mask]
        volatility = np.std(x_valid)
        base = window
        factor = np.clip(volatility / (np.mean(np.abs(x_valid)) + 1e-8), 0.5, 2.0)
        adaptive_window = int(base * factor)
        adaptive_window = max(polyorder + 2, adaptive_window)
        if adaptive_window % 2 == 0:
            adaptive_window += 1

        try:
            filtered_values = savgol_filter(x_valid, adaptive_window, polyorder)
            filtered[mask, i] = filtered_values
        except Exception:
            continue

    return filtered


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
            smoothed = lowess(x[mask], np.arange(T)[mask], frac=frac, return_sorted=False)
            filtered[mask, i] = smoothed
    return filtered


def wiener_filter(data: np.ndarray, mysize: int = 15) -> np.ndarray:
    """Apply Wiener filter to the data."""
    return np.column_stack([
        wiener(data[:, i]) if not np.isnan(data[:, i]).all() else data[:, i]
        for i in range(data.shape[1])
    ])


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


def _remove_outliers(data_col: np.ndarray, method: str, threshold: float) -> np.ndarray:
    """
    Remove outliers from a data column using the specified method.
    Replaces detected outliers with np.nan.
    Always returns shape (N,)
    """
    x = data_col.flatten().astype(np.float64)  # ensure float

    if x.size == 0 or np.isnan(x).all():
        return x  # nothing to do

    def mask_to_nan(mask):
        return np.where(mask, np.nan, x)

    if method == "zscore":
        return fast_zscore_outlier_removal(x, threshold)

    elif method == "iqr":
        return fast_iqr_outlier_removal(x, threshold)

    elif method == "mad":
        med = np.median(x)
        mad = np.median(np.abs(x - med)) + 1e-6
        modified_z = np.abs((x - med) / mad) * 1.4826
        outlier_mask = modified_z > threshold
        return mask_to_nan(outlier_mask)

    elif method == "quantile":
        low, high = np.percentile(x, [threshold * 100, 100 - threshold * 100])
        outlier_mask = (x < low) | (x > high)
        return mask_to_nan(outlier_mask)

    elif method == "isolation_forest":
        pred = IsolationForest(contamination=threshold).fit_predict(x.reshape(-1, 1))
        outlier_mask = pred != 1
        return mask_to_nan(outlier_mask)

    elif method == "lof":
        pred = LocalOutlierFactor(n_neighbors=20, contamination=threshold).fit_predict(x.reshape(-1, 1))
        outlier_mask = pred != 1
        return mask_to_nan(outlier_mask)

    elif method == "ecod":
        try:
            from pyod.models.ecod import ECOD
            model = ECOD()
            pred = model.fit(x.reshape(-1, 1)).predict(x.reshape(-1, 1))
            outlier_mask = pred == 1
            return mask_to_nan(outlier_mask)
        except ImportError:
            warnings.warn("pyod not installed. Falling back to IQR.")
            Q1, Q3 = np.percentile(x, [25, 75])
            IQR = Q3 - Q1 + 1e-8
            outlier_mask = (x < Q1 - threshold * IQR) | (x > Q3 + threshold * IQR)
            return mask_to_nan(outlier_mask)

    raise ValueError(f"Unsupported outlier method: {method}")


def _remove_outliers_wrapper(args):
    """Wrapper function for parallel outlier removal."""
    i, col, method, threshold = args
    cleaned = _remove_outliers(col, method, threshold)
    return i, cleaned


class TimeSeriesPreprocessor:
    """
    State-of-the-art preprocessing for time series data with advanced features:
    - Automatic configuration based on data statistics
    - Log transformation for skewed data
    - Outlier removal with multiple methods
    - Empirical Wavelet Transform (EWT) for decomposition
    - Detrending and differencing for stationarity
    - Adaptive filtering with Savitzky-Golay
    - Time feature generation
    - Missing value imputation with multiple strategies
    """
    def __init__(
        self,
        normalize=True,
        differencing=False,
        detrend=False,
        apply_ewt=False,
        window_size=24,
        horizon=10,
        remove_outliers=False,
        outlier_threshold=0.05,
        outlier_method="iqr",
        impute_method="auto",
        ewt_bands=5,
        trend_imf_idx=0,
        log_transform=False,
        filter_window=5,
        filter_polyorder=2,
        apply_filter=False,
        self_tune=False,
        generate_time_features=False,
        apply_imputation=False,
    ):
        """
        Initialize the TimeSeriesPreprocessor with the specified parameters.
        
        Args:
            normalize: Whether to normalize data using StandardScaler
            differencing: Whether to apply differencing for stationarity
            detrend: Whether to remove trend component using EWT
            apply_ewt: Whether to apply Empirical Wavelet Transform
            window_size: Size of sliding window for sequence creation
            horizon: Prediction horizon length
            remove_outliers: Whether to remove outliers
            outlier_threshold: Threshold for outlier detection
            outlier_method: Method for outlier detection (iqr, zscore, mad, quantile, isolation_forest, lof, ecod)
            impute_method: Method for missing value imputation (auto, mean, interpolate, ffill, bfill, knn, iterative)
            ewt_bands: Number of frequency bands for EWT
            trend_imf_idx: Index of IMF component considered as trend
            log_transform: Whether to apply log transformation for skewed data
            filter_window: Window size for Savitzky-Golay filter
            filter_polyorder: Polynomial order for Savitzky-Golay filter
            apply_filter: Whether to apply Savitzky-Golay filter
            self_tune: Whether to automatically configure preprocessing based on data statistics
            generate_time_features: Whether to generate calendar features from timestamps
            apply_imputation: Whether to apply imputation for missing values
        """
        # Configuration parameters
        self.normalize = normalize
        self.differencing = differencing
        self.detrend = detrend
        self.apply_ewt = apply_ewt
        self.window_size = window_size
        self.horizon = horizon
        self.outlier_threshold = outlier_threshold
        self.outlier_method = outlier_method
        self.impute_method = impute_method
        self.ewt_bands = ewt_bands
        self.trend_imf_idx = trend_imf_idx
        self.log_transform = log_transform
        self.filter_window = filter_window
        self.filter_polyorder = filter_polyorder
        self.apply_filter = apply_filter
        self.remove_outliers = remove_outliers
        self.self_tune = self_tune
        self.generate_time_features = generate_time_features
        self.apply_imputation = apply_imputation
        
        # Fitted parameters (initialized as None)
        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None
        self.log_transform_flags = None
        self.filter_method = "savgol"  # Default filter method
        
        # Set matplotlib style
        set_plot_style()

    def auto_configure(self, data: np.ndarray) -> None:
        """
        Automatically configure preprocessing parameters based on data statistics.
        
        Args:
            data: Input time series data
        """
        if not self.self_tune:
            return
            
        from scipy.stats import skew, entropy

        print("\n📊 [Self-Tuning Preprocessing Configuration]")

        # Basic statistics
        mean_vals = np.nanmean(data, axis=0)
        std_vals = np.nanstd(data, axis=0)
        missing_rate = np.mean(np.isnan(data))

        # Feature quality check
        feature_coverage = np.mean(~np.isnan(data), axis=0)
        low_quality_flags = (feature_coverage < 0.5) | (std_vals < 1e-6)
        print(f"→ Feature Quality (coverage < 50% or near-zero std): {low_quality_flags.tolist()}")

        # Log transform suggestion based on skew
        skews = skew(data, nan_policy='omit')
        self.log_transform_flags = (np.abs(skews) > 1).tolist()
        print(f"→ Skewness per feature: {np.round(skews, 3)}")
        print(f"→ Log transform (per feature): {self.log_transform_flags}")
        self.log_transform = any(self.log_transform_flags)

        # Detrending via ADF test
        try:
            from statsmodels.tsa.stattools import adfuller
            pvals = [
                adfuller(data[:, i][~np.isnan(data[:, i])])[1] if np.sum(~np.isnan(data[:, i])) > 10 else 1.0
                for i in range(data.shape[1])
            ]
            self.detrend = any(p > 0.05 for p in pvals)
            print(f"→ ADF p-values: {np.round(pvals, 4)} → Detrend? {'✅' if self.detrend else '❌'}")
        except ImportError:
            print("→ ADF test skipped (statsmodels not installed)")

        # Period estimation for STL
        def estimate_period(series, min_period=2, max_period=100):
            from scipy.fft import fft
            x = series[~np.isnan(series)]
            if len(x) < max_period:
                return None
            spectrum = np.abs(fft(x - np.mean(x)))
            spectrum[:min_period] = 0
            peak = np.argmax(spectrum[min_period:max_period]) + min_period
            return peak

        try:
            from statsmodels.tsa.seasonal import STL
            periods = [estimate_period(data[:, i]) or 24 for i in range(data.shape[1])]
            seasonal_flags = []
            for i in range(data.shape[1]):
                try:
                    res = STL(data[:, i], period=periods[i], robust=True).fit()
                    seasonal_flags.append(res.seasonal.std() > 0.1 * res.trend.std())
                except Exception:
                    seasonal_flags.append(False)
            print(f"→ Estimated periods: {periods}")
            print(f"→ Seasonality flags (STL): {seasonal_flags}")
        except ImportError:
            print("→ STL skipped (statsmodels not installed)")

        # Spectral flatness for filtering
        def spectral_flatness(x):
            from scipy.fft import fft
            x = x[~np.isnan(x)]
            if len(x) < 10:
                return 1.0
            psd = np.abs(fft(x))**2
            psd = psd[1:len(psd)//2]
            return np.exp(np.mean(np.log(psd + 1e-8))) / (np.mean(psd) + 1e-8)

        # Compute average flatness across features
        flatness_scores = [spectral_flatness(data[:, i]) for i in range(data.shape[1])]
        avg_flatness = np.mean(flatness_scores)
        self.apply_filter = avg_flatness < 0.5
        print(f"→ Spectral Flatness: {np.round(flatness_scores, 3)} → Filter? {'✅' if self.apply_filter else '❌'}")

        # Compute sequence length
        T = data.shape[0]

        # Decision logic for filter method
        if avg_flatness > 0.7 and missing_rate > 0.05:
            self.filter_method = "kalman"      # robust to noise + NaNs
        elif avg_flatness < 0.5 and T > 500:
            self.filter_method = "savgol"      # clean signal, efficient
        elif avg_flatness < 0.5 and T <= 500:
            self.filter_method = "lowess"      # small series, nonlinear
        elif avg_flatness >= 0.5 and T > 1000:
            self.filter_method = "wiener"      # stationary noise, long series
        else:
            self.filter_method = "none"
            
        print(f"→ Avg spectral flatness: {avg_flatness:.3f} → Filter method: {self.filter_method}")

        autocorr_scores = [pacf_decay(data[:, i]) for i in range(data.shape[1])]
        print(f"→ PACF decay avg: {np.mean(autocorr_scores):.2f}, Missing rate: {missing_rate:.3f} → Impute: {self.impute_method or 'None'}")

        if missing_rate == 0:
            self.impute_method = None
        elif np.mean(autocorr_scores) > 3:
            self.impute_method = "interpolate"
        elif missing_rate < 0.15:
            self.impute_method = "knn"
        else:
            self.impute_method = "iterative"
        print(f"→ Autocorr decay avg: {np.mean(autocorr_scores):.2f}, Missing rate: {missing_rate:.3f} → Impute: {self.impute_method or 'None'}")

        # EWT band estimation
        band_suggestions = []
        for i in range(data.shape[1]):
            valid_data = data[:, i][~np.isnan(data[:, i])]
            if len(valid_data) > 0:
                hist, _ = np.histogram(valid_data, bins=20, density=True)
                band_suggestions.append(int(np.clip(entropy(hist) * 1.5, 2, 10)))
        if band_suggestions:
            self.ewt_bands = int(np.round(np.mean(band_suggestions)))
            print(f"→ EWT bands (entropy-based): {self.ewt_bands}")

        from scipy.stats import kurtosis

        def _select_outlier_method(data: np.ndarray, skews: np.ndarray, missing_rate: float) -> str:
            T, F = data.shape
            extreme_flags = []
            multimodal_flags = []

            for i in range(F):
                col = data[:, i][~np.isnan(data[:, i])]
                if len(col) < 10:
                    continue
                # Check for extreme outliers (> 6 std deviations)
                std = np.std(col)
                extreme_flags.append(np.any(np.abs(col - np.mean(col)) > 6 * std))

                # Kurtosis: high => heavy tails
                k = kurtosis(col)
                multimodal_flags.append(k > 5)

            extreme_ratio = np.mean(extreme_flags)
            heavy_tail_ratio = np.mean(multimodal_flags)
            high_skew_ratio = np.mean(np.abs(skews) > 2.5)

            if T > 2000 and missing_rate < 0.1:
                return "ecod"
            elif extreme_ratio > 0.2 or high_skew_ratio > 0.3:
                return "zscore"
            elif heavy_tail_ratio > 0.3:
                return "mad"
            else:
                return "iqr"
        self.outlier_method = _select_outlier_method(data, skews, missing_rate)

        # Summary table
        try:
            from tabulate import tabulate
            print(tabulate([
                ["Shape", data.shape],
                ["Detrend", self.detrend],
                ["Log Transform (any)", any(self.log_transform_flags)],
                ["Imputation", self.impute_method],
                ["EWT Bands", self.ewt_bands],
                ["Filter?", self.apply_filter],
                ["Outlier Method", self.outlier_method],
                ["Filter Method", self.filter_method],
            ], headers=["Config", "Value"], tablefmt="pretty"))
        except ImportError:
            pass

        print("✅ Self-tuning configuration complete.\n")

    def fit_transform(self, data: np.ndarray, time_stamps=None, feats=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess input data and return (X, y, full_processed_data).
        
        Args:
            data: Input time series data of shape [samples, features]
            time_stamps: Optional timestamps for the data
            feats: Optional subset of features to use for target
            
        Returns:
            X: Input sequences of shape [num_sequences, window_size, features]
            y: Target sequences of shape [num_sequences, horizon, target_features]
            processed: Full processed data of shape [samples, features]
        """
        # If processed is tensor, convert to numpy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        processed = data.copy()
        
        # Auto-configure preprocessing parameters if requested
        self.auto_configure(processed)
                
        # Impute missing values if needed
        if self.apply_imputation:
            processed = self._impute_missing(processed)
            self._plot_comparison(data, processed, "After Imputation", time_stamps)
            
        # Check if processed contains nan
        if np.any(np.isnan(processed)):
            raise ValueError("Imputation failed, data still contains NaN values.")

        # Apply log transform if needed
        if hasattr(self, 'log_transform_flags') and any(self.log_transform_flags):
            self.log_offset = np.zeros(processed.shape[1], dtype=np.float64)

            for i, flag in enumerate(self.log_transform_flags):
                if flag:
                    col = processed[:, i]
                    min_val = np.nanmin(col)
                    offset = np.abs(min_val) + 1.0 if min_val <= 0 else 0.0
                    self.log_offset[i] = offset

                    # Apply log transform safely with offset
                    processed[:, i] = np.log(col + offset)

        # Remove outliers if requested
        if self.remove_outliers:
            n_features = processed.shape[1]
            method = self.outlier_method
            threshold = self.outlier_threshold

            args_list = [(i, processed[:, i], method, threshold) for i in range(n_features)]
            cleaned_cols = [None] * n_features

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(_remove_outliers_wrapper, args) for args in args_list]
                for future in as_completed(futures):
                    i, cleaned = future.result()
                    if cleaned.shape != processed[:, i].shape:
                        raise ValueError(f"Outlier method returned shape {cleaned.shape} but expected {processed[:, i].shape} for feature {i}")
                    cleaned_cols[i] = cleaned

            cleaned = np.stack(cleaned_cols, axis=1)
            self._plot_comparison(processed, cleaned, "After Outlier Removal", time_stamps)
            processed = cleaned

            
        # Check if processed contains nan after outlier removal
        if np.any(np.isnan(processed)):
            # call impute_missing again
            processed = self._impute_missing(processed)
        
        # Apply EWT and detrending if needed
        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)
        
        # Apply filtering if needed
        if self.apply_filter:
            filtered = self._apply_filter(processed, method=self.filter_method)
            self._plot_comparison(processed, filtered, f"After {self.filter_method.capitalize()} Filtering", time_stamps)
            processed = filtered
        
        # Apply differencing if needed
        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0
        
        # Normalize if needed
        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)
        
        # Generate time features if requested
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)
        
        # Create sequences and return
        X, y = self._create_sequences(processed, feats)
        return X, y, processed

    def transform(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        """
        Apply the same transformation as in fit_transform but without fitting.
        
        Args:
            data: Input time series data
            time_stamps: Optional timestamps for the data
            
        Returns:
            X: Input sequences
        """
        if self.normalize and self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_transform first.")
        
        processed = data.copy()
        
        # Apply log transform if needed
        if hasattr(self, 'log_transform_flags') and any(self.log_transform_flags):
            for i, flag in enumerate(self.log_transform_flags):
                if flag:
                    processed[:, i] = np.log(processed[:, i] + self.log_offset[i])
        
        # Apply EWT if needed
        if self.apply_ewt:
            if self.ewt_boundaries is None:
                raise ValueError("EWT not fitted. Call fit_transform first.")
                
            for i in range(processed.shape[1]):
                try:
                    from ewtpy import EWT1D
                    ewt, _, _ = EWT1D(processed[:, i], N=len(self.ewt_boundaries[i]), 
                                     detect="given_bounds", boundaries=self.ewt_boundaries[i])
                    
                    # Apply detrending if needed
                    if self.detrend:
                        processed[:, i] -= ewt[:, self.trend_imf_idx]
                except ImportError:
                    warnings.warn("PyEWT not installed. Skipping EWT.")
                    break
        
        # Apply filtering if needed
        if self.apply_filter:
            processed = self._apply_filter(processed, method=self.filter_method)
        
        # Apply differencing if needed
        if self.differencing:
            processed[1:] = np.diff(processed, axis=0)
            processed[0] = 0
        
        # Apply normalization if needed
        if self.normalize:
            processed = self.scaler.transform(processed)
        
        # Generate time features if requested
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)
        
        # Create sequences and return
        return self._create_sequences(processed)[0]

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predicted values to the original scale.
        
        Args:
            predictions: Predicted values
            
        Returns:
            Original-scale predictions
        """
        # Inverse normalization
        if self.normalize and self.scaler is not None:
            predictions = self.scaler.inverse_transform(predictions)
        
        # Inverse differencing
        if self.differencing and self.diff_values is not None:
            last_value = self.diff_values[-1]
            result = np.zeros_like(predictions)
            for i in range(len(predictions)):
                last_value += predictions[i]
                result[i] = last_value
            predictions = result
        
        # Add trend back if detrended
        if self.detrend and self.trend_component is not None:
            n = len(predictions)
            trend_to_add = np.zeros_like(predictions)
            
            for i in range(predictions.shape[1]):
                trend = self.trend_component[:, i]
                
                if n <= len(trend):
                    trend_to_add[:, i] = trend[:n]
                else:
                    # Extrapolate trend for future points
                    look_back = min(10, len(trend))
                    slope = (trend[-1] - trend[-look_back]) / look_back
                    
                    for j in range(n):
                        if j < len(trend):
                            trend_to_add[j, i] = trend[j]
                        else:
                            trend_to_add[j, i] = trend[-1] + slope * (j - len(trend) + 1)
            
            predictions += trend_to_add
        
        # Inverse log transform
        if hasattr(self, 'log_transform_flags') and any(self.log_transform_flags) and self.log_offset is not None:
            for i, flag in enumerate(self.log_transform_flags):
                if flag and i < predictions.shape[1]:
                    predictions[:, i] = np.exp(predictions[:, i]) - self.log_offset[i]
        
        return predictions

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values in the data.
        
        Args:
            data: Input data with potential missing values
            
        Returns:
            Data with imputed values
        """
        df = pd.DataFrame(data)
        method = self.impute_method
        
        if method == "auto":
            filled = df.copy()
            for col in df.columns:
                missing_rate = df[col].isna().mean()
                if missing_rate < 0.05:
                    filled[col] = df[col].interpolate().ffill().bfill()
                elif missing_rate < 0.2:
                    filled[col] = KNNImputer(n_neighbors=3).fit_transform(df[[col]]).ravel()
                else:
                    try:
                        from fancyimpute import IterativeImputer
                        filled[col] = IterativeImputer().fit_transform(df[[col]]).ravel()
                    except ImportError:
                        filled[col] = df[col].fillna(df[col].mean())
            return filled.values
        
        if method == "mean":
            return df.fillna(df.mean()).values
            
        elif method == "interpolate":
            return df.interpolate().ffill().bfill().values
            
        elif method == "ffill":
            return df.ffill().bfill().values
            
        elif method == "bfill":
            return df.bfill().ffill().values
            
        elif method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df)
            
        elif method == "iterative":
            try:
                from fancyimpute import IterativeImputer
                return IterativeImputer().fit_transform(df)
            except ImportError:
                warnings.warn("fancyimpute not installed, using mean fallback")
                return df.fillna(df.mean()).values
        
        raise ValueError(f"Unsupported imputation method: {method}")
    
    def _apply_filter(self, data: np.ndarray, method: str = "savgol", **kwargs) -> np.ndarray:
        """
        Apply filtering to the data using the specified method.
        
        Args:
            data: Input data to filter
            method: Filter method (savgol, kalman, lowess, wiener, emd, none)
            **kwargs: Additional parameters for specific filter methods
            
        Returns:
            Filtered data
        """
        method = method.lower()
        
        if method == "savgol":
            window = kwargs.get("window", self.filter_window)
            polyorder = kwargs.get("polyorder", self.filter_polyorder)
            return adaptive_savgol_filter(data, window=window, polyorder=polyorder)
            
        elif method == "kalman":
            return kalman_filter(data)
            
        elif method == "lowess":
            frac = kwargs.get("frac", 0.05)
            return lowess_filter(data, frac=frac)
            
        elif method == "wiener":
            mysize = kwargs.get("mysize", 15)
            return wiener_filter(data, mysize=mysize)
            
        elif method == "emd":
            keep_ratio = kwargs.get("keep_ratio", 0.5)
            return emd_filter(data, keep_ratio=keep_ratio)
            
        elif method == "none":
            return data
            
        else:
            raise ValueError(f"Unknown filter method: {method}")

    def _apply_ewt_and_detrend(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        """
        Apply Empirical Wavelet Transform and detrend data.
        
        Args:
            data: Input data
            time_stamps: Optional timestamps
            
        Returns:
            Transformed data
        """
        try:
            from ewtpy import EWT1D
        except ImportError:
            warnings.warn("PyEWT not installed. Skipping EWT.")
            return data

        from statsmodels.tsa.stattools import arma_order_select_ic

        def _select_best_imf_by_aic(self, signal: np.ndarray, imfs: np.ndarray) -> int:
            """
            Select the IMF index that best explains the trend using lowest AIC on residual.
            """
            aic_scores = []
            for i in range(imfs.shape[1]):
                try:
                    residual = signal - imfs[:, i]
                    order = arma_order_select_ic(residual, ic='aic', max_ar=2, max_ma=2)['aic_min_order']
                    model = sm.tsa.ARIMA(residual, order=order).fit()
                    aic_scores.append(model.aic)
                except Exception:
                    aic_scores.append(np.inf)
            return int(np.argmin(aic_scores))

        self.ewt_components = []
        self.ewt_boundaries = []
        
        if self.detrend:
            self.trend_component = np.zeros_like(data)
            
        for i in range(data.shape[1]):
            signal = data[:, i]
            if np.isnan(signal).any():
                continue
                            
            ewt, _, bounds = EWT1D(signal, N=self.ewt_bands)
            self.ewt_components.append(ewt)
            self.ewt_boundaries.append(bounds)

            # Select best IMF for detrending
            if self.detrend:
                best_idx = _select_best_imf_by_aic(signal, ewt)
                trend = ewt[:, best_idx]
                self.trend_component[:, i] = trend
                data[:, i] = signal - trend

        return data

    def _generate_time_features(self, timestamps, freq='h') -> np.ndarray:
        """
        Generate time features from timestamps.
        
        Args:
            timestamps: Array of timestamps
            freq: Frequency of the data ('h' for hourly, etc.)
            
        Returns:
            Time features array
        """
        df = pd.DataFrame({'ts': pd.to_datetime(timestamps)})
        df['month'] = df.ts.dt.month / 12.0
        df['day'] = df.ts.dt.day / 31.0
        df['weekday'] = df.ts.dt.weekday / 6.0
        df['hour'] = df.ts.dt.hour / 23.0 if freq.lower() == 'h' else 0.0
        
        return df[['month', 'day', 'weekday', 'hour']].values.astype(np.float32)

    def _create_sequences(self, data: np.ndarray, feats=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences from the data.
        
        Args:
            data: Input data
            feats: Optional subset of features for the target
            
        Returns:
            X: Input sequences
            y: Target sequences
        """
        feats = list(range(data.shape[1])) if feats is None else feats
        X, y = [], []
        
        max_idx = len(data) - self.window_size - self.horizon + 1
        for i in range(max_idx):
            X.append(data[i:i+self.window_size])
            y.append(data[i+self.window_size:i+self.window_size+self.horizon][:, feats])
            
        return np.array(X), np.array(y)

    def _plot_comparison(self, original: np.ndarray, cleaned: np.ndarray, 
                        title: str = "Preprocessing Comparison", time_stamps=None) -> None:
        """
        Plot a comparison between original and processed data.
        
        Args:
            original: Original data
            cleaned: Processed data
            title: Plot title
            time_stamps: Optional timestamps for x-axis
        """
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)
        
        # Ensure shape is (n_samples, n_features)
        if original.shape[0] == 1:
            original = original.T
        elif original.shape[1] == 1 and original.shape[0] > 1:
            pass  # Already correct
        elif original.shape[0] != cleaned.shape[0]:
            # Try to reshape as (n_samples, n_features) if flattened
            raise ValueError(f"Original shape {original.shape} does not match cleaned shape {cleaned.shape}")

        if cleaned.shape[0] == 1:
            cleaned = cleaned.T

        if original.shape != cleaned.shape:
            raise ValueError(f"Shape mismatch after processing: original {original.shape}, cleaned {cleaned.shape}")
        
        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])
        if len(x) != original.shape[0]:
            raise ValueError(f"Length of x ({len(x)}) does not match number of samples ({original.shape[0]})")

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        for i in range(original.shape[1]):
            axs[0].plot(x, original[:, i], label=f"Feature {i}")
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}")

        axs[0].set_title("Original")
        axs[1].set_title("Cleaned")
        axs[0].legend()
        axs[1].legend()
        axs[0].grid(True)
        axs[1].grid(True)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    def get_ewt_components(self) -> Optional[List]:
        """
        Get the EWT components if EWT was applied.
        
        Returns:
            List of EWT components or None
        """
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> Optional[np.ndarray]:
        """
        Get the trend component if detrending was applied.
        
        Returns:
            Trend component array or None
        """
        return self.trend_component if self.detrend else None