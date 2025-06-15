# ============================
# Standard Library
# ============================
import warnings
from typing import List, Optional, Tuple

# ============================
# Visualization
# ============================
import matplotlib.pyplot as plt

# ============================
# External Libraries - Core
# ============================
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, kurtosis, skew

# ============================
# Scientific Computing & ML
# ============================
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf, adfuller, pacf
from tabulate import tabulate

from .pre.impute import SAITSImputer
from .pre.outlier import _remove_outliers

# ============================
# Optional Imports
# ============================
try:
    from pykalman import KalmanFilter
except ImportError:
    KalmanFilter = None

try:
    from PyEMD import EMD
except ImportError:
    EMD = None

from tqdm import tqdm

from .pre.ewt import *

# ============================
# Internal Imports
# ============================
from .pre.filters import *
from .pre.outlier import *
from .pre.outlier import _remove_outliers_parallel

FILTER_METHODS = {
    "savgol": lambda self, d, k: adaptive_savgol_filter(
        d, window=self.filter_window, polyorder=self.filter_polyorder
    ),
    "kalman": lambda self, d, k: kalman_filter(d),
    "lowess": lambda self, d, k: lowess_filter(d, frac=k.get("frac", 0.05)),
    "wiener": lambda self, d, k: wiener_filter(d, mysize=k.get("mysize", 15)),
    "emd": lambda self, d, k: emd_filter(d, keep_ratio=k.get("keep_ratio", 0.5)),
    "none": lambda self, d, k: d,
}


def apply_log_transform(
    data: np.ndarray, log_flags: List[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.array(
        [
            max(0.0, -np.nanmin(data[:, i]) + 1.0) if log_flags[i] else 0.0
            for i in range(data.shape[1])
        ]
    )
    transformed = np.column_stack(
        [
            np.log(data[:, i] + offsets[i]) if log_flags[i] else data[:, i]
            for i in range(data.shape[1])
        ]
    )
    return transformed, offsets


# Set consistent matplotlib styling
def set_plot_style():
    plt.rcParams.update(
        {
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
        }
    )


def compute_basic_stats(data):
    valid_mask = ~np.isnan(data)
    coverage = np.mean(valid_mask, axis=0)
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    skews = skew(data, nan_policy="omit")
    kurts = kurtosis(data, nan_policy="omit")
    return coverage, means, stds, skews, kurts


def detect_stationarity(data, D):
    pvals = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        try:
            pval = adfuller(clean)[1] if len(clean) > 10 else 1.0
        except Exception:
            pval = 1.0
        pvals.append(pval)
    return pvals


def detect_seasonality(data, D):
    seasonal_flags, detected_periods = [], []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 10:
            seasonal_flags.append(False)
            detected_periods.append(None)
            continue
        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
        freqs, psd = welch(norm, nperseg=min(256, len(norm)))
        peaks, _ = find_peaks(psd, height=0.1 * np.max(psd))
        if len(peaks) == 0:
            seasonal_flags.append(False)
            detected_periods.append(None)
            continue
        peak_freq = freqs[peaks[np.argmax(psd[peaks])]]
        period = int(round(1.0 / peak_freq)) if peak_freq > 0 else None
        try:
            acf_vals = acf(norm, nlags=min(100, len(norm) // 2))
            acf_peaks, _ = find_peaks(acf_vals, height=0.2)
            strength = np.max(acf_vals[acf_peaks]) if len(acf_peaks) > 0 else 0
            is_seasonal = strength > 0.3
        except Exception:
            is_seasonal = False
        seasonal_flags.append(is_seasonal)
        detected_periods.append(period if is_seasonal else None)
    return seasonal_flags, detected_periods


def analyze_signal_quality(data, D):
    flatness_scores, snr_scores = [], []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 10:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue
        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
        spec = np.abs(np.fft.rfft(norm)) ** 2
        spec = spec[1 : len(spec) // 2]
        if len(spec) == 0:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue
        flat = np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8)
        snr = np.max(spec) / (np.mean(spec) + 1e-8)
        flatness_scores.append(flat)
        snr_scores.append(snr)
    return flatness_scores, snr_scores


def score_pacf(data, D):
    pacf_scores = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 30:
            pacf_scores.append(0)
            continue
        try:
            pacf_vals = pacf(clean, nlags=min(20, len(clean) // 3), method="ols")
            score = np.sum(np.abs(pacf_vals[1:]) > 0.2)
        except Exception:
            score = 0
        pacf_scores.append(score)
    return pacf_scores


def estimate_ewt_bands(data, D):
    band_estimates = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 20:
            band_estimates.append(3)
            continue
        hist, _ = np.histogram(clean, bins=20, density=True)
        hist += 1e-10
        hist /= np.sum(hist)
        ent = entropy(hist)
        band_estimates.append(int(np.clip(ent * 2, 2, 10)))
    return band_estimates


def summarize_configuration(params):
    print(
        "\n"
        + tabulate(
            [
                ["Dataset Dimensions", params["dimensions"]],
                ["Missing Values", f"{params['missing_rate']:.2%}"],
                [
                    "Stationarity",
                    "Non-stationary" if params["detrend"] else "Stationary",
                ],
                ["Seasonality", "Present" if params["seasonal"] else "Not detected"],
                [
                    "Transformation",
                    "Log (selective)" if params["log_transform"] else "None",
                ],
                [
                    "Signal Processing",
                    params["filter_method"] if params["apply_filter"] else "None",
                ],
                ["Imputation", params["impute_method"] or "None"],
                ["Outlier Detection", params["outlier_method"]],
                ["Outlier Threshold", f"{params['outlier_threshold']:.2f}"],
                ["Decomposition", f"{params['ewt_bands']} bands"],
            ],
            headers=["Parameter", "Configuration"],
            tablefmt="pretty",
        )
    )


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
        epochs=500,
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
        self.epochs = epochs

        # Fitted parameters (initialized as None)
        self.scaler = None
        self.log_offset = None
        self.diff_values = None
        self.trend_component = None
        self.ewt_components = None
        self.ewt_boundaries = None
        self.log_transform_flags = None
        self.filter_method = "savgol"  # Default filter method

        self.available_methods = {
            "ecod": True,
            "tranad": True,
            "isolation_forest": True,
            "lof": True,
            "zscore": True,
            "mad": True,
            "quantile": True,
            "iqr": True,
        }

        # Set matplotlib style
        set_plot_style()

    def _parallel_outlier_clean(self, data: np.ndarray) -> np.ndarray:
        if self.outlier_method in {"tranad", "isolation_forest", "ecod", "lof"}:
            # Whole-matrix method — skip per-column parallelism
            cleaned = _remove_outliers(
                data,
                self.outlier_method,
                self.outlier_threshold,
                seq_len=self.horizon,
                epochs=self.epochs,
            )
            return cleaned
        else:
            # Per-column parallel strategy
            n_features = data.shape[1]
            cleaned_cols = Parallel(n_jobs=-1)(
                delayed(_remove_outliers_parallel)(
                    i, data[:, i], self.outlier_method, self.outlier_threshold
                )
                for i in tqdm(range(n_features), desc="Removing outliers")
            )
            cleaned_cols.sort(key=lambda tup: tup[0])
            return np.stack([col for _, col in cleaned_cols], axis=1)

    def _inverse_log_transform(self, data: np.ndarray) -> np.ndarray:
        for i, flag in enumerate(self.log_transform_flags):
            if flag and i < data.shape[1]:
                data[:, i] = np.exp(data[:, i]) - self.log_offset[i]
        return data

    def _should_log_transform(self, skew, kurt) -> bool:
        return np.abs(skew) > 1.0 or kurt > 5.0

    def _centered(self, data: np.ndarray, means: np.ndarray) -> np.ndarray:
        return data - means[np.newaxis, :]

    def auto_configure(self, data: np.ndarray, verbose=True) -> None:
        if not self.self_tune:
            return

        print("\n📊 [Auto-Configuration]")
        T, D = data.shape
        
        # === COMPUTE DATA CHARACTERISTICS ===
        coverage, means, stds, skews, kurts = compute_basic_stats(data)
        missing_rate = 1.0 - np.mean(coverage)
        flatness_scores, snr_scores = analyze_signal_quality(data, D)
        pacf_scores = score_pacf(data, D)
        
        # Initialize log transform flags
        self.log_transform_flags = [self._should_log_transform(sk, ku) for sk, ku in zip(skews, kurts)]
        
        # Consolidated statistics
        stats = {
            'T': T, 'D': D, 'missing_rate': missing_rate,
            'avg_flatness': np.mean(flatness_scores),
            'avg_snr': np.mean(snr_scores),
            'temporal_score': np.mean(pacf_scores),
            'means': means, 'stds': stds, 'skews': skews, 'kurts': kurts,
            'extreme_ratio': np.nanmean(np.any(np.abs(np.nan_to_num(self._centered(data, means))) > (6 * stds), axis=0)),
            'heavy_tails': np.nanmean(kurts > 5),
            'high_skew': np.nanmean(np.abs(skews) > 2.5)
        }
        
        # === DETECT ARCHETYPE AND CONFIGURE ===
        # Clean regular data
        if stats['missing_rate'] < 0.01 and stats['avg_flatness'] > 0.7:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method = 'none'
            self.apply_filter = False
            self.impute_method = 'interpolate'
            self.outlier_method = 'quantile'
            self.outlier_threshold = 3.0
            archetype = 'clean_regular'
        
        # Noisy temporal data
        elif stats['temporal_score'] > 0.8 and stats['avg_snr'] < 2.0:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method = 'savgol'
            self.apply_filter = True
            self.impute_method = 'interpolate'
            self.outlier_method = 'mad'
            self.outlier_threshold = 3.5 + 0.5 * np.mean(np.abs(stats['skews']))
            archetype = 'noisy_temporal'
        
        # Sparse irregular data
        elif stats['missing_rate'] > 0.3 and stats['temporal_score'] < 0.3:
            self.log_transform = False
            self.filter_method = 'none'
            self.apply_filter = False
            self.impute_method = 'iterative' if stats['missing_rate'] < 0.6 else 'ffill'
            self.outlier_method = 'mad'
            self.outlier_threshold = 4.0
            archetype = 'sparse_irregular'
        
        # Heavy outliers
        elif stats['extreme_ratio'] > 0.1 and stats['heavy_tails'] > 0.5:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method = 'wiener'
            self.apply_filter = True
            self.impute_method = 'mad' if stats['missing_rate'] < 0.1 else 'iterative'
            self.outlier_method = 'mad'
            self.outlier_threshold = 4.5
            archetype = 'heavy_outliers'
        
        # === HEURISTIC CONFIGURATION ===
        else:
            self.log_transform = any(self.log_transform_flags)
            
            # Filter selection
            if stats['missing_rate'] > 0.2:
                self.filter_method, self.apply_filter = 'kalman', True
            elif stats['avg_flatness'] < 0.4 and stats['T'] > 500:
                self.filter_method, self.apply_filter = 'savgol', True
            elif stats['avg_flatness'] < 0.5:
                self.filter_method, self.apply_filter = 'lowess', True
            elif stats['avg_flatness'] >= 0.5 and stats['T'] > 50:
                self.filter_method, self.apply_filter = 'wiener', True
            else:
                self.filter_method, self.apply_filter = 'none', False
            
            # Imputation selection
            if stats['missing_rate'] == 0:
                self.impute_method = 'interpolate'
            elif stats['missing_rate'] < 0.05:
                self.impute_method = 'interpolate' if stats['temporal_score'] > 0.6 else 'mean'
            elif stats['missing_rate'] < 0.15:
                self.impute_method = 'knn' if stats['temporal_score'] < 0.4 else 'interpolate'
            elif stats['missing_rate'] < 0.3:
                self.impute_method = 'iterative' if stats['temporal_score'] > 0.5 else 'knn'
            elif stats['missing_rate'] < 0.6:
                self.impute_method = 'iterative' if stats['temporal_score'] > 0.3 else 'ffill'
            else:
                self.impute_method = 'ffill' if stats['temporal_score'] > 0.6 else 'bfill'
            
            # Outlier detection selection
            if stats['heavy_tails'] > 0.3 or stats['high_skew'] > 0.3:
                self.outlier_method = 'mad'
            elif hasattr(self, 'available_methods') and self.available_methods.get('tranad', False):
                self.outlier_method = 'tranad'
            elif hasattr(self, 'available_methods') and self.available_methods.get('ecod', False):
                self.outlier_method = 'ecod'
            elif stats['T'] > 3000 and stats['missing_rate'] < 0.1:
                self.outlier_method = 'isolation_forest'
            elif stats['D'] > 5:
                self.outlier_method = 'lof'
            else:
                self.outlier_method = 'zscore'
            
            # Adaptive threshold
            base = 3.5
            skew_adj = min(1.5, 0.5 * np.mean(np.abs(stats['skews'])))
            kurt_adj = 0.2 * max(0, np.mean(stats['kurts']) - 3)
            if stats['extreme_ratio'] > 0.05:
                base += 0.5
            self.outlier_threshold = base + skew_adj + kurt_adj
            
            archetype = 'heuristic'
        
        # === STANDARD CONFIGURATIONS ===
        # Stationarity
        pvals = detect_stationarity(data, D)
        self.detrend = any(p > 0.05 for p in pvals)
        
        # Seasonality
        seasonal_flags, detected_periods = detect_seasonality(data, D)
        self.seasonal = any(seasonal_flags)
        
        # Wavelet bands
        bands = estimate_ewt_bands(data, D)
        self.ewt_bands = int(np.round(np.mean(bands)))
        
        # === PRINT SUMMARY ===
        if verbose:
            config_summary = {
                "dimensions": f"{stats['T']} × {stats['D']}",
                "missing_rate": stats['missing_rate'],  # Keep as float for formatting
                "pattern": archetype,
                "log_transform": self.log_transform,
                "filter_method": self.filter_method,
                "apply_filter": self.apply_filter,
                "impute_method": self.impute_method,
                "outlier_method": self.outlier_method,
                "outlier_threshold": self.outlier_threshold,
                "detrend": self.detrend,
                "seasonal": self.seasonal,
                "ewt_bands": self.ewt_bands
            }
            summarize_configuration(config_summary)
        
        print("✅ Configuration complete.\n")
            
    def fit_transform(
        self, data: np.ndarray, time_stamps=None, feats=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess input time series data and return preprocessed X, y, and full processed data.
        """
        # 1. Tensor → NumPy
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        processed = data.copy()

        # 2. Auto-configuration
        self.auto_configure(processed)

        # 3. Imputation
        # check if there are NaNs in the data
        if np.any(np.isnan(processed)):
            if self.apply_imputation:
                processed = self._impute_missing(processed)
                self._plot_comparison(data, processed, "After Imputation", time_stamps)

        if np.any(np.isnan(processed)):
            raise ValueError("NaNs remain after imputation.")

        # 4. Log Transformation
        if hasattr(self, "log_transform_flags") and any(self.log_transform_flags):
            processed, self.log_offset = apply_log_transform(
                processed, self.log_transform_flags
            )

        # 5. Outlier Removal (parallelized)
        if self.remove_outliers:
            # method, threshold = self.outlier_method, self.outlier_threshold
            processed = self._parallel_outlier_clean(processed)
            self._plot_comparison(data, processed, "After Outlier Removal", time_stamps)

            # Post-outlier imputation
            if np.any(np.isnan(processed)):
                processed = self._impute_missing(processed)

        # 6. EWT + Detrending
        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)

        # 7. Filtering
        if self.apply_filter:
            filtered = self._apply_filter(processed, method=self.filter_method)
            self._plot_comparison(
                processed,
                filtered,
                f"After {self.filter_method.capitalize()} Filtering",
                time_stamps,
            )
            processed = filtered

        # 8. Differencing
        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed = np.vstack(
                [np.zeros_like(processed[0]), np.diff(processed, axis=0)]
            )

        # 9. Normalization
        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)

        time_feats = None
        # 10. Timestamp features
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)

        # 11. Sequence generation
        X, y, time_f = self._create_sequences(processed, feats, time_feats)
        return X, y, processed, time_f

    def transform(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        """
        Apply the same transformation as in fit_transform but without refitting.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        processed = data.copy()

        # Log transform
        if self.log_transform and self.log_offset is not None:
            processed, _ = apply_log_transform(processed, self.log_transform_flags)

        # Filtering
        if self.apply_filter:
            processed = self._apply_filter(processed, method=self.filter_method)

        # Differencing
        if self.differencing:
            processed = np.vstack(
                [np.zeros_like(processed[0]), np.diff(processed, axis=0)]
            )

        # Detrending via EWT
        if self.apply_ewt:
            for i in range(processed.shape[1]):
                if self.ewt_boundaries and i < len(self.ewt_boundaries):
                    ewt, _, _ = EWT1D(
                        processed[:, i],
                        N=len(self.ewt_boundaries[i]),
                        detect="given_bounds",
                        boundaries=self.ewt_boundaries[i],
                    )
                    if self.detrend:
                        processed[:, i] -= ewt[:, self.trend_imf_idx]

        # Normalization
        if self.normalize and self.scaler:
            processed = self.scaler.transform(processed)

        # Timestamp features
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)
            processed = np.concatenate((processed, time_feats), axis=1)

        return self._create_sequences(processed)[0]

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predicted values to the original scale.
        """
        preds = predictions.copy()

        # Step 1: Denormalization
        if self.normalize and self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)

        # Step 2: Inverse differencing
        if self.differencing and self.diff_values is not None:
            last_value = self.diff_values[0]
            restored = np.zeros_like(preds)
            for t in range(len(preds)):
                last_value = last_value + preds[t]
                restored[t] = last_value
            preds = restored

        # Step 3: Restore trend (if detrended)
        if self.detrend and self.trend_component is not None:
            n, d = preds.shape
            trend_to_add = np.zeros_like(preds)
            for i in range(d):
                trend = self.trend_component[:, i]
                if n <= len(trend):
                    trend_to_add[:, i] = trend[:n]
                else:
                    # Linear extrapolation of trend
                    look_back = min(10, len(trend))
                    slope = (trend[-1] - trend[-look_back]) / look_back
                    for j in range(n):
                        if j < len(trend):
                            trend_to_add[j, i] = trend[j]
                        else:
                            trend_to_add[j, i] = trend[-1] + slope * (
                                j - len(trend) + 1
                            )
            preds += trend_to_add

        # Step 4: Inverse log transformation
        if (
            self.log_transform
            and self.log_offset is not None
            and hasattr(self, "log_transform_flags")
        ):
            preds = self._inverse_log_transform(preds)

        return preds

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        """
        Impute missing values in a state-of-the-art, adaptive and parallelized manner.

        Args:
            data: Input time series (T × D) with NaNs

        Returns:
            Imputed array of same shape
        """
        df = pd.DataFrame(data)
        method = self.impute_method

        if method == "saits":
            try:
                saits_model = SAITSImputer(seq_len=self.window_size, epochs=500)
                saits_model.fit(data)
                return saits_model.impute(data)
            except Exception as e:
                print(f"[ERROR] SAITS imputation failed: {e}")
                raise

        if method == "auto":
            try:
                from fancyimpute import IterativeImputer as FancyIter
            except ImportError:
                FancyIter = None

            def impute_column(i, col, window_size=24):
                series = df[col]
                missing_rate = series.isna().mean()

                try:
                    # 1. Low missing rate → Interpolation
                    if missing_rate < 0.05:
                        filled = series.interpolate(
                            method="linear", limit_direction="both"
                        )
                        return i, filled.ffill().bfill().values

                    # 2. Moderate → KNN Imputer
                    elif missing_rate < 0.2:
                        return (
                            i,
                            KNNImputer(n_neighbors=3)
                            .fit_transform(series.to_frame())
                            .ravel(),
                        )

                    # 3. High → IterativeImputer or fallback
                    elif FancyIter is not None:
                        imputed = (
                            FancyIter(max_iter=10, random_state=0)
                            .fit_transform(series.to_frame())
                            .ravel()
                        )
                        return i, imputed

                except Exception as e:
                    print(f"[WARN] Fallback imputation for column {col}: {e}")

                # Fallback: use lag/seasonality or global mean
                seasonal = series.copy()
                for t in range(len(series)):
                    if pd.isna(seasonal.iloc[t]) and t >= window_size:
                        seasonal.iloc[t] = series.iloc[t - window_size]
                return i, seasonal.fillna(series.mean()).values

            results = Parallel(n_jobs=-1)(
                delayed(impute_column)(i, col, self.window_size)
                for i, col in enumerate(
                    tqdm(df.columns, desc="🔧 Imputing Missing Values")
                )
            )

            results.sort(key=lambda x: x[0])
            imputed_matrix = np.column_stack([col for _, col in results])
            return imputed_matrix

        # === Manual strategies (non-parallel, full-matrix) ===
        if method == "mean":
            return df.fillna(df.mean()).values
        elif method == "interpolate":
            return df.interpolate(method="linear").ffill().bfill().values
        elif method == "ffill":
            return df.ffill().bfill().values
        elif method == "bfill":
            return df.bfill().ffill().values
        elif method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df)
        elif method == "iterative":
            try:
                from fancyimpute import IterativeImputer as FancyIter
            except ImportError:
                try:
                    from sklearn.experimental import enable_iterative_imputer
                    from sklearn.impute import IterativeImputer as FancyIter
                except ImportError:
                    raise ImportError("Iterative imputer not available.")
            return FancyIter(random_state=0).fit_transform(df)

        raise ValueError(f"Unsupported imputation method: {method}")

    def _apply_filter(
        self, data: np.ndarray, method: str = "savgol", **kwargs
    ) -> np.ndarray:
        if method not in FILTER_METHODS:
            raise ValueError(f"Unknown filter method: {method}")
        return FILTER_METHODS[method](self, data, kwargs)

    def _apply_ewt_and_detrend(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        """
        Apply Empirical Wavelet Transform and detrend data using parallel processing.

        Args:
            data: Input data
            time_stamps: Optional timestamps

        Returns:
            Transformed data
        """
        try:
            from pyeewt import EWT1D
        except ImportError:
            warnings.warn("PyEWT not installed. Skipping EWT.")
            return data

        # Parallelized EWT + Detrending
        output, ewt_components, ewt_boundaries, trend_components = (
            apply_ewt_and_detrend_parallel(
                data, self.ewt_bands, self.detrend, self.trend_imf_idx
            )
        )

        self.ewt_components = ewt_components
        self.ewt_boundaries = ewt_boundaries
        if self.detrend:
            self.trend_component = trend_components

        return output

    def _generate_time_features(self, timestamps, freq="h") -> np.ndarray:
        """
        Generate time features from timestamps.

        Args:
            timestamps: Array of timestamps
            freq: Frequency of the data ('h' for hourly, etc.)

        Returns:
            Time features array
        """
        df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
        df["month"] = df.ts.dt.month / 12.0
        df["day"] = df.ts.dt.day / 31.0
        df["weekday"] = df.ts.dt.weekday / 6.0
        df["hour"] = df.ts.dt.hour / 23.0 if freq.lower() == "h" else 0.0

        return df[["month", "day", "weekday", "hour"]].values.astype(np.float32)

    def _create_sequences(
        self, data: np.ndarray, feats=None, time_feats=None
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        X, y, time_f = [], [], []

        max_idx = len(data) - self.window_size - self.horizon + 1
        for i in tqdm(range(max_idx), desc="Creating sequences"):
            X.append(data[i : i + self.window_size])
            time_f.append(
                time_feats[i : i + self.window_size] if time_feats is not None else None
            )
            y.append(
                data[i + self.window_size : i + self.window_size + self.horizon][
                    :, feats
                ]
            )

        return np.array(X), np.array(y), np.array(time_f)

    def _plot_comparison(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        title: str = "Preprocessing Comparison",
        time_stamps=None,
    ) -> None:
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
            raise ValueError(
                f"Original shape {original.shape} does not match cleaned shape {cleaned.shape}"
            )

        if cleaned.shape[0] == 1:
            cleaned = cleaned.T

        if original.shape != cleaned.shape:
            raise ValueError(
                f"Shape mismatch after processing: original {original.shape}, cleaned {cleaned.shape}"
            )

        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])
        if len(x) != original.shape[0]:
            raise ValueError(
                f"Length of x ({len(x)}) does not match number of samples ({original.shape[0]})"
            )

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
