# ============================
# Standard Library
# ============================
import warnings
from dataclasses import dataclass
from dataclasses import field
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

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
from joblib import Parallel
from joblib import delayed
from scipy.signal import find_peaks
from scipy.signal import welch
from scipy.stats import entropy
from scipy.stats import kurtosis
from scipy.stats import skew

# ============================
# Scientific Computing & ML
# ============================
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import pacf
from tabulate import tabulate
from tqdm import tqdm

from .pre.ewt import *
from .pre.filters import *
from .pre.impute import SAITSImputer
from .pre.outlier import *
from .pre.outlier import _remove_outliers
from .pre.outlier import _remove_outliers_parallel


# ============================
# Optional Imports
# ============================
try:
    from pykalman import KalmanFilter  # type: ignore
except Exception:
    KalmanFilter = None

try:
    from PyEMD import EMD  # type: ignore
except Exception:
    EMD = None


# ----------------------------
# Plot style (unchanged)
# ----------------------------
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


# ----------------------------
# Utilities
# ----------------------------
def apply_log_transform(
    data: np.ndarray, log_flags: List[bool]
) -> Tuple[np.ndarray, np.ndarray]:
    offsets = np.array(
        [
            max(0.0, -np.nanmin(data[:, i]) + 1.0) if log_flags[i] else 0.0
            for i in range(data.shape[1])
        ],
        dtype=float,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = np.column_stack(
            [
                np.log(data[:, i] + offsets[i]) if log_flags[i] else data[:, i]
                for i in range(data.shape[1])
            ]
        )
    return transformed, offsets


def compute_basic_stats(data: np.ndarray):
    valid_mask = ~np.isnan(data)
    coverage = np.mean(valid_mask, axis=0)
    means = np.nanmean(data, axis=0)
    stds = np.nanstd(data, axis=0)
    skews = skew(data, nan_policy="omit")
    kurts = kurtosis(data, nan_policy="omit")
    return coverage, means, stds, skews, kurts


def detect_stationarity(data: np.ndarray, D: int):
    pvals = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) <= 10:
            pvals.append(1.0)
            continue
        try:
            pvals.append(adfuller(clean, autolag="AIC")[1])
        except Exception:
            pvals.append(1.0)
    return pvals


def detect_seasonality(data: np.ndarray, D: int):
    seasonal_flags, detected_periods = [], []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 10:
            seasonal_flags.append(False)
            detected_periods.append(None)
            continue
        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
        try:
            # PSD
            nperseg = min(256, max(16, len(norm)))
            freqs, psd = welch(norm, nperseg=nperseg)
            if not np.any(psd > 0):
                seasonal_flags.append(False)
                detected_periods.append(None)
                continue
            peaks, _ = find_peaks(psd, height=0.1 * np.max(psd))
            if len(peaks) == 0:
                seasonal_flags.append(False)
                detected_periods.append(None)
                continue
            peak_freq = freqs[peaks[np.argmax(psd[peaks])]]
            period = int(round(1.0 / peak_freq)) if peak_freq > 0 else None

            # ACF corroboration
            acf_vals = acf(norm, nlags=min(100, len(norm) // 2), fft=True)
            acf_peaks, _ = find_peaks(acf_vals, height=0.2)
            strength = np.max(acf_vals[acf_peaks]) if len(acf_peaks) > 0 else 0.0
            is_seasonal = strength > 0.3
        except Exception:
            is_seasonal, period = False, None

        seasonal_flags.append(is_seasonal)
        detected_periods.append(period if is_seasonal else None)
    return seasonal_flags, detected_periods


def analyze_signal_quality(data: np.ndarray, D: int):
    flatness_scores, snr_scores = [], []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 10:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue
        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)
        spec = np.abs(np.fft.rfft(norm)) ** 2
        spec = spec[1 : max(2, len(spec) // 2)]
        if len(spec) == 0:
            flatness_scores.append(1.0)
            snr_scores.append(0.0)
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            flat = np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8)
        snr = np.max(spec) / (np.mean(spec) + 1e-8)
        flatness_scores.append(float(flat))
        snr_scores.append(float(snr))
    return flatness_scores, snr_scores


def score_pacf(data: np.ndarray, D: int):
    scores = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 30:
            scores.append(0)
            continue
        try:
            pacf_vals = pacf(clean, nlags=min(20, len(clean) // 3), method="ywm")
            scores.append(int(np.sum(np.abs(pacf_vals[1:]) > 0.2)))
        except Exception:
            scores.append(0)
    return scores


def estimate_ewt_bands(data: np.ndarray, D: int):
    band_estimates = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 20:
            band_estimates.append(3)
            continue
        hist, _ = np.histogram(clean, bins=20, density=True)
        hist = np.maximum(hist, 1e-10)
        hist /= np.sum(hist)
        ent = float(entropy(hist))
        band_estimates.append(int(np.clip(ent * 2, 2, 10)))
    return band_estimates


def summarize_configuration(params: Dict[str, Any]):
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


# ----------------------------
# Filter dispatch
# ----------------------------
def _dispatch_filter(self_ref, data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Centralized filter dispatcher. All methods receive (self_ref, data, **kwargs).
    """
    if method == "savgol":
        return adaptive_savgol_filter(
            data, window=self_ref.filter_window, polyorder=self_ref.filter_polyorder
        )
    elif method == "kalman":
        if KalmanFilter is None:
            warnings.warn("KalmanFilter not available; returning input unchanged.")
            return data
        return kalman_filter(data, **kwargs)
    elif method == "lowess":
        return lowess_filter(data, frac=kwargs.get("frac", 0.05))
    elif method == "wiener":
        return wiener_filter(data, mysize=kwargs.get("mysize", 15))
    elif method == "emd":
        if EMD is None:
            warnings.warn("PyEMD not available; returning input unchanged.")
            return data
        return emd_filter(data, keep_ratio=kwargs.get("keep_ratio", 0.5))
    elif method == "none":
        return data
    else:
        raise ValueError(f"Unknown filter method: {method}")


# ===================================================
#                PREPROCESSOR CLASS
# ===================================================
@dataclass
class TimeSeriesPreprocessor:
    """
    State-of-the-art preprocessing for time series data with advanced features.
    """

    # Core behavior
    normalize: bool = True
    differencing: bool = False
    detrend: bool = False
    apply_ewt: bool = False

    # Windowing
    window_size: int = 24
    horizon: int = 10

    # Outliers
    remove_outliers: bool = False
    outlier_threshold: float = 0.05
    outlier_method: str = "iqr"

    # Imputation
    impute_method: str = "auto"
    apply_imputation: bool = False
    epochs: int = 500

    # EWT
    ewt_bands: int = 5
    trend_imf_idx: int = 0

    # Transforms & filters
    log_transform: bool = False
    filter_window: int = 5
    filter_polyorder: int = 2
    apply_filter: bool = False
    self_tune: bool = False
    generate_time_features: bool = False

    # ----- fitted/learned attributes -----
    scaler: Optional[StandardScaler] = field(default=None, init=False)
    log_offset: Optional[np.ndarray] = field(default=None, init=False)
    diff_values: Optional[np.ndarray] = field(default=None, init=False)
    trend_component: Optional[np.ndarray] = field(default=None, init=False)
    ewt_components: Optional[List] = field(default=None, init=False)
    ewt_boundaries: Optional[List] = field(default=None, init=False)
    log_transform_flags: Optional[List[bool]] = field(default=None, init=False)
    filter_method: str = field(default="savgol", init=False)
    seasonal: bool = field(default=False, init=False)

    # Optional method availability map (kept from your original)
    available_methods: Dict[str, bool] = field(
        default_factory=lambda: {
            "ecod": True,
            "tranad": True,
            "isolation_forest": True,
            "lof": True,
            "zscore": True,
            "mad": True,
            "quantile": True,
            "iqr": True,
        },
        init=False,
    )

    def __post_init__(self):
        set_plot_style()

    # ---------- helpers ----------
    @staticmethod
    def _should_log_transform(sk: float, ku: float) -> bool:
        return (abs(sk) > 1.0) or (ku > 5.0)

    @staticmethod
    def _centered(data: np.ndarray, means: np.ndarray) -> np.ndarray:
        return data - means[np.newaxis, :]

    def _ensure_log_flags(self, data: np.ndarray):
        if self.log_transform_flags is None:
            # derive from current data if not computed by self_tune
            _, _, _, skews, kurts = compute_basic_stats(data)
            self.log_transform_flags = [
                self._should_log_transform(sk, ku) for sk, ku in zip(skews, kurts)
            ]
            # obey top-level switch if user forced False
            if not self.log_transform:
                self.log_transform_flags = [False] * data.shape[1]

    # ---------- configuration ----------
    def auto_configure(self, data: np.ndarray, verbose: bool = True) -> None:
        if not self.self_tune:
            # still provide default flags so later stages are consistent
            self._ensure_log_flags(data)
            return

        print("\n[Auto-Configuration]")
        T, D = data.shape

        coverage, means, stds, skews, kurts = compute_basic_stats(data)
        missing_rate = 1.0 - float(np.mean(coverage))
        flatness_scores, snr_scores = analyze_signal_quality(data, D)
        pacf_scores = score_pacf(data, D)

        self.log_transform_flags = [
            self._should_log_transform(sk, ku) for sk, ku in zip(skews, kurts)
        ]

        stats = {
            "T": T,
            "D": D,
            "means": means,
            "stds": stds,
            "skews": skews,
            "kurts": kurts,
            "coverage": coverage,
            "missing_rate": missing_rate,
            "avg_flatness": float(np.mean(flatness_scores)),
            "avg_snr": float(np.mean(snr_scores)),
            "temporal_score": float(np.mean(pacf_scores)),
            "extreme_ratio": float(
                np.nanmean(
                    np.any(
                        np.abs(np.nan_to_num(self._centered(data, means))) > (6 * stds),
                        axis=0,
                    )
                )
            ),
            "heavy_tails": float(np.nanmean(kurts > 5)),
            "high_skew": float(np.nanmean(np.abs(skews) > 2.5)),
        }

        # Archetypes
        if stats["missing_rate"] < 0.01 and stats["avg_flatness"] > 0.75:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "interpolate"
            self.outlier_method, self.outlier_threshold = "quantile", 3.0
            archetype = "clean_regular"

        elif stats["temporal_score"] > 0.75 and stats["avg_snr"] < 1.8:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method, self.apply_filter = "savgol", True
            self.impute_method = "interpolate"
            self.outlier_method = "mad"
            self.outlier_threshold = 3.0 + np.clip(
                np.mean(np.abs(stats["skews"])), 0.0, 1.0
            )
            archetype = "noisy_temporal"

        elif stats["missing_rate"] > 0.35 and stats["temporal_score"] < 0.3:
            self.log_transform = False
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "iterative" if stats["missing_rate"] < 0.6 else "ffill"
            self.outlier_method, self.outlier_threshold = "mad", 4.0
            archetype = "sparse_irregular"

        elif stats["extreme_ratio"] > 0.08 and stats["heavy_tails"] > 0.4:
            self.log_transform = any(self.log_transform_flags)
            self.filter_method, self.apply_filter = "wiener", True
            self.impute_method = "mad" if stats["missing_rate"] < 0.1 else "iterative"
            self.outlier_method, self.outlier_threshold = "mad", 4.5
            archetype = "heavy_outliers"

        else:
            # Heuristic selection
            self.log_transform = any(self.log_transform_flags)

            if stats["missing_rate"] > 0.2:
                self.filter_method, self.apply_filter = "kalman", True
            elif stats["avg_flatness"] < 0.4 and stats["T"] > 500:
                self.filter_method, self.apply_filter = "savgol", True
            elif stats["avg_flatness"] < 0.5:
                self.filter_method, self.apply_filter = "lowess", True
            elif stats["avg_flatness"] >= 0.5 and stats["T"] > 50:
                self.filter_method, self.apply_filter = "wiener", True
            else:
                self.filter_method, self.apply_filter = "none", False

            if stats["missing_rate"] == 0:
                self.impute_method = "interpolate"
            elif stats["missing_rate"] < 0.05:
                self.impute_method = (
                    "interpolate" if stats["temporal_score"] > 0.5 else "mean"
                )
            elif stats["missing_rate"] < 0.15:
                self.impute_method = (
                    "knn" if stats["temporal_score"] < 0.3 else "interpolate"
                )
            elif stats["missing_rate"] < 0.3:
                self.impute_method = (
                    "saits" if stats["temporal_score"] > 0.3 else "iterative"
                )
            elif stats["missing_rate"] < 0.6:
                self.impute_method = "saits"
            else:
                self.impute_method = (
                    "saits" if stats["temporal_score"] > 0.5 else "ffill"
                )

            if stats["heavy_tails"] > 0.3 or stats["high_skew"] > 0.3:
                self.outlier_method = "mad"
            elif self.available_methods.get("tranad", False):
                self.outlier_method = "tranad"
            elif self.available_methods.get("ecod", False):
                self.outlier_method = "ecod"
            elif stats["T"] > 3000 and stats["missing_rate"] < 0.1:
                self.outlier_method = "isolation_forest"
            elif stats["D"] > 5:
                self.outlier_method = "lof"
            else:
                self.outlier_method = "zscore"

            base = 3.5
            skew_adj = np.clip(0.5 * np.mean(np.abs(stats["skews"])), 0, 1.5)
            kurt_adj = 0.2 * max(0, np.mean(stats["kurts"]) - 3)
            base += 0.5 if stats["extreme_ratio"] > 0.05 else 0
            self.outlier_threshold = float(base + skew_adj + kurt_adj)
            archetype = "heuristic"

        # Stationarity/seasonality/ewt
        self.detrend = any(p > 0.05 for p in detect_stationarity(data, D))
        self.seasonal = any(detect_seasonality(data, D)[0])
        self.ewt_bands = int(np.round(np.mean(estimate_ewt_bands(data, D))))

        if verbose:
            summarize_configuration(
                {
                    "dimensions": f"{T} × {D}",
                    "missing_rate": stats["missing_rate"],
                    "pattern": archetype,
                    "log_transform": self.log_transform,
                    "filter_method": self.filter_method,
                    "apply_filter": self.apply_filter,
                    "impute_method": self.impute_method,
                    "outlier_method": self.outlier_method,
                    "outlier_threshold": self.outlier_threshold,
                    "detrend": self.detrend,
                    "seasonal": self.seasonal,
                    "ewt_bands": self.ewt_bands,
                }
            )
        print("✅ Configuration complete.\n")

    # ---------- core pipeline ----------
    def fit_transform(
        self,
        data: np.ndarray,
        time_stamps: Optional[np.ndarray] = None,
        feats: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Preprocess input time series data and return:
            X, y, processed, time_features
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        processed = np.array(data, dtype=float, copy=True)

        # Configure
        self.auto_configure(processed)
        self._ensure_log_flags(processed)

        # Imputation (only if requested and NaNs present)
        if np.any(np.isnan(processed)) and self.apply_imputation:
            processed = self._impute_missing(processed)
            self._plot_comparison(data, processed, "After Imputation", time_stamps)
        if np.any(np.isnan(processed)):
            raise ValueError("NaNs remain after imputation.")

        # Log transform
        if any(self.log_transform_flags or []):
            processed, self.log_offset = apply_log_transform(
                processed, self.log_transform_flags
            )

        # Outliers
        if self.remove_outliers:
            processed = self._parallel_outlier_clean(processed)
            self._plot_comparison(data, processed, "After Outlier Removal", time_stamps)
            if np.any(np.isnan(processed)):
                processed = self._impute_missing(processed)

        # EWT + detrend
        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed, time_stamps)

        # Filtering
        if self.apply_filter:
            processed = self._apply_filter(processed, method=self.filter_method)
            self._plot_comparison(
                data,
                processed,
                f"After {self.filter_method.capitalize()} Filtering",
                time_stamps,
            )

        # Differencing
        if self.differencing:
            self.diff_values = processed[0:1].copy()
            processed = np.vstack(
                [np.zeros_like(processed[0]), np.diff(processed, axis=0)]
            )

        # Normalization (last, before windowing)
        if self.normalize:
            self.scaler = StandardScaler()
            processed = self.scaler.fit_transform(processed)

        # Time features
        time_feats = None
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)

        # Windowing
        X, y, time_f = self._create_sequences(processed, feats, time_feats)
        return X, y, processed, time_f

    def transform(
        self, data: np.ndarray, time_stamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Apply fitted transformation to new data and return X windows.
        """
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        processed = np.array(data, dtype=float, copy=True)

        self._ensure_log_flags(processed)

        # Log transform (use same flags/offset)
        if any(self.log_transform_flags or []):
            processed, _ = apply_log_transform(processed, self.log_transform_flags)

        # Filtering
        if self.apply_filter:
            processed = self._apply_filter(processed, method=self.filter_method)

        # Differencing
        if self.differencing:
            processed = np.vstack(
                [np.zeros_like(processed[0]), np.diff(processed, axis=0)]
            )

        # Detrending via EWT (if boundaries known)
        if self.apply_ewt and self.ewt_boundaries is not None:
            for i in range(processed.shape[1]):
                if self.ewt_boundaries and i < len(self.ewt_boundaries):
                    try:
                        ewt, _, _ = EWT1D(
                            processed[:, i],
                            N=len(self.ewt_boundaries[i]),
                            detect="given_bounds",
                            boundaries=self.ewt_boundaries[i],
                        )
                        if self.detrend:
                            processed[:, i] -= ewt[:, self.trend_imf_idx]
                    except Exception:
                        pass  # safe fallback

        # Normalization
        if self.normalize and self.scaler is not None:
            processed = self.scaler.transform(processed)

        # (Optional) time features appended only inside windowing like fit_transform
        time_feats = None
        if time_stamps is not None and self.generate_time_features:
            time_feats = self._generate_time_features(time_stamps)

        X, _, _ = self._create_sequences(processed, time_feats=time_feats)
        return X

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predicted values to the original scale.
        """
        preds = np.array(predictions, dtype=float, copy=True)

        # Denormalize
        if self.normalize and self.scaler is not None:
            preds = self.scaler.inverse_transform(preds)

        # Inverse differencing
        if self.differencing and self.diff_values is not None:
            last_value = self.diff_values[0]
            restored = np.zeros_like(preds)
            for t in range(len(preds)):
                last_value = last_value + preds[t]
                restored[t] = last_value
            preds = restored

        # Restore trend
        if self.detrend and self.trend_component is not None:
            n, d = preds.shape
            trend_to_add = np.zeros_like(preds)
            for i in range(d):
                trend = self.trend_component[:, i]
                if len(trend) == 0:
                    continue
                if n <= len(trend):
                    trend_to_add[:, i] = trend[:n]
                else:
                    look_back = min(10, len(trend) - 1) if len(trend) > 1 else 1
                    slope = (
                        (trend[-1] - trend[-look_back - 1]) / look_back
                        if look_back > 0
                        else 0.0
                    )
                    for j in range(n):
                        if j < len(trend):
                            trend_to_add[j, i] = trend[j]
                        else:
                            trend_to_add[j, i] = trend[-1] + slope * (
                                j - len(trend) + 1
                            )
            preds += trend_to_add

        # Inverse log
        if self.log_offset is not None and (self.log_transform_flags is not None):
            for i, flag in enumerate(self.log_transform_flags):
                if flag and i < preds.shape[1]:
                    preds[:, i] = np.exp(preds[:, i]) - self.log_offset[i]

        return preds

    # ---------- sub-steps ----------
    def _parallel_outlier_clean(self, data: np.ndarray) -> np.ndarray:
        method = self.outlier_method
        if method in {"tranad", "isolation_forest", "ecod", "lof"}:
            return _remove_outliers(
                data,
                method,
                self.outlier_threshold,
                seq_len=self.horizon,
                epochs=self.epochs,
            )
        # Per-column parallel
        n_features = data.shape[1]
        cleaned_cols = Parallel(n_jobs=-1)(
            delayed(_remove_outliers_parallel)(
                i, data[:, i], method, self.outlier_threshold
            )
            for i in tqdm(range(n_features), desc="Removing outliers")
        )
        cleaned_cols.sort(key=lambda t: t[0])
        return np.stack([col for _, col in cleaned_cols], axis=1)

    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        method = self.impute_method.lower()

        if method == "saits":
            saits_model = SAITSImputer(seq_len=self.window_size, epochs=self.epochs)
            saits_model.fit(data)
            return saits_model.impute(data)

        if method == "auto":
            try:
                from fancyimpute import IterativeImputer as FancyIter
            except Exception:
                FancyIter = None

            def impute_column(i, col, window_size=24):
                series = df[col]
                mr = float(series.isna().mean())
                try:
                    if mr < 0.05:
                        filled = series.interpolate(
                            method="linear", limit_direction="both"
                        )
                        return i, filled.ffill().bfill().values
                    elif mr < 0.2:
                        return (
                            i,
                            KNNImputer(n_neighbors=3)
                            .fit_transform(series.to_frame())
                            .ravel(),
                        )
                    elif FancyIter is not None:
                        imputed = (
                            FancyIter(max_iter=10, random_state=0)
                            .fit_transform(series.to_frame())
                            .ravel()
                        )
                        return i, imputed
                except Exception as e:
                    print(f"[WARN] Fallback imputation for column {col}: {e}")
                # Seasonal/lag fallback then mean
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
            return np.column_stack([col for _, col in results])

        if method == "mean":
            return df.fillna(df.mean()).values
        if method == "interpolate":
            return df.interpolate(method="linear").ffill().bfill().values
        if method == "ffill":
            return df.ffill().bfill().values
        if method == "bfill":
            return df.bfill().ffill().values
        if method == "knn":
            return KNNImputer(n_neighbors=5).fit_transform(df.values)
        if method == "iterative":
            try:
                from fancyimpute import IterativeImputer as FancyIter
            except Exception:
                try:
                    from sklearn.impute import IterativeImputer as FancyIter
                except Exception:
                    raise ImportError("Iterative imputer not available.")
            return FancyIter(random_state=0).fit_transform(df.values)

        raise ValueError(f"Unsupported imputation method: {method}")

    def _apply_filter(
        self, data: np.ndarray, method: str = "savgol", **kwargs
    ) -> np.ndarray:
        return _dispatch_filter(self, data, method, **kwargs)

    def _apply_ewt_and_detrend(self, data: np.ndarray, time_stamps=None) -> np.ndarray:
        try:
            from ewtpy import EWT1D  # noqa: F401
        except Exception:
            warnings.warn("PyEWT not installed. Skipping EWT.")
            return data

        output, ewt_components, ewt_boundaries, trend_components = (
            apply_ewt_and_detrend_parallel(
                data, self.ewt_bands, self.detrend, self.trend_imf_idx
            )
        )
        print(f"✅ EWT applied with {self.ewt_bands} bands.")
        self.ewt_components = ewt_components
        self.ewt_boundaries = ewt_boundaries
        if self.detrend:
            self.trend_component = trend_components
        return output

    def _generate_time_features(self, timestamps, freq: str = "h") -> np.ndarray:
        df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
        df["month"] = df.ts.dt.month / 12.0
        df["day"] = df.ts.dt.day / 31.0
        df["weekday"] = df.ts.dt.weekday / 6.0
        df["hour"] = df.ts.dt.hour / 23.0 if freq.lower() == "h" else 0.0
        return df[["month", "day", "weekday", "hour"]].values.astype(np.float32)

    def _create_sequences(
        self,
        data: np.ndarray,
        feats: Optional[List[int]] = None,
        time_feats: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        feats = list(range(data.shape[1])) if feats is None else feats
        X, y, time_f = [], [], []

        max_idx = len(data) - self.window_size - self.horizon + 1
        if max_idx <= 0:
            raise ValueError(
                f"Not enough data for window_size={self.window_size} and horizon={self.horizon} (len={len(data)})."
            )

        for i in tqdm(range(max_idx), desc="Creating sequences"):
            X.append(data[i : i + self.window_size])
            if time_feats is not None:
                time_f.append(time_feats[i : i + self.window_size])
            y.append(
                data[i + self.window_size : i + self.window_size + self.horizon][
                    :, feats
                ]
            )

        Xn = np.asarray(X)
        yn = np.asarray(y)
        tf = np.asarray(time_f) if time_feats is not None else None
        return Xn, yn, tf

    # ---------- viz ----------
    def _plot_comparison(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        title: str = "Preprocessing Comparison",
        time_stamps=None,
        max_features: int = 8,
    ) -> None:
        original = np.atleast_2d(original)
        cleaned = np.atleast_2d(cleaned)

        if original.shape[0] == 1:
            original = original.T
        if cleaned.shape[0] == 1:
            cleaned = cleaned.T
        if original.shape != cleaned.shape:
            raise ValueError(
                f"Shape mismatch after processing: original {original.shape}, cleaned {cleaned.shape}"
            )

        x = time_stamps if time_stamps is not None else np.arange(original.shape[0])
        if len(x) != original.shape[0]:
            raise ValueError(
                f"Length of x ({len(x)}) != n_samples ({original.shape[0]})"
            )

        d = original.shape[1]
        idx = list(range(min(d, max_features)))

        fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        for i in idx:
            axs[0].plot(x, original[:, i], label=f"Feature {i}")
            axs[1].plot(x, cleaned[:, i], label=f"Feature {i}")

        axs[0].set_title("Original")
        axs[1].set_title("Cleaned")
        axs[0].legend(ncol=min(len(idx), 4))
        axs[1].legend(ncol=min(len(idx), 4))
        axs[0].grid(True)
        axs[1].grid(True)
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

    # ---------- getters ----------
    def get_ewt_components(self) -> Optional[List]:
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> Optional[np.ndarray]:
        return self.trend_component if self.detrend else None
