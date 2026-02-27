# time_series_preprocessor.py
# =============================================================================
# Modern / clean refactor of your TimeSeriesPreprocessor
#
# What changed (cleanliness / less repetition):
# - ONE canonical preprocessing runner: _run_pipeline(mode="fit"/"transform")
#   so fit_transform() and transform() don't duplicate logic.
# - Stages are explicit and ordered; stage toggles are centralized.
# - Outlier threshold calibration is fit-aware and reused.
# - Time-features + windowing are always created via the same helper.
# - Plot checkpoints use one small helper.
#
# What stayed the same (features / behavior):
# - Same public API: fit_transform / transform / inverse_transform
# - Same external imports (assumed present)
# - Same auto_configure logic you wrote (kept, only lightly cleaned)
# - Same log transform behavior (per-channel flags, learned offsets)
# - Same EWT+detrend call: apply_ewt_and_detrend_parallel(...)
# - Same filter methods via centralized dispatch
# - Same imputation methods including SAITS + auto
# - Same windowing outputs (vectorized, with safe fallback)
# =============================================================================

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy.signal import find_peaks, welch
from scipy.stats import entropy, kurtosis, skew
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from statsmodels.tsa.stattools import acf, adfuller, pacf
from tabulate import tabulate
from tqdm import tqdm

# ---- local deps (explicit) ---------------------------------------------------
from .ewt import apply_ewt_and_detrend_parallel
from .filters import (
    adaptive_savgol_filter,
    emd_filter,
    kalman_filter,
    lowess_filter,
    wiener_filter,
)
from .impute import SAITSImputer
from .outlier import _remove_outliers, _remove_outliers_parallel

Mode = Literal["fit", "transform"]

# ---- optional deps -----------------------------------------------------------
try:
    from pykalman import KalmanFilter  # noqa: F401

    HAS_KALMAN = True
except Exception:
    HAS_KALMAN = False

try:
    from PyEMD import EMD  # noqa: F401

    HAS_EMD = True
except Exception:
    HAS_EMD = False


# -----------------------------------------------------------------------------
# Plot style (kept)
# -----------------------------------------------------------------------------
def set_plot_style() -> None:
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


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _as_2d(data: Any) -> np.ndarray:
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    x = np.asarray(data, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [T,D], got {x.shape}")
    return x


def apply_log_transform(
    data: np.ndarray,
    log_flags: List[bool],
    offsets: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply per-feature log transform with stable offsets.

    - If offsets is None: compute offsets from data (fit-time).
    - If offsets is given: use them (transform-time).
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"apply_log_transform expects 2D array, got shape {x.shape}")

    D = x.shape[1]
    if len(log_flags) != D:
        raise ValueError(f"log_flags length {len(log_flags)} != D {D}")

    if offsets is None:
        offsets = np.zeros(D, dtype=float)
        for i, flag in enumerate(log_flags):
            if flag:
                mn = float(np.nanmin(x[:, i]))
                offsets[i] = max(0.0, -mn + 1.0)
    else:
        offsets = np.asarray(offsets, dtype=float)
        if offsets.shape != (D,):
            raise ValueError(f"offsets shape {offsets.shape} != ({D},)")

    out = x.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for i, flag in enumerate(log_flags):
            if flag:
                out[:, i] = np.log(out[:, i] + offsets[i])
    return out, offsets


def compute_basic_stats(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(data, dtype=float)
    valid_mask = ~np.isnan(x)
    coverage = np.mean(valid_mask, axis=0)
    means = np.nanmean(x, axis=0)
    stds = np.nanstd(x, axis=0)
    skews = skew(x, nan_policy="omit")
    kurts = kurtosis(x, nan_policy="omit")
    return coverage, means, stds, skews, kurts


def detect_stationarity(data: np.ndarray, D: int) -> List[float]:
    pvals: List[float] = []
    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) <= 10:
            pvals.append(1.0)
            continue
        try:
            pvals.append(float(adfuller(clean, autolag="AIC")[1]))
        except Exception:
            pvals.append(1.0)
    return pvals


def detect_seasonality(
    data: np.ndarray, D: int
) -> Tuple[List[bool], List[Optional[int]]]:
    seasonal_flags: List[bool] = []
    detected_periods: List[Optional[int]] = []

    for i in range(D):
        clean = data[:, i][~np.isnan(data[:, i])]
        if len(clean) < 10:
            seasonal_flags.append(False)
            detected_periods.append(None)
            continue

        norm = (clean - np.mean(clean)) / (np.std(clean) + 1e-8)

        try:
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

            peak_freq = float(freqs[peaks[np.argmax(psd[peaks])]])
            period = int(round(1.0 / peak_freq)) if peak_freq > 0 else None

            acf_vals = acf(norm, nlags=min(100, len(norm) // 2), fft=True)
            acf_peaks, _ = find_peaks(acf_vals, height=0.2)
            strength = float(np.max(acf_vals[acf_peaks])) if len(acf_peaks) > 0 else 0.0
            is_seasonal = strength > 0.3
        except Exception:
            is_seasonal, period = False, None

        seasonal_flags.append(bool(is_seasonal))
        detected_periods.append(period if is_seasonal else None)

    return seasonal_flags, detected_periods


def analyze_signal_quality(data: np.ndarray, D: int) -> Tuple[List[float], List[float]]:
    flatness_scores: List[float] = []
    snr_scores: List[float] = []

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
            flat = float(np.exp(np.mean(np.log(spec + 1e-8))) / (np.mean(spec) + 1e-8))
        snr = float(np.max(spec) / (np.mean(spec) + 1e-8))

        flatness_scores.append(flat)
        snr_scores.append(snr)

    return flatness_scores, snr_scores


def score_pacf(data: np.ndarray, D: int) -> List[int]:
    scores: List[int] = []
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


def estimate_ewt_bands(data: np.ndarray, D: int) -> List[int]:
    band_estimates: List[int] = []
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


def _get_iterative_imputer_class() -> Optional[Any]:
    try:
        from fancyimpute import IterativeImputer as IterativeImputerCls

        return IterativeImputerCls
    except Exception:
        try:
            from sklearn.impute import IterativeImputer as IterativeImputerCls

            return IterativeImputerCls
        except Exception:
            return None


def summarize_configuration(params: Dict[str, Any]) -> None:
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


# -----------------------------------------------------------------------------
# Filter dispatch (kept, centralized)
# -----------------------------------------------------------------------------
def _dispatch_filter(self_ref, data: np.ndarray, method: str, **kwargs) -> np.ndarray:
    m = (method or "none").lower().strip()

    if m in {"none", "off", "false"}:
        return data

    if m == "kalman" and not HAS_KALMAN:
        warnings.warn("KalmanFilter not available; returning input unchanged.")
        return data

    if m == "emd" and not HAS_EMD:
        warnings.warn("PyEMD not available; returning input unchanged.")
        return data

    dispatch = {
        "savgol": lambda: adaptive_savgol_filter(
            data,
            window=self_ref.filter_window,
            polyorder=self_ref.filter_polyorder,
            n_jobs=kwargs.get("n_jobs", -1),
            robust_center=kwargs.get("robust_center", True),
            fill_nans_for_filter=kwargs.get("fill_nans_for_filter", True),
        ),
        "kalman": lambda: kalman_filter(
            data,
            n_iter=kwargs.get("n_iter", 5),
            min_points=kwargs.get("min_points", 10),
        ),
        "lowess": lambda: lowess_filter(
            data,
            frac=kwargs.get("frac", 0.05),
            it=kwargs.get("it", 0),
            delta=kwargs.get("delta", 0.0),
        ),
        "wiener": lambda: wiener_filter(
            data,
            mysize=kwargs.get("mysize", 15),
            noise=kwargs.get("noise", None),
            fill_nans_for_filter=kwargs.get("fill_nans_for_filter", True),
        ),
        "emd": lambda: emd_filter(
            data,
            keep_ratio=kwargs.get("keep_ratio", 0.5),
            n_jobs=kwargs.get("n_jobs", 1),
        ),
    }

    if m not in dispatch:
        raise ValueError(f"Unknown filter method: {method}")
    return dispatch[m]()


# =============================================================================
# Preprocessor Class
# =============================================================================
@dataclass
class TimeSeriesPreprocessor:
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

    # UX / plots
    plot: bool = False
    plot_max_features: int = 8

    # Learned / fitted attributes
    scaler: Optional[Any] = field(default=None, init=False)
    log_offset: Optional[np.ndarray] = field(default=None, init=False)
    diff_values: Optional[np.ndarray] = field(default=None, init=False)
    trend_component: Optional[np.ndarray] = field(default=None, init=False)
    ewt_components: Optional[List[Any]] = field(default=None, init=False)
    ewt_boundaries: Optional[List[Any]] = field(default=None, init=False)
    log_transform_flags: Optional[List[bool]] = field(default=None, init=False)

    # Outlier calibration
    outlier_thresholds_: Optional[np.ndarray] = field(default=None, init=False)
    outlier_calibration_: Dict[str, Any] = field(default_factory=dict, init=False)

    # Auto-config results
    scaling_method: str = field(default="standard", init=False) # 'standard', 'robust', 'quantile', 'log_only'
    filter_method: str = field(default="savgol", init=False)
    seasonal: bool = field(default=False, init=False)

    # Bookkeeping
    fitted_: bool = field(default=False, init=False)
    feature_dim_: Optional[int] = field(default=None, init=False)

    # Optional method availability map (kept)
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

    def __post_init__(self) -> None:
        set_plot_style()

    # -------------------------------------------------------------------------
    # Small helpers
    # -------------------------------------------------------------------------
    @staticmethod
    def _should_log_transform(sk: float, ku: float) -> bool:
        return (abs(sk) > 1.0) or (ku > 5.0)

    @staticmethod
    def _centered(data: np.ndarray, means: np.ndarray) -> np.ndarray:
        return data - means[np.newaxis, :]

    @staticmethod
    def _mad_sigma(x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        x = x[~np.isnan(x)]
        if x.size < 8:
            return float("nan")
        med = float(np.median(x))
        mad = float(np.median(np.abs(x - med)))
        return 1.4826 * mad + 1e-12

    def _maybe_plot(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        title: str,
        time_stamps: Optional[np.ndarray],
    ) -> None:
        if not self.plot:
            return
        self._plot_comparison(original, cleaned, title=title, time_stamps=time_stamps)

    def _ensure_log_flags(self, data: np.ndarray) -> None:
        if self.log_transform_flags is not None:
            return
        _, _, _, skews, kurts = compute_basic_stats(data)
        flags = [
            self._should_log_transform(float(sk), float(ku))
            for sk, ku in zip(skews, kurts)
        ]
        if not self.log_transform:
            flags = [False] * data.shape[1]
        self.log_transform_flags = flags

    # -------------------------------------------------------------------------
    # Auto-configure (your logic, lightly cleaned structurally)
    # -------------------------------------------------------------------------
    def auto_configure(self, data: np.ndarray, verbose: bool = True) -> None:
        if not self.self_tune:
            self._ensure_log_flags(data)
            return

        print("\n[Auto-Configuration — robust evidence-based edition]")

        x = np.asarray(data, dtype=float)
        T, D = x.shape

        coverage, means, stds, skews, kurts = compute_basic_stats(x)
        missing_rate = 1.0 - float(np.mean(coverage))

        flatness_scores, snr_scores = analyze_signal_quality(x, D)
        pacf_scores = score_pacf(x, D)
        seasonal_flags, periods = detect_seasonality(x, D)

        def _nanmedian(a: Any) -> float:
            return float(np.nanmedian(np.asarray(a, dtype=float)))

        def _frac(mask: Any) -> float:
            mask = np.asarray(mask, dtype=bool)
            return float(np.mean(mask)) if mask.size > 0 else 0.0

        med_flat = _nanmedian(flatness_scores)
        med_snr = _nanmedian(snr_scores)
        med_pacf = _nanmedian(pacf_scores)

        skew_abs = np.abs(np.asarray(skews, dtype=float))
        kurt = np.asarray(kurts, dtype=float)
        std = np.asarray(stds, dtype=float)

        high_skew_fraction = _frac(skew_abs > 2.0)
        heavy_tails_fraction = _frac(kurt > 6.0)

        # longest NaN run proxy
        max_nan_run = 0
        for j in range(D):
            col = x[:, j]
            isn = np.isnan(col)
            if not np.any(isn):
                continue
            run = 0
            best = 0
            for v in isn:
                if v:
                    run += 1
                    best = max(best, run)
                else:
                    run = 0
            max_nan_run = max(max_nan_run, best)
        nan_run_ratio = float(max_nan_run / max(1, T))

        med_std = _nanmedian(std)
        max_std = float(np.nanmax(std)) if np.isfinite(np.nanmax(std)) else 0.0
        scale_heterogeneity = float(max_std / (med_std + 1e-8)) if max_std > 0 else 1.0

        # Log decision
        log_recommend_per_channel: List[bool] = []
        for i in range(D):
            sk = float(skews[i])
            ku = float(kurts[i])
            sd = float(stds[i])
            mu = float(np.nanmean(x[:, i]))
            mn = float(np.nanmin(x[:, i])) if np.any(~np.isnan(x[:, i])) else 0.0
            mostly_positive = mn > -1e-6

            recommend = self._should_log_transform(sk, ku)
            recommend = recommend or (sd > 0 and mostly_positive and (mu > 10.0 * sd))
            recommend = recommend or (scale_heterogeneity > 8.0 and mostly_positive)
            log_recommend_per_channel.append(bool(recommend))

        log_fraction = float(np.mean(log_recommend_per_channel)) if D > 0 else 0.0
        log_score = (
            1.2 * log_fraction
            + 0.6 * high_skew_fraction
            + 0.6 * heavy_tails_fraction
            + 0.2 * float(scale_heterogeneity > 6.0)
            - 0.4 * float(missing_rate > 0.45)
        )
        self.log_transform = log_score > 0.75
        self.log_transform_flags = (
            log_recommend_per_channel if self.log_transform else [False] * D
        )

        extreme_ratio = float(
            np.nanmean(np.any(np.abs(self._centered(x, means)) > 6.0 * stds, axis=0))
        )
        strong_periods = len(set(int(p) for p in periods if (p is not None and p > 1)))

        stats: Dict[str, Any] = {
            "T": T,
            "D": D,
            "missing_rate": float(missing_rate),
            "nan_run_ratio": float(nan_run_ratio),
            "med_flatness": float(med_flat),
            "med_snr": float(med_snr),
            "med_pacf": float(med_pacf),
            "seasonal_fraction": float(np.mean(seasonal_flags))
            if len(seasonal_flags)
            else 0.0,
            "strong_periods": int(strong_periods),
            "extreme_ratio": float(extreme_ratio),
            "heavy_tails_fraction": float(heavy_tails_fraction),
            "high_skew_fraction": float(high_skew_fraction),
            "scale_heterogeneity": float(scale_heterogeneity),
            "log_fraction_recommended": float(log_fraction),
            "log_score": float(log_score),
        }

        archetype = "heuristic_robust"

        if (
            stats["missing_rate"] < 0.02
            and stats["med_flatness"] > 0.80
            and stats["med_snr"] > 3.0
        ):
            archetype = "clean_high_quality"
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "none" if stats["missing_rate"] == 0 else "interpolate"
            self.outlier_method, self.outlier_threshold = "quantile", 3.5

        elif (
            stats["med_pacf"] > 0.7
            and stats["med_snr"] < 2.2
            and stats["seasonal_fraction"] < 0.4
        ):
            archetype = "noisy_autoregressive"
            self.filter_method, self.apply_filter = "savgol", True
            self.impute_method = (
                "interpolate"
                if (stats["missing_rate"] < 0.12 and stats["nan_run_ratio"] < 0.05)
                else "saits"
            )
            self.outlier_method = "mad"
            self.outlier_threshold = float(
                3.3
                + 1.0 * stats["high_skew_fraction"]
                + 0.6 * stats["heavy_tails_fraction"]
            )

        elif stats["missing_rate"] > 0.30 and (
            stats["med_pacf"] < 0.35 or stats["nan_run_ratio"] > 0.08
        ):
            archetype = "sparse_irregular"
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "saits" if stats["missing_rate"] < 0.70 else "ffill"
            self.outlier_method, self.outlier_threshold = "mad", 4.5

        elif stats["extreme_ratio"] > 0.10 or stats["heavy_tails_fraction"] > 0.45:
            archetype = "heavy_tailed_outliers"
            self.filter_method, self.apply_filter = (
                ("wiener", True) if stats["D"] <= 64 else ("savgol", True)
            )
            self.impute_method = (
                "iterative" if stats["missing_rate"] < 0.15 else "saits"
            )
            self.outlier_method = "mad"
            self.outlier_threshold = float(
                4.6
                + 1.5 * stats["heavy_tails_fraction"]
                + 0.5 * stats["high_skew_fraction"]
            )

        else:
            # Filtering
            if stats["T"] < 200:
                self.filter_method, self.apply_filter = "none", False
            else:
                if stats["missing_rate"] > 0.25 and stats["T"] > 400:
                    self.filter_method, self.apply_filter = "kalman", True
                elif stats["med_flatness"] < 0.45 and stats["T"] > 800:
                    self.filter_method, self.apply_filter = "savgol", True
                elif stats["med_snr"] < 1.6:
                    self.filter_method, self.apply_filter = "wiener", True
                else:
                    self.filter_method, self.apply_filter = "none", False

            # Imputation
            if stats["missing_rate"] == 0:
                self.impute_method = "none"
            elif stats["missing_rate"] < 0.08 and stats["nan_run_ratio"] < 0.03:
                self.impute_method = (
                    "interpolate" if stats["med_pacf"] > 0.55 else "knn"
                )
            elif stats["missing_rate"] < 0.25:
                self.impute_method = (
                    "saits"
                    if (stats["nan_run_ratio"] > 0.06 or stats["D"] >= 5)
                    else "iterative"
                )
            else:
                self.impute_method = "saits"

            # Outlier method
            if (
                stats["heavy_tails_fraction"] > 0.35
                or stats["high_skew_fraction"] > 0.4
            ):
                self.outlier_method = "mad"
            elif (
                self.available_methods.get("tranad", False)
                and stats["T"] > 2000
                and stats["D"] <= 64
            ):
                self.outlier_method = "tranad"
            elif self.available_methods.get("ecod", False) and stats["D"] > 8:
                self.outlier_method = "ecod"
            elif stats["T"] > 4000 and stats["missing_rate"] < 0.12:
                self.outlier_method = "isolation_forest"
            else:
                self.outlier_method = "lof" if stats["D"] > 6 else "zscore"

            base = 3.6
            base += 1.0 * stats["high_skew_fraction"]
            base += 1.0 * stats["heavy_tails_fraction"]
            base += 1.2 if stats["extreme_ratio"] > 0.07 else 0.0
            base += 0.3 if self.impute_method == "saits" else 0.0
            self.outlier_threshold = float(base)

        # Fit-time outlier calibration (per-channel)
        try:
            self.outlier_thresholds_ = self._calibrate_outlier_thresholds(
                x,
                base=float(self.outlier_threshold),
                method=str(self.outlier_method),
                q=0.995 if T >= 1000 else 0.99,
                clamp=(2.5, 8.0),
            )
        except Exception:
            self.outlier_thresholds_ = None

        # Stationarity / seasonality / EWT
        if stats["T"] >= 200 and stats["missing_rate"] < 0.35:
            try:
                pvals = detect_stationarity(x, D)
                self.detrend = any(p > 0.05 for p in pvals) or (
                    stats["med_pacf"] > 0.55
                )
            except Exception:
                self.detrend = stats["med_pacf"] > 0.55
        else:
            self.detrend = stats["med_pacf"] > 0.55

        self.seasonal = (stats["seasonal_fraction"] > 0.25) or (
            stats["strong_periods"] >= 2
        )
        raw_bands = (
            3.0 + 0.9 * stats["strong_periods"] + 1.5 * (1.0 - stats["med_flatness"])
        )
        self.ewt_bands = int(max(3, min(12, round(raw_bands))))

        if not self.log_transform:
            self.log_transform_flags = [False] * D

        if stats["high_skew_fraction"] > 0.5 or (stats["scale_heterogeneity"] > 10.0):
             self.scaling_method = "quantile"
        elif stats["extreme_ratio"] > 0.15 or stats["heavy_tails_fraction"] > 0.4:
             self.scaling_method = "robust" 
        else:
             self.scaling_method = "standard"

        if verbose:
            summarize_configuration(
                {
                    "dimensions": f"{T} × {D}",
                    "missing_rate": stats["missing_rate"],
                    "pattern": archetype,
                    "log_transform": self.log_transform,
                    "log_fraction": stats["log_fraction_recommended"],
                    "scaling_method": self.scaling_method, # New field
                    "filter_method": self.filter_method,
                    "apply_filter": self.apply_filter,
                    "impute_method": self.impute_method,
                    "outlier_method": self.outlier_method,
                    "outlier_threshold": float(self.outlier_threshold),
                    "detrend": self.detrend,
                    "seasonal": self.seasonal,
                    "ewt_bands": int(self.ewt_bands),
                }
            )

        print("✅ Configuration complete.\n")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def fit_transform(
        self,
        data: np.ndarray,
        time_stamps: Optional[np.ndarray] = None,
        feats: Optional[List[int]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        x = _as_2d(data)
        self.feature_dim_ = x.shape[1]

        # configure (fit-time only)
        self.auto_configure(x, verbose=True)
        self._ensure_log_flags(x)

        processed = self._run_pipeline(x, mode="fit", time_stamps=time_stamps)

        time_feats = self._maybe_make_time_features(time_stamps, T=processed.shape[0])
        X, y, tf = self._create_sequences(processed, feats=feats, time_feats=time_feats)

        self.fitted_ = True
        return X, y, processed, tf

    def transform(
        self, data: np.ndarray, time_stamps: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")

        x = _as_2d(data)
        if self.feature_dim_ is not None and x.shape[1] != self.feature_dim_:
            raise ValueError(
                f"Feature dim mismatch: got {x.shape[1]}, expected {self.feature_dim_}"
            )

        processed = self._run_pipeline(x, mode="transform", time_stamps=time_stamps)

        time_feats = self._maybe_make_time_features(time_stamps, T=processed.shape[0])
        X, _, _ = self._create_sequences(processed, feats=None, time_feats=time_feats)
        return X

    def inverse_transform(self, predictions: np.ndarray) -> np.ndarray:
        preds = np.asarray(predictions, dtype=float)
        squeeze_h = False

        if preds.ndim == 3:
            N, H, D = preds.shape
            flat = preds.reshape(N * H, D)
        elif preds.ndim == 2:
            N, D = preds.shape
            H = 1
            flat = preds
            squeeze_h = True
        else:
            raise ValueError(f"predictions must be 2D or 3D, got shape {preds.shape}")

        if self.normalize and self.scaler is not None:
            flat = self.scaler.inverse_transform(flat)

        # Inverse differencing (approx)
        if self.differencing and self.diff_values is not None:
            last_value = self.diff_values[0].copy()
            restored = np.zeros_like(flat)
            for t in range(len(flat)):
                last_value = last_value + flat[t]
                restored[t] = last_value
            flat = restored

        # Restore trend
        if self.detrend and self.trend_component is not None:
            n, d = flat.shape
            trend_to_add = np.zeros_like(flat)
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
            flat = flat + trend_to_add

        # Inverse log
        if self.log_offset is not None and self.log_transform_flags is not None:
            for i, flag in enumerate(self.log_transform_flags):
                if flag and i < flat.shape[1]:
                    flat[:, i] = np.exp(flat[:, i]) - self.log_offset[i]

        if preds.ndim == 3:
            out = flat.reshape(N, H, D)
        else:
            out = flat
            if squeeze_h:
                out = out  # [N,D]
        return out

    # -------------------------------------------------------------------------
    # Canonical pipeline runner (removes duplication)
    # -------------------------------------------------------------------------
    def _run_pipeline(
        self, x: np.ndarray, *, mode: Mode, time_stamps: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Runs the preprocessing stages in a single place.

        Rules:
        - fit: may learn offsets/scaler/diff seed; uses auto_configure results.
        - transform: reuses learned offsets/scaler; does NOT silently change configuration.
        """
        processed = np.array(x, dtype=float, copy=True)

        # 1) Impute (optional)
        if np.any(np.isnan(processed)):
            if self.apply_imputation:
                processed = self._impute_missing(processed)
                self._maybe_plot(x, processed, "After Imputation", time_stamps)
            if np.any(np.isnan(processed)):
                raise ValueError(
                    "NaNs remain after imputation. Enable apply_imputation or change method."
                )

        # 2) Log transform (fit learns offsets; transform uses learned)
        self._ensure_log_flags(processed)
        if any(self.log_transform_flags or []):
            if mode == "fit":
                processed, self.log_offset = apply_log_transform(
                    processed, self.log_transform_flags
                )
            else:
                if self.log_offset is None:
                    raise RuntimeError("log_offset is not fitted.")
                processed, _ = apply_log_transform(
                    processed, self.log_transform_flags, offsets=self.log_offset
                )

        # 3) Outliers (kept: user-controlled; beware leakage but you already chose this)
        if self.remove_outliers:
            processed = self._parallel_outlier_clean(processed)
            self._maybe_plot(x, processed, "After Outlier Removal", time_stamps)
            if np.any(np.isnan(processed)) and self.apply_imputation:
                processed = self._impute_missing(processed)

        # 4) EWT + detrend
        if self.apply_ewt:
            processed = self._apply_ewt_and_detrend(processed)

        # 5) Filtering
        if self.apply_filter:
            processed = self._apply_filter(processed, method=self.filter_method)
            self._maybe_plot(
                x,
                processed,
                f"After {self.filter_method.capitalize()} Filtering",
                time_stamps,
            )

        # 6) Differencing
        if self.differencing:
            if mode == "fit":
                self.diff_values = processed[0:1].copy()
            processed = np.vstack(
                [np.zeros_like(processed[0]), np.diff(processed, axis=0)]
            )

        # 7) Normalization / Scaling (Adaptive)
        if self.normalize:
            if mode == "fit":
                if self.scaling_method == "robust":
                     self.scaler = RobustScaler(quantile_range=(5.0, 95.0))
                elif self.scaling_method == "quantile":
                     self.scaler = QuantileTransformer(output_distribution="normal", random_state=42)
                else: # standard
                     self.scaler = StandardScaler()
                
                processed = self.scaler.fit_transform(processed)
            else:
                if self.scaler is None:
                    raise RuntimeError("scaler is not fitted.")
                processed = self.scaler.transform(processed)

        return processed

    def _maybe_make_time_features(
        self, time_stamps: Optional[np.ndarray], *, T: int
    ) -> Optional[np.ndarray]:
        if time_stamps is None or not self.generate_time_features:
            return None
        tf = self._generate_time_features(time_stamps)
        if tf.shape[0] != T:
            raise ValueError(f"time_stamps length {len(time_stamps)} != T {T}")
        return tf

    # -------------------------------------------------------------------------
    # Outliers (kept)
    # -------------------------------------------------------------------------
    def _calibrate_outlier_thresholds(
        self,
        data: np.ndarray,
        base: float,
        method: str,
        q: float = 0.995,
        clamp: Tuple[float, float] = (2.5, 8.0),
    ) -> np.ndarray:
        x = np.asarray(data, dtype=float)
        _, D = x.shape
        thr = np.full(D, float(base), dtype=float)

        for j in range(D):
            col = x[:, j]
            col = col[~np.isnan(col)]
            if col.size < 32:
                continue

            c = float(np.median(col))
            sig = self._mad_sigma(col)
            if not np.isfinite(sig) or sig <= 0:
                continue

            rz = np.abs(col - c) / sig
            if rz.size < 32:
                continue

            tail = float(np.quantile(rz, q))
            alpha = 0.50
            thr[j] = (1.0 - alpha) * float(base) + alpha * tail

        thr = np.clip(thr, clamp[0], clamp[1])
        self.outlier_calibration_ = {
            "base": float(base),
            "q": float(q),
            "clamp": tuple(map(float, clamp)),
            "method": str(method),
        }
        return thr

    def _parallel_outlier_clean(self, data: np.ndarray) -> np.ndarray:
        method = (self.outlier_method or "iqr").lower()
        x = np.asarray(data, dtype=float)

        self._ensure_outlier_thresholds(x, method)

        if method in {"tranad", "isolation_forest", "ecod", "lof"}:
            agg_thr = (
                float(np.median(self.outlier_thresholds_))
                if self.outlier_thresholds_ is not None
                else float(self.outlier_threshold)
            )
            return _remove_outliers(
                x, method, agg_thr, seq_len=self.horizon, epochs=self.epochs
            )

        n_features = x.shape[1]
        thresholds = self.outlier_thresholds_

        def _thr(i: int) -> float:
            if thresholds is None:
                return float(self.outlier_threshold)
            return float(thresholds[i])

        cleaned_cols = Parallel(n_jobs=-1)(
            delayed(_remove_outliers_parallel)(i, x[:, i], method, _thr(i))
            for i in tqdm(range(n_features), desc="Removing outliers")
        )
        cleaned_cols.sort(key=lambda t: t[0])
        return np.stack([col for _, col in cleaned_cols], axis=1)

    def _ensure_outlier_thresholds(self, data: np.ndarray, method: str) -> None:
        x = np.asarray(data, dtype=float)
        if self.outlier_thresholds_ is not None and self.outlier_thresholds_.shape == (
            x.shape[1],
        ):
            return
        try:
            self.outlier_thresholds_ = self._calibrate_outlier_thresholds(
                x,
                base=float(self.outlier_threshold),
                method=method,
                q=0.995 if x.shape[0] >= 1000 else 0.99,
                clamp=(2.5, 8.0),
            )
        except Exception:
            self.outlier_thresholds_ = None

    # -------------------------------------------------------------------------
    # Imputation (kept)
    # -------------------------------------------------------------------------
    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        df = pd.DataFrame(data)
        method = (self.impute_method or "auto").lower()

        if method == "saits":
            saits_model = SAITSImputer(seq_len=self.window_size, epochs=self.epochs)
            saits_model.fit(data)
            return saits_model.impute(data)

        if method == "auto":
            missing_rate = float(np.mean(pd.isna(df.values)))
            IterativeImputerCls = (
                _get_iterative_imputer_class() if missing_rate >= 0.15 else None
            )

            if IterativeImputerCls is not None and missing_rate >= 0.15:
                try:
                    return IterativeImputerCls(
                        random_state=0, max_iter=10
                    ).fit_transform(df.values)
                except Exception:
                    pass

            def impute_column(
                i: int, col_name: Any, window_size: int = 24
            ) -> Tuple[int, np.ndarray]:
                series = df[col_name]
                mr = float(series.isna().mean())
                try:
                    if mr == 0.0:
                        return i, series.values.astype(float)
                    if mr < 0.05:
                        filled = series.interpolate(
                            method="linear", limit_direction="both"
                        )
                        return i, filled.ffill().bfill().values
                    if mr < 0.2:
                        return i, KNNImputer(n_neighbors=3).fit_transform(
                            series.to_frame()
                        ).ravel()
                except Exception as e:
                    print(f"[WARN] Column impute fallback for {col_name}: {e}")

                vals = series.values.astype(float)
                for t in range(len(vals)):
                    if (
                        np.isnan(vals[t])
                        and t >= window_size
                        and not np.isnan(vals[t - window_size])
                    ):
                        vals[t] = vals[t - window_size]
                m = float(np.nanmean(vals)) if np.isfinite(np.nanmean(vals)) else 0.0
                vals = np.where(np.isnan(vals), m, vals)
                return i, vals

            results = Parallel(n_jobs=-1)(
                delayed(impute_column)(i, col, self.window_size)
                for i, col in enumerate(
                    tqdm(df.columns, desc="Imputing Missing Values")
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
            IterativeImputerCls = _get_iterative_imputer_class()
            if IterativeImputerCls is None:
                raise ImportError("Iterative imputer not available.")
            return IterativeImputerCls(random_state=0).fit_transform(df.values)

        raise ValueError(f"Unsupported imputation method: {method}")

    # -------------------------------------------------------------------------
    # EWT / Filtering (kept)
    # -------------------------------------------------------------------------
    def _apply_filter(
        self, data: np.ndarray, method: str = "savgol", **kwargs
    ) -> np.ndarray:
        return _dispatch_filter(self, data, method, **kwargs)

    def _apply_ewt_and_detrend(self, data: np.ndarray) -> np.ndarray:
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

    # -------------------------------------------------------------------------
    # Time features + Windowing (kept)
    # -------------------------------------------------------------------------
    def _generate_time_features(
        self, timestamps: np.ndarray, freq: str = "h"
    ) -> np.ndarray:
        df = pd.DataFrame({"ts": pd.to_datetime(timestamps)})
        df["month"] = df.ts.dt.month / 12.0
        df["day"] = df.ts.dt.day / 31.0
        df["weekday"] = df.ts.dt.weekday / 6.0
        df["hour"] = df.ts.dt.hour / 23.0 if freq.lower() == "h" else 0.0
        return df[["month", "day", "weekday", "hour"]].values.astype(np.float32)

    def _create_sequences(
        self,
        data: np.ndarray,
        *,
        feats: Optional[List[int]] = None,
        time_feats: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """
        Fast windowing:
          X: [N, window, D]
          y: [N, horizon, |feats|]
          time_f: [N, window, F] or None
        """
        x = np.asarray(data, dtype=float)
        if x.ndim != 2:
            raise ValueError(f"data must be 2D [T,D], got {x.shape}")

        T, D = x.shape
        feats_idx = list(range(D)) if feats is None else list(feats)

        max_idx = T - self.window_size - self.horizon + 1
        if max_idx <= 0:
            raise ValueError(
                f"Not enough data for window_size={self.window_size} and horizon={self.horizon} (len={T})."
            )

        try:
            from numpy.lib.stride_tricks import sliding_window_view

            X_all = sliding_window_view(x, window_shape=self.window_size, axis=0)
            X = X_all[:max_idx, :, :]

            y_src = x[self.window_size : self.window_size + max_idx + self.horizon - 1]
            y_all = sliding_window_view(y_src, window_shape=self.horizon, axis=0)
            y = y_all[:, :, feats_idx]

            tf = None
            if time_feats is not None:
                tf2 = np.asarray(time_feats, dtype=float)
                if tf2.ndim != 2 or tf2.shape[0] != T:
                    raise ValueError(f"time_feats must be [T,F], got {tf2.shape}")
                tf_all = sliding_window_view(tf2, window_shape=self.window_size, axis=0)
                tf = tf_all[:max_idx, :, :]

            return (
                np.asarray(X),
                np.asarray(y),
                (np.asarray(tf) if tf is not None else None),
            )

        except Exception:
            X_list: List[np.ndarray] = []
            y_list: List[np.ndarray] = []
            tf_list: List[np.ndarray] = []

            for i in tqdm(range(max_idx), desc="Creating sequences"):
                X_list.append(x[i : i + self.window_size])
                y_list.append(
                    x[i + self.window_size : i + self.window_size + self.horizon][
                        :, feats_idx
                    ]
                )
                if time_feats is not None:
                    tf_list.append(time_feats[i : i + self.window_size])

            Xn = np.asarray(X_list)
            yn = np.asarray(y_list)
            tf = np.asarray(tf_list) if time_feats is not None else None
            return Xn, yn, tf

    # -------------------------------------------------------------------------
    # Visualization (kept)
    # -------------------------------------------------------------------------
    def _plot_comparison(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        title: str = "Preprocessing Comparison",
        time_stamps: Optional[np.ndarray] = None,
        max_features: Optional[int] = None,
    ) -> None:
        max_features = (
            self.plot_max_features if max_features is None else int(max_features)
        )

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

    # -------------------------------------------------------------------------
    # Getters (kept)
    # -------------------------------------------------------------------------
    def get_ewt_components(self) -> Optional[List[Any]]:
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> Optional[np.ndarray]:
        return self.trend_component if self.detrend else None
