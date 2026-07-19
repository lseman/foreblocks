"""foreblocks.ts_handler.preprocessing.

Time-series preprocessing and feature engineering pipeline.

Provides fit/transform preprocessing for time series data, including imputation,
filtering, outlier handling, detrending, and feature construction. The
TimeSeriesHandler class centralizes all preprocessing stages with auto-configuration
for filtering, imputation, scaling, and log transforms based on data characteristics.

Core API:
- TimeSeriesHandler: comprehensive time-series preprocessing pipeline

"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from foreblocks.ts_handler.auto_configure import (
    _auto_select_filter_method,
    summarize_configuration,
)
from foreblocks.ts_handler.diagnostics import (
    _get_iterative_imputer_class,
    analyze_signal_quality,
    detect_seasonality,
    detect_stationarity,
    score_ljung_box,
    score_pacf,
)
from foreblocks.ts_handler.filters import (
    adaptive_savgol_filter,
    emd_filter,
    kalman_filter,
    lowess_filter,
    ssa_filter,
    stl_filter,
    wiener_filter,
)
from foreblocks.ts_handler.ewt import apply_ewt_and_detrend_parallel
from foreblocks.ts_handler.impute import SAITSImputer
from foreblocks.ts_handler.outlier import _remove_outliers, _remove_outliers_parallel
from foreblocks.ts_handler.plotting import _plot_comparison, set_plot_style
from foreblocks.ts_handler.time_features import (
    _generate_time_features,
    _infer_timestamp_frequency,
    _maybe_make_time_features,
)
from foreblocks.ts_handler.transforms import (
    _apply_log_stage,
    _apply_scaling_stage,
    _centered,
    _ensure_log_flags,
    _mad_sigma,
    _should_log_transform,
)
from foreblocks.ts_handler.utils import (
    _as_2d,
    _hybrid_impute,
    _linear_interpolate_2d,
    _longest_nan_run,
    _mean_fill_2d,
    _select_diagnostic_features,
    compute_basic_stats,
)
from foreblocks.ts_handler.windowing import _create_sequences

# ---- local deps (explicit) ---------------------------------------------------
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
        "ssa": lambda: ssa_filter(
            data,
            window_length=kwargs.get("window_length", None),
            n_components=kwargs.get("n_components", 2),
            fill_nans_for_filter=kwargs.get("fill_nans_for_filter", True),
        ),
        "stl": lambda: stl_filter(
            data,
            period=kwargs.get("period", 7),
            robust=kwargs.get("robust", True),
            seasonal=kwargs.get("seasonal", 7),
            trend=kwargs.get("trend", None),
            return_component=kwargs.get("return_component", "trend_seasonal"),
            fill_nans_for_filter=kwargs.get("fill_nans_for_filter", True),
        ),
    }

    if m not in dispatch:
        raise ValueError(f"Unknown filter method: {method}")
    return dispatch[m]()


# =============================================================================
# Handler Class
# =============================================================================
@dataclass
class TimeSeriesHandler:
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
    time_feature_mode: str = "cyclical"
    verbose: bool = False

    # UX / plots
    plot: bool = False
    plot_max_features: int = 8

    # Learned / fitted attributes
    scaler: Any | None = field(default=None, init=False)
    log_offset: np.ndarray | None = field(default=None, init=False)
    diff_values: np.ndarray | None = field(default=None, init=False)
    trend_component: np.ndarray | None = field(default=None, init=False)
    ewt_components: list[Any] | None = field(default=None, init=False)
    ewt_boundaries: list[Any] | None = field(default=None, init=False)
    log_transform_flags: list[bool] | None = field(default=None, init=False)

    # Outlier calibration
    outlier_thresholds_: np.ndarray | None = field(default=None, init=False)
    outlier_calibration_: dict[str, Any] = field(default_factory=dict, init=False)
    filter_selection_: dict[str, Any] = field(default_factory=dict, init=False)

    # Auto-config results
    scaling_method: str = field(
        default="standard", init=False
    )  # 'standard', 'robust', 'quantile', 'log_only', 'box_cox'
    filter_method: str = field(default="savgol", init=False)
    seasonal: bool = field(default=False, init=False)

    # Bookkeeping
    fitted_: bool = field(default=False, init=False)
    feature_dim_: int | None = field(default=None, init=False)

    # Optional method availability map (kept)
    available_methods: dict[str, bool] = field(
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
    def _maybe_plot(
        self,
        cleaned: np.ndarray,
        title: str,
        time_stamps: np.ndarray | None,
        original: np.ndarray | None = None,
    ) -> None:
        if not self.plot:
            return
        orig = original if original is not None else cleaned
        self._plot_comparison(orig, cleaned, title=title, time_stamps=time_stamps)

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(f"[Preprocessing] {message}")

    def _build_knn_imputer(self, n_samples: int) -> Any:
        from sklearn.impute import KNNImputer

        return KNNImputer(n_neighbors=min(5, max(2, n_samples - 1)))

    def _fallback_imputation_method(self, x: np.ndarray) -> str:
        return "interpolate" if _longest_nan_run(x) <= self.window_size else "ffill"

    def _resolve_imputation_method(self, x: np.ndarray) -> str:
        method = (self.impute_method or "auto").lower()
        if method != "auto" and method != "saits":
            return method

        saits_min_points = max(32, 2 * self.window_size + self.horizon)
        missing_rate = float(np.mean(np.isnan(x)))
        longest_gap = _longest_nan_run(x)

        if method == "saits":
            if x.shape[0] >= saits_min_points:
                return "saits"
            fallback = self._fallback_imputation_method(x)
            self._vprint(
                f"SAITS skipped for short series (T={x.shape[0]}). Falling back to {fallback}."
            )
            return fallback

        if x.shape[0] >= saits_min_points and (
            missing_rate >= 0.25 or longest_gap > max(4, self.window_size // 2)
        ):
            return "saits"
        if missing_rate < 0.05 and longest_gap <= max(2, self.window_size // 4):
            return "interpolate"
        if x.shape[1] > 1 and missing_rate < 0.20:
            return "knn"
        if x.shape[1] > 1 and missing_rate >= 0.15:
            try:
                _get_iterative_imputer_class()
                return "iterative"
            except Exception:
                pass
        return "hybrid"

    @staticmethod
    def _infer_timestamp_frequency(timestamps: np.ndarray) -> str:
        return _infer_timestamp_frequency(timestamps)

    # -------------------------------------------------------------------------
    # Outlier calibration
    # -------------------------------------------------------------------------
    def _calibrate_outlier_thresholds(
        self,
        data: np.ndarray,
        base: float,
        method: str,
        q: float = 0.995,
        clamp: tuple[float, float] = (2.5, 8.0),
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
            sig = _mad_sigma(col)
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

        iterator = range(n_features)
        if self.verbose:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="Removing outliers")

        backend = (
            "threading" if method in {"zscore", "iqr", "mad", "quantile"} else "loky"
        )
        cleaned_cols = Parallel(n_jobs=-1, backend=backend)(
            delayed(_remove_outliers_parallel)(i, x[:, i], method, _thr(i))
            for i in iterator
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
    # Imputation
    # -------------------------------------------------------------------------
    def _impute_missing(self, data: np.ndarray) -> np.ndarray:
        x = _as_2d(data)
        method = self._resolve_imputation_method(x)

        if not np.isnan(x).any():
            return x.copy()

        if method == "saits":
            saits_model = SAITSImputer(seq_len=self.window_size, epochs=self.epochs)
            saits_model.fit(x)
            return saits_model.impute(x)

        if method == "mean":
            return _mean_fill_2d(x)
        if method == "interpolate":
            return _linear_interpolate_2d(x)
        if method == "hybrid":
            return _hybrid_impute(x)
        if method == "ffill":
            return pd.DataFrame(x).ffill().bfill().to_numpy(dtype=float)
        if method == "bfill":
            return pd.DataFrame(x).bfill().ffill().to_numpy(dtype=float)
        if method == "knn":
            return self._build_knn_imputer(x.shape[0]).fit_transform(x)
        if method == "iterative":
            IterativeImputerCls = _get_iterative_imputer_class()
            if IterativeImputerCls is None:
                raise ImportError("Iterative imputer not available.")

            from sklearn.ensemble import HistGradientBoostingRegressor

            estimator = HistGradientBoostingRegressor(
                random_state=0,
                max_iter=30,
                early_stopping=True,
                max_bins=64,
            )
            return IterativeImputerCls(
                estimator=estimator,
                random_state=0,
                max_iter=3,
                n_nearest_features=min(10, x.shape[1]),
            ).fit_transform(x)

        raise ValueError(f"Unsupported imputation method: {method}")

    # -------------------------------------------------------------------------
    # EWT / Filtering
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
    # Auto-configure
    # -------------------------------------------------------------------------
    def auto_configure(self, data: np.ndarray, verbose: bool = True) -> None:
        if not self.self_tune:
            _ensure_log_flags(self, data)
            return

        print("\n[Auto-Configuration — robust evidence-based edition]")

        x = np.asarray(data, dtype=float)
        T, D = x.shape

        coverage, means, stds, skews, kurts = compute_basic_stats(x)
        missing_rate = 1.0 - float(np.mean(coverage))
        diag_idx = _select_diagnostic_features(
            D, max_features=min(32, max(8, int(np.sqrt(max(D, 1))) * 4))
        )
        x_diag = x[:, diag_idx]
        D_diag = x_diag.shape[1]

        flatness_scores, snr_scores = analyze_signal_quality(x_diag, D_diag)
        ljung_box_pvals = score_ljung_box(x_diag, D_diag)
        pacf_scores = score_pacf(x_diag, D_diag)
        seasonal_flags, periods = detect_seasonality(x_diag, D_diag)

        def _nanmedian(a: Any) -> float:
            return float(np.nanmedian(np.asarray(a, dtype=float)))

        def _frac(mask: Any) -> float:
            mask = np.asarray(mask, dtype=bool)
            return float(np.mean(mask)) if mask.size > 0 else 0.0

        med_flat = _nanmedian(flatness_scores)
        med_snr = _nanmedian(snr_scores)
        med_pacf = _nanmedian(pacf_scores)

        # Determine formal AR structure via Ljung-Box p-values
        is_autoregressive = _frac(np.array(ljung_box_pvals) < 0.05) > 0.25

        # Jarque-Bera Test for formal Non-Normality / Heavy Tails
        jb_pvals: list[float] = []
        from scipy.stats import jarque_bera

        for i in range(D):
            clean = x[:, i][~np.isnan(x[:, i])]
            if len(clean) > 8:
                try:
                    res = jarque_bera(clean)
                    jb_pvals.append(float(res.pvalue))
                except Exception:
                    jb_pvals.append(1.0)
            else:
                jb_pvals.append(1.0)

        is_heavy_tailed = _frac(np.array(jb_pvals) < 0.01) > 0.25

        skew_abs = np.abs(np.asarray(skews, dtype=float))
        kurt = np.asarray(kurts, dtype=float)
        std = np.asarray(stds, dtype=float)

        high_skew_fraction = _frac(skew_abs > 2.0)
        heavy_tails_fraction = _frac(kurt > 6.0)

        # longest NaN run proxy
        max_nan_run = _longest_nan_run(x)
        nan_run_ratio = float(max_nan_run / max(1, T))

        med_std = _nanmedian(std)
        max_std = float(np.nanmax(std)) if np.isfinite(np.nanmax(std)) else 0.0
        scale_heterogeneity = float(max_std / (med_std + 1e-8)) if max_std > 0 else 1.0

        # Log decision
        log_recommend_per_channel: list[bool] = []
        for i in range(D):
            sk = float(skews[i])
            ku = float(kurts[i])
            sd = float(stds[i])
            mu = float(np.nanmean(x[:, i]))
            mn = float(np.nanmin(x[:, i])) if np.any(~np.isnan(x[:, i])) else 0.0
            mostly_positive = mn > -1e-6

            recommend = _should_log_transform(sk, ku)
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
        strong_periods = len({int(p) for p in periods if (p is not None and p > 1)})

        stats: dict[str, Any] = {
            "T": T,
            "D": D,
            "missing_rate": float(missing_rate),
            "nan_run_ratio": float(nan_run_ratio),
            "med_flatness": float(med_flat),
            "med_snr": float(med_snr),
            "med_pacf": float(med_pacf),
            "is_autoregressive": bool(is_autoregressive),
            "seasonal_fraction": (
                float(np.mean(seasonal_flags)) if len(seasonal_flags) else 0.0
            ),
            "strong_periods": int(strong_periods),
            "extreme_ratio": float(extreme_ratio),
            "is_heavy_tailed": bool(is_heavy_tailed),
            "heavy_tails_fraction": float(heavy_tails_fraction),
            "high_skew_fraction": float(high_skew_fraction),
            "scale_heterogeneity": float(scale_heterogeneity),
            "log_fraction_recommended": float(log_fraction),
            "log_score": float(log_score),
            "dominant_period": (
                int(
                    round(np.nanmedian([p for p in periods if p is not None and p > 1]))
                )
                if any(p is not None and p > 1 for p in periods)
                else 7
            ),
        }

        archetype = "heuristic_robust"

        if (
            stats["missing_rate"] < 0.02
            and stats["med_flatness"] > 0.80
            and stats["med_snr"] > 3.0
            and not stats["is_heavy_tailed"]
        ):
            archetype = "clean_high_quality"
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "none" if stats["missing_rate"] == 0 else "interpolate"
            self.outlier_method, self.outlier_threshold = "zscore", 3.5

        elif (
            (stats["med_pacf"] > 0.7 or stats["is_autoregressive"])
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
            (stats["med_pacf"] < 0.35 and not stats["is_autoregressive"])
            or stats["nan_run_ratio"] > 0.08
        ):
            archetype = "sparse_irregular"
            self.filter_method, self.apply_filter = "none", False
            self.impute_method = "saits" if stats["missing_rate"] < 0.70 else "ffill"
            self.outlier_method, self.outlier_threshold = "mad", 4.5

        elif (
            stats["extreme_ratio"] > 0.10
            or stats["heavy_tails_fraction"] > 0.45
            or stats["is_heavy_tailed"]
        ):
            archetype = "heavy_tailed_outliers"
            self.filter_method, self.apply_filter = (
                ("ssa", True) if stats["D"] <= 64 else ("savgol", True)
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
            self.scaling_method = "box_cox"

        else:
            # Filtering
            if stats["T"] < 200:
                self.filter_method, self.apply_filter = "none", False
            else:
                if stats["strong_periods"] >= 1 and stats["missing_rate"] < 0.2:
                    self.filter_method, self.apply_filter = "stl", True
                elif stats["heavy_tails_fraction"] > 0.3 or stats["med_snr"] < 1.0:
                    self.filter_method, self.apply_filter = "ssa", True
                elif stats["missing_rate"] > 0.25 and stats["T"] > 400:
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
                or stats["is_heavy_tailed"]
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
                self.detrend = (
                    any(p > 0.05 for p in pvals) or stats["is_autoregressive"]
                )
            except Exception:
                self.detrend = stats["is_autoregressive"]
        else:
            self.detrend = stats["is_autoregressive"]

        self.seasonal = (stats["seasonal_fraction"] > 0.25) or (
            stats["strong_periods"] >= 2
        )
        raw_bands = (
            3.0 + 0.9 * stats["strong_periods"] + 1.5 * (1.0 - stats["med_flatness"])
        )
        self.ewt_bands = int(max(3, min(12, round(raw_bands))))

        if not self.log_transform:
            self.log_transform_flags = [False] * D

        saits_min_points = max(32, 2 * self.window_size + self.horizon)
        if self.impute_method == "saits" and T < saits_min_points:
            self.impute_method = (
                "interpolate" if stats["nan_run_ratio"] < 0.10 else "ffill"
            )

        self.apply_imputation = bool(
            stats["missing_rate"] > 0
            and self.impute_method not in {"none", "off", "false"}
        )
        self.remove_outliers = bool(
            self.outlier_method not in {"none", "off", "false"}
            and (
                stats["extreme_ratio"] > 0.01
                or stats["heavy_tails_fraction"] > 0.15
                or stats["is_heavy_tailed"]
            )
        )

        if stats["high_skew_fraction"] > 0.5 or (stats["scale_heterogeneity"] > 10.0):
            self.scaling_method = "quantile"
        elif stats["extreme_ratio"] > 0.15 or stats["heavy_tails_fraction"] > 0.4:
            self.scaling_method = "robust"
        else:
            self.scaling_method = "standard"

        self.filter_selection_ = {}
        if stats["T"] >= max(48, self.window_size * 3):
            try:
                filter_selection = self._auto_select_filter_method(x, stats)
                self.filter_selection_ = filter_selection
                self.filter_method = str(filter_selection["best_method"])
                self.apply_filter = bool(filter_selection["apply_filter"])
                if verbose and isinstance(filter_selection.get("scores"), pd.DataFrame):
                    top = filter_selection["scores"].head(3).copy()
                    if not top.empty:
                        print("[Auto-Filter Ranking]")
                        print(
                            top[
                                [
                                    "score",
                                    "fidelity_mse",
                                    "roughness",
                                    "residual_autocorr",
                                    "derivative_corr",
                                ]
                            ].round(4)
                        )
            except Exception as exc:
                if self.verbose:
                    print(f"[Preprocessing] Filter auto-selection skipped: {exc}")

        if verbose:
            summarize_configuration(
                {
                    "dimensions": f"{T} × {D}",
                    "missing_rate": stats["missing_rate"],
                    "pattern": archetype,
                    "log_transform": self.log_transform,
                    "log_fraction": stats["log_fraction_recommended"],
                    "scaling_method": self.scaling_method,
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
        time_stamps: np.ndarray | None = None,
        feats: list[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
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
        self, data: np.ndarray, time_stamps: np.ndarray | None = None
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
    # Internal pipeline runner
    # -------------------------------------------------------------------------
    def _run_pipeline(
        self, x: np.ndarray, *, mode: str, time_stamps: np.ndarray | None
    ) -> np.ndarray:
        """
        Runs the preprocessing stages in a single place.

        Rules:
        - fit: may learn offsets/scaler/diff seed; uses auto_configure results.
        - transform: reuses learned offsets/scaler; does NOT silently change configuration.
        """
        processed = np.array(x, dtype=float, copy=True)
        self._vprint(f"Starting {mode} pipeline (shape: {x.shape})")

        # 1) Impute (optional)
        if np.any(np.isnan(processed)):
            if self.apply_imputation:
                self._vprint(f"Applying imputation ({self.impute_method})")
                processed = self._impute_missing(processed)
                self._maybe_plot(processed, "After Imputation", time_stamps, original=x)
            if np.any(np.isnan(processed)):
                raise ValueError(
                    "NaNs remain after imputation. Enable apply_imputation or change method."
                )

        # 2) Log transform (fit learns offsets; transform uses learned)
        processed = _apply_log_stage(
            self, processed, mode, self.log_transform, self.log_transform_flags, self._vprint
        )

        # 3) Outliers (kept: user-controlled; beware leakage but you already chose this)
        if self.remove_outliers:
            self._vprint(f"Applying outlier removal ({self.outlier_method})")
            processed = self._parallel_outlier_clean(processed)
            self._maybe_plot(processed, "After Outlier Removal", time_stamps, original=x)
            if np.any(np.isnan(processed)) and self.apply_imputation:
                self._vprint("Re-applying imputation after outlier removal")
                processed = self._impute_missing(processed)
            if np.any(np.isnan(processed)):
                raise ValueError(
                    "NaNs remain after outlier removal. Enable imputation or adjust the outlier settings."
                )

        # 4) EWT + detrend
        if self.apply_ewt:
            self._vprint(f"Applying EWT & detrending ({self.ewt_bands} bands)")
            processed = self._apply_ewt_and_detrend(processed)

        # 5) Filtering
        if self.apply_filter:
            self._vprint(f"Applying signal filtering ({self.filter_method})")
            processed = self._apply_filter(processed, method=self.filter_method)
            self._maybe_plot(
                processed,
                f"After {self.filter_method.capitalize()} Filtering",
                time_stamps,
                original=x,
            )

        # 6) Differencing
        if self.differencing:
            self._vprint("Applying differencing")
            if mode == "fit":
                self.diff_values = processed[0:1].copy()
            processed = np.vstack(
                [
                    np.zeros_like(processed[0]),
                    np.diff(processed, axis=0),
                ]
            )

        # 7) Normalization / Scaling (Adaptive)
        return _apply_scaling_stage(
            self, processed, mode, self.normalize, self.scaling_method, self._vprint
        )

    def _maybe_make_time_features(
        self, time_stamps: np.ndarray | None, *, T: int
    ) -> np.ndarray | None:
        return _maybe_make_time_features(
            time_stamps,
            self.generate_time_features,
            self._infer_timestamp_frequency,
            _generate_time_features,
            T,
        )

    def _ensure_log_flags(self, data: np.ndarray) -> None:
        _ensure_log_flags(self, data)

    def _generate_time_features(self, timestamps: np.ndarray) -> np.ndarray:
        return _generate_time_features(
            timestamps,
            freq=self._infer_timestamp_frequency(timestamps),
            time_feature_mode=self.time_feature_mode,
        )

    def _auto_select_filter_method(
        self, data: np.ndarray, stats: dict[str, Any]
    ) -> dict[str, Any]:
        return _auto_select_filter_method(
            data,
            stats,
            self.filter_method,
            self._apply_filter,
            self.verbose,
        )

    def _plot_comparison(
        self,
        original: np.ndarray,
        cleaned: np.ndarray,
        title: str = "Preprocessing Comparison",
        time_stamps: np.ndarray | None = None,
        max_features: int | None = None,
    ) -> None:
        _plot_comparison(
            original, cleaned, title=title, time_stamps=time_stamps, max_features=max_features
        )

    def _centered(self, data: np.ndarray, means: np.ndarray) -> np.ndarray:
        return _centered(data, means)

    def _create_sequences(
        self,
        data: np.ndarray,
        *,
        feats: list[int] | None = None,
        time_feats: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        return _create_sequences(
            data,
            window_size=self.window_size,
            horizon=self.horizon,
            feats=feats,
            time_feats=time_feats,
        )

    # -------------------------------------------------------------------------
    # Getters
    # -------------------------------------------------------------------------
    def get_ewt_components(self) -> list[Any] | None:
        return self.ewt_components if self.apply_ewt else None

    def get_trend_component(self) -> np.ndarray | None:
        return self.trend_component if self.detrend else None
