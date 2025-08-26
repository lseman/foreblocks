import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Assumes BaseFeatureTransformer and AdaptiveMI are available
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple, Union

import hdbscan
import numpy as np
import pandas as pd
import torch
from scipy import fft, stats
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_regression,
    mutual_info_regression,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
from sklearn.preprocessing import (
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    StandardScaler,
)

# NOTE: AdaptiveMI import/path remains untouched to avoid breaking downstream
from foretools.aux.adaptive_mi import AdaptiveMI

# Type hints for better code clarity
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]
ModelLike = Any


# Update FeatureConfig to include RFECV parameters
@dataclass
class FeatureConfig:
    """Configuration class for feature engineering parameters."""

    # Core settings
    task: str = "classification"
    random_state: int = 42
    corr_threshold: float = 0.95
    rare_threshold: float = 0.01

    # Feature creation flags
    create_interactions: bool = True
    create_math_features: bool = True
    create_binning: bool = True
    create_clustering: bool = True
    create_statistical: bool = True
    create_fourier: bool = False
    use_autoencoder: bool = False

    # Feature limits
    max_interactions: int = 50
    max_selected_interactions: int = 20
    n_bins: int = 5
    n_clusters: int = 8
    n_fourier_terms: int = 3

    # Selection parameters
    use_quantile_transform: bool = True
    target_encode_threshold: int = 10
    mi_threshold: float = 0.001
    shap_threshold: float = 0.001

    # RFECV parameters
    use_rfecv: bool = False
    rfecv_step: Union[int, float] = 0.1
    rfecv_cv: int = 5
    rfecv_min_features: Optional[int] = None  # Auto-calculated if None
    rfecv_max_features: Optional[int] = None  # Auto-calculated if None
    rfecv_patience: int = 5
    rfecv_use_ensemble: bool = True
    rfecv_stability_selection: bool = True

    # Autoencoder settings
    autoencoder_epochs: int = 50
    autoencoder_batch_size: int = 64
    autoencoder_lr: float = 1e-3
    autoencoder_device: str = "cuda" if torch.cuda.is_available() else "cpu"
    autoencoder_latent_ratio: float = 0.25

    # Interaction-specific settings
    create_interactions: bool = True
    max_pairs_screen: int = 200  # pairs to score in stage-1
    max_interactions: int = 200  # hard cap of generated features
    max_selected_interactions: int = 64  # final kept features
    scorer: str = "mi"  # {"mi", "hsic"}
    task: str = "regression"  # {"regression","classification"}
    n_splits: int = 5  # CV splits for stability selection
    min_selected_per_fold: int = 20  # per-fold top-k before aggregating
    importance_agg: str = "median"  # {"mean","median","max"}
    random_state: int = 42
    max_rows_score: int = 50000  # speed guard for scoring
    model_backend: str = "xgb"  # {"xgb","lgb"}
    device: Optional[str] = None  # "cuda" for xgb if desired
    # robust ops toggles
    include_sum: bool = True
    include_diff: bool = True
    include_prod: bool = True
    include_ratio: bool = True
    include_norm_ratio: bool = True
    include_minmax: bool = True
    # screening knobs
    pair_corr_with_y: bool = True  # use |corr(feature, y)| to order cols
    pair_max_per_feature: int = 32  # limit partners per anchor to reduce blow-up
    corr_avoid_redundancy: float = 0.995  # skip pairs with |corr(a,b)| above this
    # safety
    epsilon: float = 1e-8
    dtype_out: str = "float32"

    # Random Fourier Features parameters
    create_rff: bool = True
    rff_n_components: int = 100
    rff_gamma: Union[float, str] = "auto"
    rff_kernel: str = "rbf"
    rff_max_features: int = 50


class BaseFeatureTransformer(ABC):
    """Abstract base class for feature transformers."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureTransformer":
        """Fit the transformer."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


class DateTimeTransformer(BaseFeatureTransformer):
    """
    Datetime feature extraction with tz support, DST-safe cyclicals, fixed anchors,
    and optional holiday/business flags.
    """

    def __init__(
        self,
        config,
        include_cyclical: bool = True,
        include_flags: bool = True,
        include_elapsed: bool = True,
        group_key: Optional[str] = None,  # e.g., customer_id, series_id
        country_holidays: Optional[str] = None,  # e.g., "BR", "US", "GB"
    ):
        super().__init__(config)
        self.include_cyclical = include_cyclical
        self.include_flags = include_flags
        self.include_elapsed = include_elapsed
        self.group_key = group_key
        self.country_holidays = country_holidays

        self.datetime_cols_: list = []
        self._anchors_global_: Dict[str, pd.Timestamp] = {}
        self._anchors_group_: Dict[str, Dict[Union[str, int], pd.Timestamp]] = {}
        self._has_holidays = False
        self._holiday_set = None

    def _coerce_dt(self, s: pd.Series) -> pd.Series:
        if not pd.api.types.is_datetime64_any_dtype(s):
            s = pd.to_datetime(s, errors="coerce", utc=True)
        # normalize to UTC, keep as tz-aware to avoid DST misalignment; downstream uses .dt components
        if s.dt.tz is None:
            s = s.dt.tz_localize("UTC")
        return s

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DateTimeTransformer":
        # detect datetime columns, including tz-aware
        self.datetime_cols_ = [
            c
            for c in X.columns
            if pd.api.types.is_datetime64_any_dtype(X[c])
            or (
                pd.api.types.is_object_dtype(X[c])
                and pd.to_datetime(X[c], errors="ignore") is not X[c]
            )
        ]
        # Coerce and store anchors (global, and optionally per-group)
        if not self.datetime_cols_:
            self.is_fitted = True
            return self

        if self.country_holidays:
            try:
                import holidays as _hol

                self._holiday_set = _hol.country_holidays(self.country_holidays)
                self._has_holidays = True
            except Exception:
                self._has_holidays = False

        gkey = self.group_key if (self.group_key in X.columns) else None

        for col in self.datetime_cols_:
            dt = self._coerce_dt(X[col])
            valid = dt.dropna()
            if valid.empty:
                # Anchor to a fixed epoch if column is all NaT
                self._anchors_global_[col] = pd.Timestamp("1970-01-01", tz="UTC")
                continue
            self._anchors_global_[col] = valid.min()

            if gkey is not None:
                self._anchors_group_.setdefault(col, {})
                # per-group anchor = group-specific min timestamp (prevents leakage across groups)
                grp = X[[gkey, col]].copy()
                grp[col] = self._coerce_dt(grp[col])
                a = grp.dropna().groupby(gkey)[col].min()
                self._anchors_group_[col] = a.to_dict()

        self.is_fitted = True
        return self

    def _cyc(self, x: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
        # robust cyclic transform (sin, cos), ignoring NaN
        r = 2.0 * np.pi * (x / period)
        return np.sin(r), np.cos(r)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.datetime_cols_:
            return pd.DataFrame(index=X.index)

        out = {}
        gkey = self.group_key if (self.group_key in X.columns) else None

        for col in self.datetime_cols_:
            dt = self._coerce_dt(X[col])
            # basic date parts (Int64 to keep NaNs)
            out[f"{col}_year"] = dt.dt.year.astype("Int64")
            out[f"{col}_month"] = dt.dt.month.astype("Int64")
            out[f"{col}_day"] = dt.dt.day.astype("Int64")
            out[f"{col}_weekday"] = dt.dt.weekday.astype("Int64")
            out[f"{col}_quarter"] = dt.dt.quarter.astype("Int64")
            out[f"{col}_hour"] = dt.dt.hour.astype("Int64")
            out[f"{col}_dayofyear"] = dt.dt.dayofyear.astype("Int64")

            # ISO week is safer: weeks can be 1..53
            iso = dt.dt.isocalendar()
            out[f"{col}_isoweek"] = iso.week.astype("Int64")
            out[f"{col}_isoyear"] = iso.year.astype("Int64")

            if self.include_flags:
                out[f"{col}_is_weekend"] = (dt.dt.weekday >= 5).astype("Int64")
                out[f"{col}_is_month_start"] = dt.dt.is_month_start.astype("Int64")
                out[f"{col}_is_month_end"] = dt.dt.is_month_end.astype("Int64")
                out[f"{col}_is_quarter_start"] = dt.dt.is_quarter_start.astype("Int64")
                out[f"{col}_is_quarter_end"] = dt.dt.is_quarter_end.astype("Int64")
                out[f"{col}_is_year_start"] = dt.dt.is_year_start.astype("Int64")
                out[f"{col}_is_year_end"] = dt.dt.is_year_end.astype("Int64")

                if self._has_holidays:
                    # holiday flag by date in local calendar (assume UTC dates; if you have a tz column, localize before fit/transform)
                    dates = dt.dt.date.astype("object")
                    out[f"{col}_is_holiday"] = pd.Series(
                        [
                            d in self._holiday_set if pd.notnull(d) else pd.NA
                            for d in dates
                        ],
                        index=X.index,
                        dtype="Int64",
                    )

            # elapsed (fixed anchors learned in fit)
            if self.include_elapsed:
                if gkey and col in self._anchors_group_:
                    # per-row anchor from its group, fallback to global anchor
                    anchors = pd.Series(
                        X[gkey].map(self._anchors_group_[col]), index=X.index
                    )
                    anchors = anchors.fillna(self._anchors_global_[col])
                    elapsed = (dt - anchors).dt.total_seconds()
                else:
                    elapsed = (dt - self._anchors_global_[col]).dt.total_seconds()

                out[f"{col}_elapsed_seconds"] = pd.Series(
                    elapsed, index=X.index
                ).astype("Int64")
                # day resolution as well
                out[f"{col}_elapsed_days"] = (
                    out[f"{col}_elapsed_seconds"] // 86400
                ).astype("Int64")

            # cyclical encodings (DST-safe variants)
            if self.include_cyclical:
                # hour-of-week in [0, 167] is less aliasing than hour + weekday
                how = dt.dt.weekday * 24 + dt.dt.hour
                s, c = self._cyc(how.astype(float).values, period=24.0 * 7.0)
                out[f"{col}_hourweek_sin"] = s
                out[f"{col}_hourweek_cos"] = c

                # day-of-year with 365.2425 for leap year smoothness
                doy = dt.dt.dayofyear.astype(float).values
                s, c = self._cyc(doy, period=365.2425)
                out[f"{col}_dayofyear_sin"] = s
                out[f"{col}_dayofyear_cos"] = c

                # month-of-year (1..12)
                moy = dt.dt.month.astype(float).values
                s, c = self._cyc(moy, period=12.0)
                out[f"{col}_month_sin"] = s
                out[f"{col}_month_cos"] = c

                # ISO week-of-year (1..53)
                woy = iso.week.astype(float).values
                s, c = self._cyc(woy, period=53.0)
                out[f"{col}_isoweek_sin"] = s
                out[f"{col}_isoweek_cos"] = c

        return pd.DataFrame(out, index=X.index)


class MathematicalTransformer(BaseFeatureTransformer):
    """
    SOTA mathematical feature transformer:
      - Evaluates multiple monotone transforms and keeps those that improve shape.
      - Yeo–Johnson (default) robust to negatives/zeros; optional Box–Cox for positive-only.
      - Adds robust signed transforms (slog1p, asinh) for heavy-tailed / signed features.
    """

    def __init__(
        self,
        config: Any,
        method: str = "yeo-johnson",  # 'yeo-johnson' (default) or 'box-cox'
        min_variance: float = 1e-6,
        max_missing: float = 0.5,
        include_basic: bool = True,  # test log/sqrt/reciprocal/slog1p/asinh
        improvement_threshold: float = 0.1,  # lower score (better) by at least this to keep
        standardize: bool = False,  # optionally z-score after transform (per column)
        winsor_p: float = 0.0,  # e.g., 0.001 for 0.1% tail clipping
        eps: float = 1e-12,  # numerical guard
    ):
        super().__init__(config)
        self.method = method
        self.min_variance = float(min_variance)
        self.max_missing = float(max_missing)
        self.include_basic = bool(include_basic)
        self.improvement_threshold = float(improvement_threshold)
        self.standardize = bool(standardize)
        self.winsor_p = float(winsor_p)
        self.eps = float(eps)

        # Fitted artifacts
        self.power_transformers_: Dict[str, PowerTransformer] = {}  # per-col PT
        self.valid_transforms_: Dict[str, List[str]] = {}  # kept basic transforms
        self.valid_cols_power_: List[str] = []  # cols with PT kept
        self.col_medians_: Dict[str, float] = {}  # for imputation (no leakage)

        self.is_fitted = False

    # -------------------------
    # Helpers
    # -------------------------
    def _normality_score(self, arr: np.ndarray) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size < 20:
            # If too small, discourage adding noise: treat as "not improvable"
            return np.inf
        sk = stats.skew(vals, bias=False)
        ku = stats.kurtosis(vals, fisher=True, bias=False)
        return float(abs(sk) + 0.5 * abs(ku))

    def _winsorize(self, v: np.ndarray) -> np.ndarray:
        if self.winsor_p <= 0.0:
            return v
        a = v.copy()
        m = np.isfinite(a)
        if m.sum() < 20:
            return a
        lo = np.nanpercentile(a[m], 100 * self.winsor_p)
        hi = np.nanpercentile(a[m], 100 * (1 - self.winsor_p))
        a[m] = np.clip(a[m], lo, hi)
        return a

    def _safe_arr(self, s: pd.Series) -> np.ndarray:
        a = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float64)
        a[~np.isfinite(a)] = np.nan
        return a

    def _stdize(self, a: np.ndarray) -> np.ndarray:
        if not self.standardize:
            return a
        x = a.astype(np.float64, copy=True)
        m = np.nanmean(x)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd < self.eps:
            return a
        x = (x - m) / sd
        return x

    # Basic transforms (element-wise, monotone)
    def _t_logp(self, a: np.ndarray) -> np.ndarray:
        # log1p for non-negative part; negatives -> NaN
        x = np.where(a > 0, np.log1p(a), np.nan)
        return x

    def _t_sqrtp(self, a: np.ndarray) -> np.ndarray:
        x = np.where(a >= 0, np.sqrt(a), np.nan)
        return x

    def _t_recip(self, a: np.ndarray) -> np.ndarray:
        # safe reciprocal
        x = np.divide(
            1.0,
            a,
            out=np.full_like(a, np.nan),
            where=np.isfinite(a) & (np.abs(a) > self.eps),
        )
        return x

    def _t_slog1p(self, a: np.ndarray) -> np.ndarray:
        # signed log(1+|x|): robust and defined for all reals
        return np.sign(a) * np.log1p(np.abs(a))

    def _t_asinh(self, a: np.ndarray) -> np.ndarray:
        # arcsinh: similar to signed log for large |x|, linear near 0
        return np.arcsinh(a)

    def _candidate_basic(self, col: str, data: pd.Series) -> Dict[str, np.ndarray]:
        """Return dict of candidate basic transforms (raw arrays, not standardized)."""
        a = self._safe_arr(data)
        a = self._winsorize(a)
        cands = {"identity": a}
        if not self.include_basic:
            return cands
        cands.update(
            {
                "log": self._t_logp(a),
                "sqrt": self._t_sqrtp(a),
                "reciprocal": self._t_recip(a),
                "slog1p": self._t_slog1p(a),
                "asinh": self._t_asinh(a),
            }
        )
        return cands

    # -------------------------
    # Fit
    # -------------------------
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "MathematicalTransformer":
        if not getattr(self.config, "create_math_features", True):
            self.is_fitted = True
            return self

        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        self.valid_transforms_.clear()
        self.power_transformers_.clear()
        self.valid_cols_power_.clear()
        self.col_medians_.clear()

        for col in numerical_cols:
            if col not in X.columns:
                continue
            series = X[col]
            miss = series.isna().mean()
            if miss > self.max_missing:
                continue

            data = series.dropna()
            if data.size < 10 or float(np.nanvar(data)) < self.min_variance:
                continue

            self.col_medians_[col] = float(series.median(skipna=True))

            # ===== basic transforms: keep those that improve score by threshold =====
            cands = self._candidate_basic(col, data)
            base_score = self._normality_score(cands["identity"])

            kept: List[str] = []
            for name, arr in cands.items():
                if name == "identity":
                    continue
                if np.nanvar(arr) <= self.min_variance:
                    continue
                score = self._normality_score(arr)
                if (score + self.improvement_threshold) < base_score:
                    kept.append(name)
            if kept:
                self.valid_transforms_[col] = kept

            # ===== power transformer (YJ or BC): include only if improves score =====
            if self.method == "box-cox" and (data <= 0).any():
                # BC not valid for non-positive; skip cleanly
                pass
            else:
                try:
                    pt = PowerTransformer(method=self.method, standardize=False)
                    # winsorize before fitting to avoid unstable λ
                    a_fit = self._winsorize(self._safe_arr(data))
                    pt.fit(a_fit.reshape(-1, 1))
                    transformed = pt.transform(a_fit.reshape(-1, 1)).ravel()
                    if np.nanvar(transformed) > self.min_variance:
                        if (
                            self._normality_score(transformed)
                            + self.improvement_threshold
                        ) < base_score:
                            self.power_transformers_[col] = pt
                            self.valid_cols_power_.append(col)
                except Exception as e:
                    warnings.warn(f"[MathTF] Skipping {col} {self.method}: {e}")

        self.is_fitted = True
        return self

    # -------------------------
    # Transform
    # -------------------------
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not getattr(self.config, "create_math_features", True):
            return pd.DataFrame(index=X.index)

        out: Dict[str, np.ndarray] = {}

        # apply basic transforms
        for col, names in self.valid_transforms_.items():
            if col not in X.columns:
                continue
            s = X[col].copy()
            # impute like we did for PT, to keep behavior stable
            s = s.fillna(self.col_medians_.get(col, float(s.median(skipna=True))))
            a = self._winsorize(self._safe_arr(s))

            for name in names:
                if name == "log":
                    t = self._t_logp(a)
                elif name == "sqrt":
                    t = self._t_sqrtp(a)
                elif name == "reciprocal":
                    t = self._t_recip(a)
                elif name == "slog1p":
                    t = self._t_slog1p(a)
                elif name == "asinh":
                    t = self._t_asinh(a)
                else:
                    continue

                t = self._stdize(t).astype(np.float32)
                out[f"{col}_{name}"] = t

        # apply power transforms (yeo-johnson / box-cox)
        for col in self.valid_cols_power_:
            if col not in X.columns:
                continue
            s = X[col].fillna(
                self.col_medians_.get(col, float(X[col].median(skipna=True)))
            )
            a = self._winsorize(self._safe_arr(s))
            try:
                t = self.power_transformers_[col].transform(a.reshape(-1, 1)).ravel()
                t = self._stdize(t).astype(np.float32)
                out[f"{col}_{self.method.replace('-', '_')}"] = t
            except Exception:
                # if transform fails at inference for any reason, skip gracefully
                continue

        return pd.DataFrame(out, index=X.index)


# Optional parallelism
try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False


try:
    from joblib import Parallel, delayed

    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False


class InteractionTransformer(BaseFeatureTransformer):
    """
    Turbo interaction & polynomial generator (improved, AdaptiveMI-compatible):
      - smarter pair pre-screen: variance × (1 - |corr|) heuristic
      - commutative op canonicalization (no duplicate A∘B vs B∘A)
      - light winsorization + robust _clean guardrails
      - cached arrays + train means (no leakage)
      - parallel scoring (joblib) with safe fallback
      - recipe-based transform (compute only selected)
      - optional redundancy pruning (rank-Pearson)
    """

    def __init__(self, config: Any):
        super().__init__(config)

        # discovered during fit
        self.numerical_cols_: List[str] = []
        self.col_means_: Dict[str, float] = {}

        # recipes
        self.selected_interactions_: List[Tuple[str, str, str]] = []  # (col1, op, col2)
        self.selected_polynomials_: List[Tuple[str, Union[float, str]]] = (
            []
        )  # (col, power)
        self.feature_scores_: Dict[str, float] = {}

        # speed/quality knobs
        self.max_interactions = getattr(config, "max_interactions", 100)
        self.max_polynomials = getattr(config, "max_polynomials", 50)
        self.max_pairs = getattr(config, "max_pairs_screen", 800)
        self.prescreen_topk = getattr(config, "interaction_prescreen_topk", 32)
        self.min_variance = getattr(config, "min_variance_threshold", 1e-6)
        self.redundancy_corr = getattr(config, "interaction_redundancy_corr", 0.985)
        self.enable_redundancy_prune = getattr(
            config, "interaction_prune_redundancy", False
        )

        self.random_state = getattr(config, "random_state", 42)
        self.eps = 1e-8

        # Turbo controls
        self.fast_mode = getattr(config, "interaction_fast_mode", True)
        self.row_subsample = getattr(config, "interaction_row_subsample", 5000)
        self.n_jobs = getattr(config, "interaction_n_jobs", -1)
        self.prescreen_use_spearman = getattr(
            config, "interaction_prescreen_spearman", False
        )

        # operations (meta uses train means at transform)
        self.operations = {
            "sum": (
                getattr(config, "include_sum", True),
                lambda a, b, meta=None: a + b,
            ),
            "diff": (
                getattr(config, "include_diff", True),
                lambda a, b, meta=None: a - b,
            ),
            "prod": (
                getattr(config, "include_prod", True),
                lambda a, b, meta=None: a * b,
            ),
            "ratio": (
                getattr(config, "include_ratio", True),
                lambda a, b, meta=None: np.divide(
                    a,
                    b,
                    out=np.full_like(a, np.nan),
                    where=(np.abs(b) > self.eps) & np.isfinite(b),
                ),
            ),
            "min": (getattr(config, "include_minmax", True), np.minimum),
            "max": (getattr(config, "include_minmax", True), np.maximum),
            "zdiff": (
                getattr(config, "include_zdiff", True),
                lambda a, b, meta=None: (
                    a - (meta["a_mean"] if meta else np.nanmean(a))
                )
                - (b - (meta["b_mean"] if meta else np.nanmean(b))),
            ),
            "log_ratio": (
                getattr(config, "include_logratio", True),
                lambda a, b, meta=None: np.log1p(np.abs(a) + self.eps)
                - np.log1p(np.abs(b) + self.eps),
            ),
            "root_prod": (
                getattr(config, "include_rootprod", True),
                lambda a, b, meta=None: np.sign(a * b) * np.sqrt(np.abs(a * b)),
            ),
        }
        # mark commutative ops to canonicalize (avoid A∘B and B∘A duplicates)
        self._commutative_ops = {"sum", "prod", "min", "max", "root_prod"}

        # polynomials
        self.powers = {
            "squared": (getattr(config, "include_square", True), 2.0),
            "sqrt": (getattr(config, "include_sqrt", True), 0.5),
            "cubed": (getattr(config, "include_cube", False), 3.0),
            "reciprocal": (getattr(config, "include_reciprocal", False), -1.0),
            "log": (getattr(config, "include_log", False), "log"),
        }

        # scorer (unchanged to preserve AdaptiveMI behavior)
        self.ami_scorer = AdaptiveMI(
            subsample=min(getattr(config, "max_rows_score", 2000), 2000),
            spearman_gate=getattr(config, "mi_spearman_gate", 0.05),
            min_overlap=getattr(config, "mi_min_overlap", 50),
            ks=(3, 5, 10),
            n_bins=getattr(config, "mi_bins", 16),
            random_state=self.random_state,
        )

        # light winsorization for stability (percentile caps)
        self._winsor_p = float(
            getattr(config, "interaction_winsor_p", 0.001)
        )  # 0.1% tails by default

    # ---------------- utils ----------------

    def _winsorize(self, arr: np.ndarray) -> np.ndarray:
        """Light symmetric winsorization to reduce extreme tails (keeps AMI stable)."""
        a = arr.copy()
        finite = np.isfinite(a)
        if finite.sum() < 20:
            return a
        lo = np.nanpercentile(a[finite], 100 * self._winsor_p)
        hi = np.nanpercentile(a[finite], 100 * (1 - self._winsor_p))
        a[finite] = np.clip(a[finite], lo, hi)
        return a

    def _clean(self, arr: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(arr, dtype=np.float32)
        # sanitize
        arr[~np.isfinite(arr)] = np.nan
        arr = np.where(np.abs(arr) > 1e10, np.nan, arr)
        # winsorize finite values
        arr = self._winsorize(arr)
        # guardrails
        finite = np.isfinite(arr)
        if finite.sum() < 10:
            return None
        if np.nanvar(arr) < self.min_variance:
            return None
        return arr

    def _poly(self, arr: np.ndarray, power: Union[float, str]) -> np.ndarray:
        with np.errstate(all="ignore"):
            if power == 0.5:
                out = np.where(arr >= 0, np.sqrt(arr), np.nan)
            elif power == "log":
                out = np.where(arr > 0, np.log1p(arr), np.nan)
            elif power == -1.0:
                out = np.divide(
                    1.0,
                    arr,
                    out=np.full_like(arr, np.nan),
                    where=(np.abs(arr) > self.eps) & np.isfinite(arr),
                )
            else:
                out = np.power(arr, power)
        return np.clip(out, -1e10, 1e10)

    def _score(self, arr: np.ndarray, y: Optional[np.ndarray]) -> float:
        vals = arr[np.isfinite(arr)]
        if vals.size < 20:
            return 0.0
        if y is None:
            # unsupervised fallback: dispersion score
            m, s = np.mean(vals), np.std(vals)
            cv = s / abs(m) if abs(m) > self.eps else s
            rng = (np.max(vals) - np.min(vals)) / (s + 1e-8)
            return float(max(0.0, 0.7 * cv + 0.3 * rng))
        mask = np.isfinite(arr) & np.isfinite(y)
        if mask.sum() < 20:
            return 0.0
        try:
            # keep AMI call intact
            return float(
                max(
                    0.0,
                    self.ami_scorer.score_pairwise(arr[mask].reshape(-1, 1), y[mask])[
                        0
                    ],
                )
            )
        except Exception:
            return 0.0

    def _subsample_idx(self, n_rows: int) -> np.ndarray:
        if not self.fast_mode or self.row_subsample is None:
            return np.arange(n_rows)
        k = min(int(self.row_subsample), n_rows)
        rng = np.random.RandomState(self.random_state)
        return np.sort(rng.choice(n_rows, size=k, replace=False))

    def _robust_var(self, x: np.ndarray) -> float:
        # IQR-based variance proxy to resist outliers
        x = x[np.isfinite(x)]
        if x.size < 5:
            return 0.0
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        return float(iqr * iqr)

    def _pair_heuristic(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Heuristic to rank column pairs before expensive feature gen/AMI:
          score = robust_var(a)*robust_var(b) * (1 - |corr|)
        Encourages informative-but-not-collinear pairs.
        """
        mask = np.isfinite(a) & np.isfinite(b)
        if mask.sum() < 25:
            return 0.0
        aa, bb = a[mask], b[mask]
        va = self._robust_var(aa)
        vb = self._robust_var(bb)
        if va <= 0 or vb <= 0:
            return 0.0
        sa = aa.std()
        sb = bb.std()
        corr = 0.0
        if sa > 1e-12 and sb > 1e-12:
            corr = float(np.corrcoef(aa, bb)[0, 1])
            if not np.isfinite(corr):
                corr = 0.0
        return float((1.0 - abs(corr)) * va * vb)

    def _prescreen_columns(self, X: pd.DataFrame, y: Optional[pd.Series]) -> List[str]:
        cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not cols:
            return []
        # cheap variance-based prescreen
        scores: Dict[str, float] = {}
        for c in cols:
            vals = pd.to_numeric(X[c], errors="coerce").to_numpy(dtype=float)
            scores[c] = float(np.nanvar(vals))
        # optional supervised bump
        if self.prescreen_use_spearman and (y is not None):
            ys = pd.Series(y)
            for c in cols:
                s = pd.Series(X[c])
                m = s.notna() & ys.notna()
                if m.sum() >= 25:
                    rho = s[m].rank().corr(ys[m].rank())
                    if pd.notna(rho):
                        scores[c] += abs(float(rho))
        topk = min(self.prescreen_topk, len(cols))
        return [
            c
            for c, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[
                :topk
            ]
        ]

    # --------- candidate generation (fit-only, with subsample) ---------

    def _canonical_pair(self, c1: str, c2: str, op_name: str) -> Tuple[str, str, str]:
        """For commutative ops, sort column names so A∘B == B∘A."""
        if op_name in self._commutative_ops:
            if c2 < c1:
                c1, c2 = c2, c1
        return c1, op_name, c2

    def _screen_pairs(
        self, pair_cols: List[str], cache: Dict[str, np.ndarray]
    ) -> List[Tuple[str, str]]:
        """Rank pairs using the heuristic and keep up to max_pairs."""
        if len(pair_cols) < 2:
            return []
        scores = []
        for c1, c2 in combinations(pair_cols, 2):
            a, b = cache[c1], cache[c2]
            s = self._pair_heuristic(a, b)
            if s > 0:
                scores.append((s, c1, c2))
        scores.sort(reverse=True, key=lambda t: t[0])
        keep = scores[: self.max_pairs]
        return [(c1, c2) for _, c1, c2 in keep]

    def _generate_interactions_fit(
        self, pair_cols: List[str], cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        out = {}
        if len(pair_cols) < 2:
            return out

        # rank pairs first, to reduce unnecessary feature computations
        ranked_pairs = self._screen_pairs(pair_cols, cache)
        for c1, c2 in ranked_pairs:
            a = cache[c1]
            b = cache[c2]
            # train means for zdiff
            self.col_means_.setdefault(c1, float(np.nanmean(a)))
            self.col_means_.setdefault(c2, float(np.nanmean(b)))
            meta = {"a_mean": self.col_means_[c1], "b_mean": self.col_means_[c2]}

            for op_name, (flag, func) in self.operations.items():
                if not flag:
                    continue
                cc1, opn, cc2 = self._canonical_pair(c1, c2, op_name)
                key = f"{cc1}__{opn}__{cc2}"
                if key in out:
                    continue
                try:
                    feat = func(a, b, meta)
                except TypeError:
                    feat = func(a, b)
                feat = self._clean(feat)
                if feat is not None:
                    out[key] = feat
        return out

    def _generate_polynomials_fit(
        self, cols: List[str], cache: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        out = {}
        for c in cols:
            arr = cache[c]
            self.col_means_.setdefault(c, float(np.nanmean(arr)))
            for suffix, (flag, power) in self.powers.items():
                if not flag:
                    continue
                feat = self._poly(arr, power)
                feat = self._clean(feat)
                if feat is not None:
                    out[f"{c}__{suffix}"] = feat
        return out

    # ---------------- scoring ----------------

    def _score_dict(
        self, feats: Dict[str, np.ndarray], y_arr: Optional[np.ndarray]
    ) -> Dict[str, float]:
        names = list(feats.keys())
        arrays = [feats[n] for n in names]
        if _HAVE_JOBLIB and (self.n_jobs is not None) and (self.n_jobs != 0):
            n_jobs = self.n_jobs
            scores = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(self._score)(arr, y_arr) for arr in arrays
            )
        else:
            scores = [self._score(arr, y_arr) for arr in arrays]
        return {n: s for n, s in zip(names, scores)}

    def _select_topk(
        self, feats: Dict[str, np.ndarray], y_arr: Optional[np.ndarray], k: int
    ) -> List[str]:
        if not feats or k <= 0:
            return []
        scores = self._score_dict(feats, y_arr)
        self.feature_scores_.update(scores)
        ordered = [
            n
            for n, s in sorted(scores.items(), key=lambda x: x[1], reverse=True)
            if s > 0
        ]
        return ordered[:k]

    # ---------------- sklearn API ----------------

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "InteractionTransformer":
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        if not self.numerical_cols_:
            self.is_fitted = True
            return self

        # subsample rows once
        idx = self._subsample_idx(len(X))
        X_sub = X.iloc[idx, :]

        # cache numeric columns as float arrays (subsampled)
        cache: Dict[str, np.ndarray] = {}
        for c in self.numerical_cols_:
            cache[c] = pd.to_numeric(X_sub[c], errors="coerce").to_numpy(
                dtype=np.float32
            )

        y_arr = None
        if y is not None:
            y_arr = pd.Series(y).iloc[idx].to_numpy(dtype=float)
            y_arr[~np.isfinite(y_arr)] = np.nan  # AMI handles mask

        # prescreen columns (on full X; cheap) and then pair-rank on subsample cache
        pair_cols = self._prescreen_columns(X, y)

        # generate candidates on subsample & cache
        cand_inter, cand_poly = {}, {}
        if getattr(self.config, "create_interactions", True):
            cand_inter = self._generate_interactions_fit(pair_cols, cache)
        if getattr(self.config, "create_polynomials", True):
            cand_poly = self._generate_polynomials_fit(self.numerical_cols_, cache)

        # optional redundancy prune among candidates
        if self.enable_redundancy_prune:
            cand_inter = self._fast_redundancy_prune(cand_inter)
            cand_poly = self._fast_redundancy_prune(cand_poly)

        # select
        sel_inter_names = self._select_topk(cand_inter, y_arr, self.max_interactions)
        sel_poly_names = self._select_topk(cand_poly, y_arr, self.max_polynomials)

        # to recipes
        self.selected_interactions_ = []
        for nm in sel_inter_names:
            c1, op_name, c2 = nm.split("__", 2)
            self.selected_interactions_.append((c1, op_name, c2))

        self.selected_polynomials_ = []
        for nm in sel_poly_names:
            col, suffix = nm.split("__", 1)
            power = self.powers[suffix][1] if suffix in self.powers else None
            if power is not None:
                self.selected_polynomials_.append((col, power))

        self.is_fitted = True
        return self

    # faster streaming redundancy prune (Spearman-ish via Pearson on ranks; stop early)
    def _fast_redundancy_prune(
        self, feats: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if not feats:
            return feats
        names = list(feats.keys())
        kept = []
        # pre-rank to approximate Spearman cheaply
        ranked = {}
        for n in names:
            v = feats[n]
            mask = np.isfinite(v)
            r = np.full_like(v, np.nan, dtype=np.float32)
            # rank only finite elements
            r[mask] = np.argsort(np.argsort(v[mask])).astype(np.float32)
            ranked[n] = r

        for n in names:
            r = ranked[n]
            drop = False
            for k in kept:
                rk = ranked[k]
                m = np.isfinite(r) & np.isfinite(rk)
                if m.sum() < 25:
                    continue
                # Pearson on ranks ~= Spearman
                a = r[m]
                b = rk[m]
                sa = a.std()
                sb = b.std()
                if sa < 1e-12 or sb < 1e-12:
                    continue
                corr = float(np.corrcoef(a, b)[0, 1])
                if abs(corr) >= self.redundancy_corr:
                    drop = True
                    break
            if not drop:
                kept.append(n)
        return {k: feats[k] for k in kept}

    def _compute_interaction(
        self, X: pd.DataFrame, c1: str, op_name: str, c2: str
    ) -> Optional[np.ndarray]:
        # canonicalize at transform too, to match recipe naming
        if op_name in self._commutative_ops and c2 < c1:
            c1, c2 = c2, c1
        a = pd.to_numeric(X[c1], errors="coerce").to_numpy(dtype=np.float32)
        b = pd.to_numeric(X[c2], errors="coerce").to_numpy(dtype=np.float32)
        meta = {
            "a_mean": self.col_means_.get(c1, float(np.nanmean(a))),
            "b_mean": self.col_means_.get(c2, float(np.nanmean(b))),
        }
        func = self.operations[op_name][1]
        try:
            feat = func(a, b, meta)
        except TypeError:
            feat = func(a, b)
        return self._clean(feat)

    def _compute_polynomial(
        self, X: pd.DataFrame, col: str, power: Union[float, str]
    ) -> Optional[np.ndarray]:
        arr = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=np.float32)
        return self._clean(self._poly(arr, power))

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            return pd.DataFrame(index=X.index)

        feats: Dict[str, np.ndarray] = {}

        for c1, op_name, c2 in self.selected_interactions_:
            if c1 in X.columns and c2 in X.columns:
                arr = self._compute_interaction(X, c1, op_name, c2)
                if arr is not None:
                    # keep canonical name
                    if op_name in self._commutative_ops and c2 < c1:
                        c1, c2 = c2, c1
                    feats[f"{c1}__{op_name}__{c2}"] = arr

        for col, power in self.selected_polynomials_:
            if col in X.columns:
                arr = self._compute_polynomial(X, col, power)
                if arr is not None:
                    suffix = next(
                        (k for k, (_, p) in self.powers.items() if p == power),
                        str(power),
                    )
                    feats[f"{col}__{suffix}"] = arr

        return (
            pd.DataFrame(feats, index=X.index, dtype=np.float32)
            if feats
            else pd.DataFrame(index=X.index)
        )


class StatisticalTransformer(BaseFeatureTransformer):
    """Creates statistical aggregation features across rows."""

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "StatisticalTransformer":
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_statistical or len(self.numerical_cols_) < 2:
            return pd.DataFrame(index=X.index)

        available_cols = [col for col in self.numerical_cols_ if col in X.columns]
        if len(available_cols) < 2:
            return pd.DataFrame(index=X.index)

        X_stats = X[available_cols]

        # Compute all stats in one pass where possible
        features = {}
        features["row_mean"] = X_stats.mean(axis=1)
        features["row_std"] = X_stats.std(axis=1)
        features["row_min"] = X_stats.min(axis=1)
        features["row_max"] = X_stats.max(axis=1)
        features["row_range"] = features["row_max"] - features["row_min"]  # moved here
        features["row_median"] = X_stats.median(axis=1)
        features["row_skew"] = X_stats.skew(axis=1)

        # Vectorized null handling
        null_mask = X_stats.isna()
        features["row_non_null_count"] = (~null_mask).sum(axis=1)
        features["row_null_ratio"] = null_mask.sum(axis=1) / len(available_cols)

        return pd.DataFrame(features, index=X.index)


class CategoricalTransformer(BaseFeatureTransformer):
    """
    Modern categorical transformer:
      - Strategies: 'auto', 'onehot', 'freq', 'hashing', 'target_kfold', 'ordinal'
      - Leakage-safe target encoding (k-fold, with prior smoothing)
      - Robust rare handling: frequency threshold, min count, top-k cap
      - Stable output schema across fit/transform
    """

    def __init__(
        self,
        config,
        strategies: Tuple[str, ...] = ("auto",),
        rare_threshold: Optional[
            float
        ] = None,  # fraction (0..1); falls back to config.rare_threshold
        min_count: int = 1,  # absolute count for rare categories
        top_k: Optional[int] = None,  # keep top_k most frequent; others -> OTHER
        hashing_dim: int = 64,
        n_splits: int = 5,
        target_min_samples: int = 100,  # only consider target encoding if enough data
        smoothing_prior: float = 10.0,  # target encoding prior weight
        random_state: int = 42,
    ):
        super().__init__(config)
        self.strategies = strategies
        self.rare_threshold = (
            rare_threshold
            if rare_threshold is not None
            else getattr(config, "rare_threshold", 0.01)
        )
        self.min_count = int(min_count)
        self.top_k = top_k or getattr(config, "cat_top_k", None)
        self.hashing_dim = int(hashing_dim)
        self.n_splits = int(n_splits)
        self.target_min_samples = int(target_min_samples)
        self.smoothing_prior = float(smoothing_prior)
        self.random_state = int(random_state)

        self.categorical_cols_: List[str] = []
        self.col_info_: Dict[str, Dict[str, Any]] = {}  # per-col strategy+artifacts
        self.is_fitted = False

    # -------------------------- utils --------------------------

    @staticmethod
    def _as_str_series(s: pd.Series) -> pd.Series:
        return s.astype(str).fillna("MISSING").replace({"": "MISSING"})

    def _apply_rare_policy(self, s: pd.Series) -> Tuple[pd.Series, List[str]]:
        vc = s.value_counts(dropna=False)
        freq = vc / vc.sum()
        rare = set()

        # threshold by relative freq and absolute count
        rare |= set(freq[freq < self.rare_threshold].index)
        rare |= set(vc[vc < self.min_count].index)

        # top-k cap if requested
        if self.top_k is not None and self.top_k > 0 and len(vc) > self.top_k:
            keep = set(vc.nlargest(self.top_k).index)
            rare |= set(vc.index.difference(keep))

        if len(vc) - len(rare) < 2:
            rare = set()  # avoid collapsing almost all to OTHER

        if rare:
            s = s.where(~s.isin(rare), "OTHER")

        return s, sorted(map(str, rare))

    def _choose_auto_strategy(self, s: pd.Series, y: Optional[pd.Series]) -> str:
        k = s.nunique(dropna=False)
        n = len(s)

        # If target is available and enough samples + medium/high cardinality,
        # consider target encoding.
        if y is not None and n >= self.target_min_samples and 10 < k < 10000:
            try:
                if y.nunique() <= 20:  # classification
                    score = chi2(
                        pd.get_dummies(
                            s, drop_first=True, sparse=False, dtype=np.uint8
                        ),
                        y,
                    )[0].mean()
                else:  # regression
                    # quick Ordinal for score proxy (safe for screening only)
                    oe = OrdinalEncoder(
                        handle_unknown="use_encoded_value", unknown_value=-1
                    )
                    Xc = oe.fit_transform(s.to_frame())
                    score = f_regression(Xc, y.values)[0].mean()
                if np.isfinite(score) and score > 0:
                    return "target_kfold"
            except Exception:
                pass

        # Low cardinality → onehot
        if k <= 12:
            return "onehot"
        # Medium → frequency
        if k <= 1000:
            return "freq"
        # Very high → hashing
        return "hashing"

    # -------------------------- fit --------------------------

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
        self.categorical_cols_ = cats
        self.col_info_.clear()

        for col in cats:
            s = self._as_str_series(X[col])
            s, rare_list = self._apply_rare_policy(s)
            n_unique = s.nunique(dropna=False)

            # pick strategy
            if "auto" in self.strategies:
                strategy = self._choose_auto_strategy(s, y)
            else:
                strategy = self.strategies[0]

            info = {"strategy": strategy, "rare": rare_list}

            if strategy == "onehot":
                enc = OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,  # sklearn >=1.2; if older, use sparse=False
                    dtype=np.uint8,
                )
                enc.fit(s.to_frame())
                feat_names = enc.get_feature_names_out([col]).tolist()
                info.update(encoder=enc, feature_names=feat_names)

            elif strategy == "ordinal":
                # more robust than LabelEncoder for unseen categories
                enc = OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    dtype=np.int64,
                )
                enc.fit(s.to_frame())
                cats_known = [list(map(str, c)) for c in enc.categories_]
                info.update(
                    encoder=enc, categories=cats_known, feature_names=[f"{col}_ord"]
                )

            elif strategy == "freq":
                vc = s.value_counts(normalize=True, dropna=False)
                mapping = vc.to_dict()
                info.update(mapping=mapping, feature_names=[f"{col}_freq"])

            elif strategy == "hashing":
                # Stable column names for hashed dims
                n_feat = min(self.hashing_dim, max(2, n_unique))
                enc = FeatureHasher(
                    n_features=n_feat, input_type="string", alternate_sign=False
                )
                info.update(
                    encoder=enc,
                    n_features=n_feat,
                    feature_names=[f"{col}_hash_{i}" for i in range(n_feat)],
                )

            elif strategy == "target_kfold" and y is not None:
                # store per-category stats for smoothing + kfold plan
                y_series = y.reindex(X.index)
                global_mean = float(np.nanmean(y_series.values))
                info.update(global_mean=global_mean)
                # Fit-time we only keep category means and counts
                grp = pd.DataFrame({"cat": s, "y": y_series}).groupby("cat")["y"]
                cat_mean = grp.mean().to_dict()
                cat_count = grp.size().to_dict()
                info.update(
                    cat_mean=cat_mean, cat_count=cat_count, feature_names=[f"{col}_te"]
                )
            else:
                # fallback
                vc = s.value_counts(normalize=True, dropna=False)
                mapping = vc.to_dict()
                info.update(mapping=mapping, feature_names=[f"{col}_freq"])

            self.col_info_[col] = info

        self.is_fitted = True
        return self

    # -------------------------- transform --------------------------

    def _target_kfold_transform(
        self,
        s: pd.Series,
        y: Optional[pd.Series],
        info: Dict[str, Any],
        index: pd.Index,
    ):
        # If y is provided at transform (train), do KFold out-of-fold encoding.
        # If not (inference), use smoothed mean from fit stats.
        name = info["feature_names"][0]
        prior = info["global_mean"]
        alpha = self.smoothing_prior

        if y is not None:
            y_series = y.reindex(index)
            kf = KFold(
                n_splits=min(self.n_splits, max(2, y_series.notna().sum())),
                shuffle=True,
                random_state=self.random_state,
            )
            out = pd.Series(index=index, dtype=float)
            for tr_idx, te_idx in kf.split(s):
                tr_c = s.iloc[tr_idx]
                tr_y = y_series.iloc[tr_idx]
                stats = (
                    pd.DataFrame({"cat": tr_c, "y": tr_y})
                    .groupby("cat")["y"]
                    .agg(["mean", "count"])
                )
                # smoothing
                smoothed = (stats["count"] * stats["mean"] + alpha * prior) / (
                    stats["count"] + alpha
                )
                enc_map = smoothed.to_dict()
                out.iloc[te_idx] = s.iloc[te_idx].map(enc_map).fillna(prior).values
            return pd.DataFrame({name: out.values}, index=index)
        else:
            # test-time: use global fit stats with smoothing
            cat_mean = info["cat_mean"]
            cat_count = info["cat_count"]
            vals = []
            for v in s.values:
                m = cat_mean.get(v, prior)
                c = cat_count.get(v, 0)
                vals.append((c * m + alpha * prior) / (c + alpha))
            return pd.DataFrame({name: vals}, index=index)

    def transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        if not self.categorical_cols_:
            return pd.DataFrame(index=X.index)

        outputs = []
        for col in self.categorical_cols_:
            info = self.col_info_.get(col)
            if info is None:
                continue

            s = self._as_str_series(X[col])

            # replicate rare policy from fit (ANY category in `rare` -> OTHER)
            rare = set(info.get("rare", []))
            if rare:
                s = s.where(~s.isin(rare), "OTHER")

            strategy = info["strategy"]

            if strategy == "onehot":
                enc = info["encoder"]
                arr = enc.transform(s.to_frame())
                df = pd.DataFrame(arr, columns=info["feature_names"], index=X.index)
                outputs.append(df.astype(np.uint8))

            elif strategy == "ordinal":
                enc = info["encoder"]
                arr = enc.transform(s.to_frame())
                outputs.append(
                    pd.DataFrame(
                        {info["feature_names"][0]: arr.ravel()}, index=X.index
                    ).astype(np.int64)
                )

            elif strategy == "freq":
                mapping = info["mapping"]
                vals = s.map(mapping).fillna(0.0).astype(float)
                outputs.append(
                    pd.DataFrame({info["feature_names"][0]: vals}, index=X.index)
                )

            elif strategy == "hashing":
                enc = info["encoder"]
                mat = enc.transform(s.tolist())  # input_type="string"
                # densify with fixed column names
                df = pd.DataFrame(
                    mat.toarray(), columns=info["feature_names"], index=X.index
                )
                outputs.append(df)

            elif strategy == "target_kfold":
                df = self._target_kfold_transform(s, y, info, X.index)
                outputs.append(df.astype(float))

            else:  # fallback to frequency
                mapping = info.get("mapping", {})
                vals = s.map(mapping).fillna(0.0).astype(float)
                outputs.append(pd.DataFrame({f"{col}_freq": vals}, index=X.index))

        return pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=X.index)


class BinningTransformer(BaseFeatureTransformer):
    """
    Adaptive, clean, SOTA-ish binning for numerical features (no Bayesian Blocks).
    Strategies:
      - 'fd'           : Freedman–Diaconis (IQR-based)
      - 'doane'        : Doane's rule (skew-adjusted Sturges)
      - 'shimazaki'    : Shimazaki–Shinomoto optimal bin count (equal-width)
      - 'kmeans'       : 1D K-Means edges (handles multi-modality)
      - 'quantile'     : mass-balanced via KBinsDiscretizer
      - 'uniform'      : equal-width via KBinsDiscretizer
      - 'mdlp'         : supervised Fayyad–Irani MDL (requires y)
    """

    def __init__(
        self,
        config,
        min_samples_per_bin: int = 10,
        max_bins: int = 100,
        strategies: Optional[List[str]] = None,
        kmeans_bins: int = 10,
        shimazaki_k_range: Tuple[int, int] = (4, 128),
        random_state: int = 42,
    ):
        super().__init__(config)
        self.min_samples_per_bin = int(min_samples_per_bin)
        self.max_bins = int(max_bins)
        self.kmeans_bins = int(kmeans_bins)
        self.shimazaki_k_range = shimazaki_k_range
        self.random_state = int(random_state)

        self.strategies = (
            strategies
            if strategies is not None
            else getattr(self.config, "binning_strategies", ["fd"])  # unsupervised
        )

        self.binning_transformers_: Dict[str, Dict[str, Any]] = {}
        self.numerical_cols_: List[str] = []
        self.is_fitted = False

    # ---------- helpers: bin-count rules -> edges ----------

    @staticmethod
    def _finite(x: np.ndarray) -> np.ndarray:
        return x[np.isfinite(x)]

    def _fd_bins(self, x: np.ndarray) -> int:
        x = self._finite(x)
        n = x.size
        if n <= 1:
            return 1
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr <= 0:
            return min(self.max_bins, max(1, int(np.sqrt(n))))
        h = 2 * iqr * (n ** (-1 / 3))
        if h <= 0:
            return min(self.max_bins, max(1, int(np.sqrt(n))))
        k = int(np.ceil((x.max() - x.min()) / h))
        return int(np.clip(k, 1, self.max_bins))

    def _doane_bins(self, x: np.ndarray) -> int:
        x = self._finite(x)
        n = x.size
        if n <= 1:
            return 1
        g1 = stats.skew(x, bias=False)
        sigma_g1 = np.sqrt((6 * (n - 2)) / ((n + 1) * (n + 3)))
        k = 1 + np.log2(n) + np.log2(1 + abs(g1) / (sigma_g1 + 1e-12))
        return int(np.clip(int(np.round(k)), 1, self.max_bins))

    def _shimazaki_bins(self, x: np.ndarray, kmin=4, kmax=128) -> int:
        # Shimazaki & Shinomoto (2007): minimize cost(k) = (2m - v) / h^2
        x = self._finite(x)
        n = x.size
        if n <= 1:
            return 1
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-12:
            return 1
        best_k, best_cost = 1, np.inf
        kmin = max(2, int(kmin))
        kmax = int(min(self.max_bins, max(kmin + 1, kmax)))
        for k in range(kmin, kmax + 1):
            edges = np.linspace(xmin, xmax, k + 1)
            counts, _ = np.histogram(x, bins=edges)
            h = (xmax - xmin) / k
            m = counts.mean()
            v = counts.var()
            cost = (2 * m - v) / (h**2 + 1e-12)
            if cost < best_cost:
                best_cost = cost
                best_k = k
        return int(np.clip(best_k, 1, self.max_bins))

    def _kmeans_edges(self, x: np.ndarray, k: int) -> np.ndarray:
        x = self._finite(x).reshape(-1, 1)
        if x.size == 0:
            return np.array([0.0, 1.0])
        k = int(np.clip(k, 2, min(self.max_bins, max(2, x.shape[0]))))
        km = KMeans(n_clusters=k, n_init="auto", random_state=self.random_state)
        km.fit(x)
        centers = np.sort(km.cluster_centers_.flatten())
        # edges at midpoints between sorted centers; extend to cover range
        inner = (centers[1:] + centers[:-1]) / 2
        left = x.min() - 1e-12
        right = x.max() + 1e-12
        edges = np.concatenate([[left], inner, [right]])
        return edges

    def _equal_width_edges(self, x: np.ndarray, k: int) -> np.ndarray:
        x = self._finite(x)
        if x.size <= 1:
            return np.array([0.0, 1.0])
        k = int(np.clip(k, 1, self.max_bins))
        xmin, xmax = x.min(), x.max()
        if xmax - xmin < 1e-12:
            return np.array([xmin, xmax + 1.0])
        return np.linspace(xmin, xmax, k + 1)

    def _mdlp_edges(self, x: np.ndarray, y: np.ndarray, max_bins: int) -> np.ndarray:
        """
        Minimal, dependency-free MDLP (Fayyad–Irani) for binary/multi-class classification.
        Recursively split where information gain exceeds MDL stopping criterion.
        Returns sorted edges spanning the full range.
        """
        x = self._finite(x)
        mask = np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            return np.array([0.0, 1.0])

        order = np.argsort(x)
        x, y = x[order], y[order]

        # compress identical x while merging labels
        # (keeps potential split points only where labels can change)
        def candidate_splits(xv, yv):
            cs = []
            for i in range(1, xv.size):
                if xv[i] != xv[i - 1] and yv[i] != yv[i - 1]:
                    cs.append((xv[i - 1] + xv[i]) / 2.0)
            return cs

        from math import log2

        def entropy(labels):
            _, cnt = np.unique(labels, return_counts=True)
            p = cnt / cnt.sum()
            return -(p * np.log2(p + 1e-12)).sum()

        def mdl_stop(parent_y, left_y, right_y, N, k):
            # Fayyad & Irani MDL stopping rule
            H_parent = entropy(parent_y)
            H_left, H_right = entropy(left_y), entropy(right_y)
            Nl, Nr = len(left_y), len(right_y)
            gain = H_parent - (Nl / N) * H_left - (Nr / N) * H_right

            k_left = len(np.unique(left_y))
            k_right = len(np.unique(right_y))
            delta = log2(3**k - 2) - (
                k * H_parent - k_left * H_left - k_right * H_right
            )
            threshold = (log2(N - 1) + delta) / N
            return gain > threshold, gain

        edges = [x.min() - 1e-12, x.max() + 1e-12]

        def split_rec(xv, yv, depth=0):
            if len(np.unique(yv)) <= 1 or xv.size < 2:
                return []

            N = xv.size
            k = len(np.unique(yv))
            splits = candidate_splits(xv, yv)
            if not splits:
                return []

            best_s, best_gain = None, -np.inf
            for s in splits:
                left_mask = xv <= s
                right_mask = ~left_mask
                yl, yr = yv[left_mask], yv[right_mask]
                if yl.size == 0 or yr.size == 0:
                    continue
                ok, gain = mdl_stop(yv, yl, yr, N, k)
                if ok and gain > best_gain:
                    best_gain, best_s = gain, s

            if best_s is None:
                return []
            # recurse on both sides
            left = xv <= best_s
            right = ~left
            return (
                split_rec(xv[left], yv[left], depth + 1)
                + [best_s]
                + split_rec(xv[right], yv[right], depth + 1)
            )

        cut_points = split_rec(x, y)
        cut_points = sorted(set(cut_points))
        if len(cut_points) + 1 > max_bins:
            # trim to max_bins by keeping most separated cuts
            # simple heuristic: uniform downsample of cuts
            step = max(1, int(np.ceil(len(cut_points) / (max_bins - 1))))
            cut_points = cut_points[::step][: max(0, max_bins - 1)]

        full_edges = np.array(
            [x.min() - 1e-12] + cut_points + [x.max() + 1e-12], dtype=float
        )
        return full_edges

    # ---------- fit/transform ----------

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BinningTransformer":
        if not getattr(self.config, "create_binning", True):
            self.is_fitted = True
            return self

        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()

        for col in self.numerical_cols_:
            data = X[[col]].dropna()
            if data.empty:
                continue

            col_values = data[col].values.astype(float)
            n_unique = int(pd.Series(col_values).nunique(dropna=True))
            n = len(col_values)
            if n_unique <= 1:
                continue

            col_min, col_max = np.min(col_values), np.max(col_values)
            if (col_max - col_min) < 1e-12:
                continue

            # dynamic cap for candidate bins based on sample size
            n_cap = max(1, n // max(self.min_samples_per_bin, 1))
            dynamic_max_bins = int(np.clip(min(self.max_bins, n_cap), 2, self.max_bins))

            self.binning_transformers_.setdefault(col, {})

            for strategy in self.strategies:
                try:
                    if strategy == "quantile":
                        k = int(
                            np.clip(
                                getattr(self.config, "n_bins", 10), 2, dynamic_max_bins
                            )
                        )
                        disc = KBinsDiscretizer(
                            n_bins=k,
                            encode="ordinal",
                            strategy="quantile",
                            subsample=min(20000, n),
                        )
                        disc.fit(col_values.reshape(-1, 1))
                        self.binning_transformers_[col][strategy] = {
                            "transformer": disc,
                            "n_bins": k,
                        }

                    elif strategy == "uniform":
                        k = int(
                            np.clip(
                                getattr(self.config, "n_bins", 10), 2, dynamic_max_bins
                            )
                        )
                        disc = KBinsDiscretizer(
                            n_bins=k,
                            encode="ordinal",
                            strategy="uniform",
                            subsample=min(20000, n),
                        )
                        disc.fit(col_values.reshape(-1, 1))
                        self.binning_transformers_[col][strategy] = {
                            "transformer": disc,
                            "n_bins": k,
                        }

                    elif strategy == "fd":
                        k = max(2, self._fd_bins(col_values))
                        k = int(np.clip(k, 2, dynamic_max_bins))
                        edges = self._equal_width_edges(col_values, k)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": k,
                        }

                    elif strategy == "doane":
                        k = max(2, self._doane_bins(col_values))
                        k = int(np.clip(k, 2, dynamic_max_bins))
                        edges = self._equal_width_edges(col_values, k)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": k,
                        }

                    elif strategy == "shimazaki":
                        k = max(
                            2, self._shimazaki_bins(col_values, *self.shimazaki_k_range)
                        )
                        k = int(np.clip(k, 2, dynamic_max_bins))
                        edges = self._equal_width_edges(col_values, k)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": k,
                        }

                    elif strategy == "kmeans":
                        k = int(
                            np.clip(
                                getattr(self.config, "n_bins", self.kmeans_bins),
                                2,
                                dynamic_max_bins,
                            )
                        )
                        edges = self._kmeans_edges(col_values, k)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }

                    elif strategy == "mdlp":
                        if y is None:
                            continue
                        y_arr = pd.Series(y).reindex(data.index).values
                        edges = self._mdlp_edges(
                            col_values, y_arr, max_bins=dynamic_max_bins
                        )
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }

                    else:
                        warnings.warn(
                            f"Unknown binning strategy: {strategy}. Skipping."
                        )
                        continue

                except Exception as e:
                    warnings.warn(f"Binning fit failed for {col} ({strategy}): {e}")
                    continue

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if (
            not getattr(self.config, "create_binning", True)
            or not self.binning_transformers_
        ):
            return pd.DataFrame(index=X.index)

        features = {}
        for col, strat_info in self.binning_transformers_.items():
            if col not in X.columns:
                continue
            col_data = X[col].astype(float)
            # simple imputation: median (keeps bin indices stable)
            col_data = col_data.fillna(col_data.median())
            X_col = col_data.values.reshape(-1, 1)

            for strategy, info in strat_info.items():
                try:
                    if "edges" in info:
                        edges = info["edges"]
                        binned = np.digitize(col_data.values, edges) - 1
                        binned = np.clip(binned, 0, len(edges) - 2)
                    else:
                        transformer = info["transformer"]
                        binned = transformer.transform(X_col).flatten()

                    name = (
                        f"{col}_bin_{strategy}"
                        if len(self.strategies) > 1
                        else f"{col}_bin"
                    )
                    features[name] = binned.astype(int)
                except Exception as e:
                    warnings.warn(
                        f"Binning transform failed for {col} ({strategy}): {e}"
                    )
                    continue

        return pd.DataFrame(features, index=X.index)


class RandomFourierFeaturesTransformer(BaseFeatureTransformer):
    """
    Random Fourier Features (RFF) transformer for approximating RBF kernels.

    Efficiently approximates kernel methods by mapping input features to a
    higher-dimensional space using random projections and trigonometric functions.

    Improvements over original:
    - Fixed scaling factor for proper Monte Carlo approximation
    - Better gamma estimation using median pairwise distances
    - Target-aware feature selection with multiple criteria
    - Local random state management
    - Robust handling of missing features
    - Better error handling and validation
    """

    def __init__(
        self,
        config: Any,
        n_components: int = 50,
        gamma: Union[float, str] = "auto",
        kernel: str = "rbf",
        max_features: int = 50,
        feature_selection_method: str = "variance",
        handle_missing_features: str = "error",  # 'error', 'ignore', 'impute'
    ):
        super().__init__(config)
        self.n_components = n_components
        self.gamma = gamma
        self.kernel = kernel
        self.max_features = max_features
        self.feature_selection_method = feature_selection_method
        self.handle_missing_features = handle_missing_features

        # Validation
        if kernel not in ["rbf", "laplacian"]:
            raise ValueError(f"Unsupported kernel: {kernel}. Use 'rbf' or 'laplacian'.")

        if feature_selection_method not in ["variance", "f_score", "mutual_info"]:
            raise ValueError(
                f"Unsupported feature selection method: {feature_selection_method}"
            )

        if handle_missing_features not in ["error", "ignore", "impute"]:
            raise ValueError(
                f"Unsupported missing feature handling: {handle_missing_features}"
            )

        # Fitted attributes
        self.gamma_ = None
        self.random_weights_ = None
        self.random_offset_ = None
        self.scaler_ = None
        self.selected_features_ = None
        self.feature_medians_ = None
        self._random_state = None

    def _get_random_state(self) -> np.random.RandomState:
        """Get local random state to avoid polluting global numpy random state."""
        if self._random_state is None:
            seed = getattr(self.config, "random_state", None)
            self._random_state = np.random.RandomState(seed)
        return self._random_state

    def _estimate_gamma(self, X: np.ndarray) -> float:
        """
        Estimate gamma parameter using median pairwise distances.
        More robust than simple 1/n_features approach.
        """
        if isinstance(self.gamma, str) and self.gamma == "auto":
            # Sample subset for efficiency if data is large
            n_samples = min(1000, X.shape[0])
            if X.shape[0] > n_samples:
                rng = self._get_random_state()
                indices = rng.choice(X.shape[0], n_samples, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X

            # Calculate pairwise distances and use median
            try:
                pairwise_dists = pdist(X_sample, metric="euclidean")
                if len(pairwise_dists) == 0:
                    return 1.0  # Fallback
                median_dist = np.median(pairwise_dists)
                if median_dist == 0:
                    return 1.0  # Fallback for identical points
                return 1.0 / (2 * median_dist**2)  # Standard RBF gamma
            except Exception:
                # Fallback to simple heuristic
                return 1.0 / X.shape[1]

        return float(self.gamma)

    def _select_features_variance(self, X: pd.DataFrame) -> List[str]:
        """Select features with highest variance."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Select features with highest variance
        variances = X[numerical_cols].var()
        return variances.nlargest(self.max_features).index.tolist()

    def _select_features_target_aware(
        self, X: pd.DataFrame, y: pd.Series, method: str
    ) -> List[str]:
        """Select features based on relationship with target."""
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()

        if len(numerical_cols) <= self.max_features:
            return numerical_cols

        # Prepare data
        X_clean = X[numerical_cols].fillna(X[numerical_cols].median())
        y_clean = y.fillna(y.median()) if y.dtype in ["float64", "int64"] else y

        # Align indices
        common_idx = X_clean.index.intersection(y_clean.index)
        X_clean = X_clean.loc[common_idx]
        y_clean = y_clean.loc[common_idx]

        if len(X_clean) == 0:
            return numerical_cols[: self.max_features]

        try:
            if method == "f_score":
                selector = SelectKBest(score_func=f_regression, k=self.max_features)
            else:  # mutual_info
                selector = SelectKBest(
                    score_func=mutual_info_regression, k=self.max_features
                )

            selector.fit(X_clean, y_clean)
            selected_mask = selector.get_support()
            return [
                col for col, selected in zip(numerical_cols, selected_mask) if selected
            ]

        except Exception as e:
            warnings.warn(
                f"Target-aware feature selection failed: {e}. Falling back to variance-based selection."
            )
            return self._select_features_variance(X)

    def _select_features(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> List[str]:
        """Select features using specified method."""
        if self.feature_selection_method == "variance" or y is None:
            return self._select_features_variance(X)
        else:
            return self._select_features_target_aware(
                X, y, self.feature_selection_method
            )

    def _generate_random_weights(
        self, n_features: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random weights for Fourier features."""
        rng = self._get_random_state()

        if self.kernel == "rbf":
            # For RBF kernel: w ~ N(0, 2*gamma*I)
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )
        elif self.kernel == "laplacian":
            # For Laplacian kernel: w ~ Laplace(0, sqrt(gamma))
            weights = rng.laplace(
                0, 1 / np.sqrt(self.gamma_), (n_features, self.n_components)
            )
        else:
            # Fallback to RBF
            weights = rng.normal(
                0, np.sqrt(2 * self.gamma_), (n_features, self.n_components)
            )

        # Random phase offset
        offset = rng.uniform(0, 2 * np.pi, self.n_components)
        return weights, offset

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "RandomFourierFeaturesTransformer":
        """Fit the Random Fourier Features transformer."""
        if not getattr(self.config, "create_rff", False):
            self.selected_features_ = []
            self.is_fitted = True
            return self

        # Select features
        self.selected_features_ = self._select_features(X, y)

        if len(self.selected_features_) < 2:
            warnings.warn(
                "Less than 2 numerical features available. RFF transformation will be minimal."
            )
            self.is_fitted = True
            return self

        # Prepare data
        X_selected = X[self.selected_features_].copy()

        # Store medians for consistent imputation
        self.feature_medians_ = X_selected.median()
        X_selected = X_selected.fillna(self.feature_medians_)

        # Normalize features
        self.scaler_ = StandardScaler()
        X_normalized = self.scaler_.fit_transform(X_selected)

        # Estimate gamma and generate weights
        self.gamma_ = self._estimate_gamma(X_normalized)
        self.random_weights_, self.random_offset_ = self._generate_random_weights(
            len(self.selected_features_)
        )

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using Random Fourier Features."""
        if not getattr(self.config, "create_rff", False) or not self.selected_features_:
            return pd.DataFrame(index=X.index)

        if not self.is_fitted:
            raise ValueError("Transformer must be fitted before transform.")

        # Handle missing features
        available_features = [f for f in self.selected_features_ if f in X.columns]
        missing_features = [f for f in self.selected_features_ if f not in X.columns]

        if missing_features:
            if self.handle_missing_features == "error":
                raise ValueError(
                    f"Missing features in transform data: {missing_features}"
                )
            elif self.handle_missing_features == "ignore":
                warnings.warn(
                    f"Missing features will be imputed with median: {missing_features}"
                )

        # Prepare feature matrix
        X_selected = pd.DataFrame(index=X.index)

        # Add available features
        for feature in self.selected_features_:
            if feature in X.columns:
                X_selected[feature] = X[feature]
            else:
                # Impute missing features with stored median
                median_val = self.feature_medians_.get(feature, 0.0)
                X_selected[feature] = median_val

        # Fill NaN values with stored medians
        for feature in self.selected_features_:
            if feature in self.feature_medians_:
                X_selected[feature] = X_selected[feature].fillna(
                    self.feature_medians_[feature]
                )

        # Normalize and transform
        try:
            X_normalized = self.scaler_.transform(X_selected[self.selected_features_])
        except Exception as e:
            raise ValueError(f"Error during scaling: {e}")

        # Generate projections
        projection = np.dot(X_normalized, self.random_weights_) + self.random_offset_

        # Generate RFF features with correct scaling
        features = {}

        if self.kernel in ["rbf", "laplacian"]:
            # Correct scaling factor for Monte Carlo approximation
            scaling_factor = np.sqrt(1.0 / self.n_components)

            cos_features = np.cos(projection) * scaling_factor
            sin_features = np.sin(projection) * scaling_factor

            for i in range(self.n_components):
                features[f"rff_cos_{i}"] = cos_features[:, i]
                features[f"rff_sin_{i}"] = sin_features[:, i]
        else:
            # For other kernels, just use the projection
            for i in range(self.n_components):
                features[f"rff_{i}"] = projection[:, i]

        return pd.DataFrame(features, index=X.index)

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance based on random weight magnitudes."""
        if not self.is_fitted or self.random_weights_ is None:
            return pd.Series()

        # Average absolute weight per input feature
        importance = np.mean(np.abs(self.random_weights_), axis=1)
        return pd.Series(importance, index=self.selected_features_)

    def get_feature_names_out(self) -> List[str]:
        """Get output feature names."""
        if not self.is_fitted:
            return []

        feature_names = []
        if self.kernel in ["rbf", "laplacian"]:
            for i in range(self.n_components):
                feature_names.extend([f"rff_cos_{i}", f"rff_sin_{i}"])
        else:
            for i in range(self.n_components):
                feature_names.append(f"rff_{i}")

        return feature_names


class FourierTransformer(BaseFeatureTransformer):
    """Creates Fourier features for periodic patterns."""

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FourierTransformer":
        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fourier_configs_ = {}

        if not self.config.create_fourier:
            self.is_fitted = True
            return self

        for col in self.numerical_cols_:
            if col not in X.columns:
                continue

            data = X[col].fillna(X[col].median())

            # Skip if insufficient variation
            if data.var() < 1e-6:
                continue

            try:
                # Normalize and compute FFT
                data_norm = (data - data.mean()) / (data.std() + 1e-8)
                fft_vals = fft.fft(data_norm.values)
                freqs = fft.fftfreq(len(data_norm))

                # Extract dominant frequencies
                magnitude = np.abs(fft_vals)
                top_freq_idx = np.argsort(magnitude)[
                    -(self.config.n_fourier_terms + 1) : -1
                ]

                valid_frequencies = [
                    freqs[idx] for idx in top_freq_idx if freqs[idx] != 0
                ]

                if valid_frequencies:
                    self.fourier_configs_[col] = {
                        "frequencies": valid_frequencies,
                        "mean": data.mean(),
                        "std": data.std(),
                    }
            except Exception as e:
                warnings.warn(f"Fourier analysis failed for {col}: {e}")

        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.config.create_fourier or not self.fourier_configs_:
            return pd.DataFrame(index=X.index)

        features = {}
        for col, config in self.fourier_configs_.items():
            if col not in X.columns:
                continue

            data = X[col].fillna(config["mean"])

            for i, freq in enumerate(config["frequencies"]):
                features[f"{col}_fourier_cos_{i}"] = np.cos(
                    2 * np.pi * freq * np.arange(len(data))
                )
                features[f"{col}_fourier_sin_{i}"] = np.sin(
                    2 * np.pi * freq * np.arange(len(data))
                )

        return pd.DataFrame(features, index=X.index)


class ClusteringTransformer(BaseFeatureTransformer):
    """Modern clustering/embedding-based feature generator."""

    def __init__(self, config, strategies=("kmeans", "gmm", "hdbscan", "umap")):
        super().__init__(config)
        self.strategies = strategies
        self.scaler_ = None
        self.models_ = {}
        self.cluster_features_ = []

    def fit(self, X, y=None):
        if not self.config.create_clustering:
            self.is_fitted = True
            return self

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            self.is_fitted = True
            return self

        Xnum = X[num_cols].fillna(X[num_cols].median())
        self.cluster_features_ = (
            Xnum.var().nlargest(min(20, len(num_cols))).index.tolist()
        )

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(Xnum[self.cluster_features_])

        n_clusters = max(2, min(self.config.n_clusters, len(X_scaled) // 10))

        # --- Fit models ---
        if "kmeans" in self.strategies:
            km = KMeans(
                n_clusters=n_clusters, n_init=10, random_state=self.config.random_state
            )
            km.fit(X_scaled)
            self.models_["kmeans"] = km

        if "gmm" in self.strategies:
            gmm = GaussianMixture(
                n_components=n_clusters, random_state=self.config.random_state
            )
            gmm.fit(X_scaled)
            self.models_["gmm"] = gmm

        if "hdbscan" in self.strategies:
            hdb = hdbscan.HDBSCAN(min_cluster_size=15)
            hdb.fit(X_scaled)
            self.models_["hdbscan"] = hdb

        if "umap" in self.strategies:
            import umap

            reducer = umap.UMAP(n_components=3, random_state=self.config.random_state)
            reducer.fit(X_scaled)
            self.models_["umap"] = reducer

        self.is_fitted = True
        return self

    def transform(self, X):
        if not self.models_:
            return pd.DataFrame(index=X.index)

        available = [c for c in self.cluster_features_ if c in X.columns]
        if not available:
            return pd.DataFrame(index=X.index)

        Xnum = X[available].fillna(X[available].median())
        X_scaled = self.scaler_.transform(Xnum)

        feats = {}

        for strat, model in self.models_.items():
            if strat == "kmeans":
                feats["kmeans_id"] = model.predict(X_scaled)
                dist = model.transform(X_scaled)
                for i in range(dist.shape[1]):
                    feats[f"kmeans_dist_{i}"] = dist[:, i]

            elif strat == "gmm":
                feats["gmm_id"] = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)
                for i in range(probs.shape[1]):
                    feats[f"gmm_prob_{i}"] = probs[:, i]

            elif strat == "hdbscan":
                feats["hdbscan_id"] = model.labels_
                feats["hdbscan_outlier"] = model.outlier_scores_

            elif strat == "umap":
                embedding = model.transform(X_scaled)
                for i in range(embedding.shape[1]):
                    feats[f"umap_{i}"] = embedding[:, i]

        return pd.DataFrame(feats, index=X.index, dtype=np.float32)
