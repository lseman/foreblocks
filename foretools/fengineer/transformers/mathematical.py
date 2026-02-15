import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

from .base import BaseFeatureTransformer


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
        self.use_target_aware = bool(getattr(config, "math_target_aware", True))
        self.target_weight = float(getattr(config, "math_target_weight", 0.25))
        self.max_transforms_per_feature = int(
            getattr(config, "math_max_transforms_per_feature", 3)
        )

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

    @staticmethod
    def _prepare_target(y: Optional[pd.Series]) -> Optional[np.ndarray]:
        if y is None:
            return None
        ys = pd.Series(y)
        if pd.api.types.is_numeric_dtype(ys):
            arr = pd.to_numeric(ys, errors="coerce").to_numpy(dtype=np.float64)
            arr[~np.isfinite(arr)] = np.nan
            return arr
        # classification-style fallback: category codes
        codes = ys.astype("category").cat.codes.astype(float).to_numpy()
        codes[codes < 0] = np.nan
        return codes

    def _target_score(self, arr: np.ndarray, y_arr: Optional[np.ndarray]) -> float:
        if y_arr is None:
            return 0.0
        m = np.isfinite(arr) & np.isfinite(y_arr)
        if m.sum() < 25:
            return 0.0
        a = arr[m]
        b = y_arr[m]
        sa = np.nanstd(a)
        sb = np.nanstd(b)
        if sa < self.eps or sb < self.eps:
            return 0.0
        c = float(np.corrcoef(a, b)[0, 1])
        if not np.isfinite(c):
            return 0.0
        return abs(c)

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
        y_arr = self._prepare_target(y)

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
            # evaluate candidates on imputed full-length array so target scoring aligns
            series_eval = series.fillna(self.col_medians_[col])

            # ===== basic transforms: keep those that improve score by threshold =====
            cands = self._candidate_basic(col, series_eval)
            base_score = self._normality_score(cands["identity"])
            base_target = self._target_score(cands["identity"], y_arr)

            ranked: List[tuple] = []
            for name, arr in cands.items():
                if name == "identity":
                    continue
                if np.nanvar(arr) <= self.min_variance:
                    continue
                score = self._normality_score(arr)
                gain_normality = base_score - score
                gain_target = self._target_score(arr, y_arr) - base_target
                combined_gain = gain_normality + (
                    self.target_weight * gain_target
                    if (self.use_target_aware and y_arr is not None)
                    else 0.0
                )
                if combined_gain > self.improvement_threshold:
                    ranked.append((combined_gain, name))
            kept = [name for _, name in sorted(ranked, reverse=True)][
                : self.max_transforms_per_feature
            ]
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
                        # evaluate PT gain on full eval array for target-aware consistency
                        a_eval = self._winsorize(self._safe_arr(series_eval))
                        transformed_eval = pt.transform(a_eval.reshape(-1, 1)).ravel()
                        gain_normality = base_score - self._normality_score(
                            transformed_eval
                        )
                        gain_target = (
                            self._target_score(transformed_eval, y_arr) - base_target
                        )
                        combined_gain = gain_normality + (
                            self.target_weight * gain_target
                            if (self.use_target_aware and y_arr is not None)
                            else 0.0
                        )
                        if combined_gain > self.improvement_threshold:
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
