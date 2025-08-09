import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from .foreminer_aux import *

# Optional deps
try:
    import phik

    HAS_PHIK = True
except Exception:
    HAS_PHIK = False

try:
    from dcor import distance_correlation as dcor_lib

    HAS_DCOR = True
except Exception:
    HAS_DCOR = False


# -------------------------
# Small internal utilities
# -------------------------
def _pairwise_align(x: pd.Series, y: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Align two columns on common non-NaN index quickly."""
    ix = x.index.intersection(y.index)
    x = x.loc[ix].to_numpy()
    y = y.loc[ix].to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]


def _entropy_discrete_bins(v: np.ndarray, bins: int = 32) -> float:
    """Simple plug-in entropy estimate with fixed bins; stable for MI normalization."""
    if v.size == 0:
        return 0.0
    hist, _ = np.histogram(v, bins=bins)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return 0.0
    p /= s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _safe_clip_quantiles(x: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    if x.empty:
        return x
    ql, qh = x.quantile([q_low, q_high])
    return x.clip(lower=ql, upper=qh)


class CorrelationAnalyzer(AnalysisStrategy):
    """Fast SOTA correlation analysis with selective advanced methods (cleaned)."""

    @property
    def name(self) -> str:
        return "correlations"

    def __init__(
        self,
        fast_threshold: int = 1000,
        medium_threshold: int = 5000,
        large_sample_for_expensive: int = 2000,
        candidate_screen_abs_spearman: float = 0.08,
        mi_bins: int = 32,
        random_state: int = 42,
        return_longform: bool = False,
    ):
        # thresholds / knobs
        self.fast_threshold = fast_threshold
        self.medium_threshold = medium_threshold
        self.large_sample_size = large_sample_for_expensive
        self.candidate_screen_abs_spearman = candidate_screen_abs_spearman
        self.mi_bins = mi_bins
        self.random_state = random_state
        self.return_longform = return_longform

        # Core (always computed)
        self.core_strategies = {
            "pearson": self._pearson_correlation,
            "spearman": self._spearman_correlation,
            "kendall": self._kendall_correlation,
        }

        # Advanced (adaptive)
        self.advanced_strategies = {
            "mutual_info": self._mutual_info_correlation,
            "distance": self._distance_correlation,
            "chatterjee": self._chatterjee_correlation,
            "robust_pearson": self._robust_pearson_correlation,
        }
        if HAS_PHIK:
            self.advanced_strategies["phik"] = self._phik_correlation

    # -----------------------------
    # Smart subsampling via PCA(1)
    # -----------------------------
    def _smart_subsample(
        self, df: pd.DataFrame, target_size: int = 2000
    ) -> pd.DataFrame:
        """Correlation-preserving subsample:
        project to PC1, stratify by quartiles, then sample uniformly within strata.
        """
        n = len(df)
        if n <= target_size:
            return df

        rng = np.random.default_rng(self.random_state)

        # Robustly get PC1 scores; fallback to first numeric column if PCA fails
        try:
            # Work on z-scored copy to stabilize PCA direction
            X = df.to_numpy(dtype=float)
            X = X - np.nanmean(X, axis=0, keepdims=True)
            X = np.nan_to_num(X, nan=0.0)
            pc1 = PCA(n_components=1, random_state=self.random_state).fit_transform(X)[
                :, 0
            ]
            s = pd.Series(pc1, index=df.index)
        except Exception:
            s = df.iloc[:, 0]

        # Stratify by quartiles of PC1
        q = s.quantile([0.25, 0.5, 0.75]).to_numpy()
        strata = [
            s <= q[0],
            (s > q[0]) & (s <= q[1]),
            (s > q[1]) & (s <= q[2]),
            s > q[2],
        ]
        per = target_size // 4
        take_idx: List[int] = []
        for mask in strata:
            idx = df.index[mask].to_numpy()
            if idx.size == 0:
                continue
            k = min(per, idx.size)
            take_idx.extend(rng.choice(idx, size=k, replace=False).tolist())

        # Top-up if undersampled due to empty strata
        take_idx = np.array(take_idx, dtype=df.index.dtype)
        if take_idx.size < target_size:
            remaining = df.index.difference(take_idx)
            k = min(target_size - take_idx.size, remaining.size)
            if k > 0:
                take_idx = np.concatenate(
                    [take_idx, rng.choice(remaining.to_numpy(), size=k, replace=False)]
                )

        return df.loc[take_idx].copy()

    # -----------------------------
    # Core correlations
    # -----------------------------
    def _pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.corr(method="pearson", min_periods=30)

    def _spearman_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.corr(method="spearman", min_periods=30)

    def _kendall_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        # Keep cost bounded
        if len(df) > 600:
            return pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )
        return df.corr(method="kendall", min_periods=30)

    # -----------------------------
    # Advanced correlations
    # -----------------------------
    def _mutual_info_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pairwise MI normalized by min(H(X), H(Y)) in [0,1]."""
        cols = df.columns
        M = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        S = self._smart_subsample(df, target_size=min(self.large_sample_size, 2000))
        # lightweight screening once
        scr = S.corr(method="spearman").abs()

        for i, c1 in enumerate(cols):
            x_all = S[c1]
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if scr.at[c1, c2] < self.candidate_screen_abs_spearman:
                    continue
                y_all = S[c2]

                x, y = _pairwise_align(x_all, y_all)
                if x.size < 50:
                    continue

                X = x.reshape(-1, 1)
                try:
                    mi = float(
                        mutual_info_regression(
                            X,
                            y,
                            random_state=self.random_state,
                            discrete_features=False,
                        )[0]
                    )
                except Exception:
                    continue

                # Normalize by min entropies (plug-in estimate on fixed bins)
                hx = _entropy_discrete_bins(x, bins=self.mi_bins)
                hy = _entropy_discrete_bins(y, bins=self.mi_bins)
                hmin = max(min(hx, hy), 1e-12)
                mi_n = max(0.0, min(mi / hmin, 1.0))
                M.iat[i, j] = mi_n
                M.iat[j, i] = mi_n
        return M

    def _distance_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Distance correlation with correct normalization. Subsample + screen."""
        cols = df.columns
        D = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        if not HAS_DCOR and len(df) > 1200:
            # Without the library, keep it small to avoid O(n^2) blowups
            df = self._smart_subsample(df, target_size=800)
        else:
            df = self._smart_subsample(
                df, target_size=min(self.large_sample_size, 1200)
            )

        scr = df.corr(method="pearson").abs()

        for i, c1 in enumerate(cols):
            x_all = df[c1]
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if scr.at[c1, c2] < 0.05:
                    continue
                y_all = df[c2]

                x, y = _pairwise_align(x_all, y_all)
                n = x.size
                if n < 50:
                    continue

                try:
                    if HAS_DCOR:
                        val = float(dcor_lib(x, y))
                    else:
                        # O(n^2) but n is controlled. Use L1 distances (fast) + double-centering
                        # Distance matrices
                        dx = np.abs(x[:, None] - x[None, :])
                        dy = np.abs(y[:, None] - y[None, :])

                        # U-centered versions (bias-reduced variant)
                        # See Székely & Rizzo; but here a pragmatic centered version
                        dx_mean_row = dx.mean(axis=1, keepdims=True)
                        dx_mean_col = dx.mean(axis=0, keepdims=True)
                        dx_mean = dx.mean()
                        Ax = dx - dx_mean_row - dx_mean_col + dx_mean

                        dy_mean_row = dy.mean(axis=1, keepdims=True)
                        dy_mean_col = dy.mean(axis=0, keepdims=True)
                        dy_mean = dy.mean()
                        Ay = dy - dy_mean_row - dy_mean_col + dy_mean

                        dcov_xy = (Ax * Ay).mean()
                        dcov_xx = (Ax * Ax).mean()
                        dcov_yy = (Ay * Ay).mean()
                        denom = np.sqrt(max(dcov_xx * dcov_yy, 1e-12))
                        val = float(max(0.0, min(dcov_xy / denom, 1.0)))
                    if not np.isnan(val):
                        D.iat[i, j] = val
                        D.iat[j, i] = val
                except Exception:
                    continue
        return D

    def _chatterjee_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Symmetrized Chatterjee ξ with pair screening."""
        cols = df.columns
        C = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)

        if len(df) > 2500:
            df = self._smart_subsample(df, target_size=1200)

        scr = df.corr(method="spearman").abs()

        def fast_chatt(a: np.ndarray, b: np.ndarray) -> float:
            n = a.size
            if n < 50:
                return np.nan
            # sort by a; rank b; accumulate |diff|
            order = np.argsort(a, kind="mergesort")  # stable
            br = rankdata(b)[order]
            diffs = np.abs(np.diff(br))
            # ξ_n = 1 - (3 * sum|Δ|) / (n^2 - 1)
            denom = n * n - 1.0
            if denom <= 0:
                return np.nan
            val = 1.0 - (3.0 * diffs.sum()) / denom
            return float(np.clip(val, 0.0, 1.0))

        for i, c1 in enumerate(cols):
            x_all = df[c1]
            for j in range(i + 1, len(cols)):
                c2 = cols[j]
                if scr.at[c1, c2] < self.candidate_screen_abs_spearman:
                    continue

                y_all = df[c2]
                x, y = _pairwise_align(x_all, y_all)
                if x.size < 50:
                    continue
                try:
                    v = np.nanmean([fast_chatt(x, y), fast_chatt(y, x)])
                    if np.isfinite(v):
                        C.iat[i, j] = v
                        C.iat[j, i] = v
                except Exception:
                    continue
        return C

    def _robust_pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Winsorize per feature (5%,95%) then Pearson."""
        W = df.copy()
        for c in W.columns:
            W[c] = _safe_clip_quantiles(W[c], 0.05, 0.95)
        return W.corr(method="pearson", min_periods=30)

    def _phik_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df_s = self._smart_subsample(
                df, target_size=min(self.large_sample_size, 2000)
            )
            return df_s.phik_matrix()
        except Exception:
            return pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )

    # -----------------------------
    # Driver
    # -----------------------------
    def analyze(self, data: pd.DataFrame, config: AnalysisConfig) -> Dict[str, Any]:
        """Adaptive correlation analysis. Returns a dict of matrices (+ optional longform)."""
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {}

        # Drop constant columns (pairwise NaN handling will happen later)
        std = numeric.std(numeric_only=True)
        numeric = numeric.loc[:, std > 1e-10]
        n_samples, n_features = numeric.shape
        if n_features < 2:
            return {}

        results: Dict[str, Any] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Core
            for name, fn in self.core_strategies.items():
                try:
                    results[name] = fn(numeric)
                except Exception as e:
                    print(f"[correlations] Skipped {name}: {e}")

            # Decide advanced set
            if n_samples < self.fast_threshold:
                adv = list(self.advanced_strategies.keys())
            elif n_samples < self.medium_threshold:
                adv = ["mutual_info", "distance", "robust_pearson"]
                if HAS_PHIK:
                    adv.append("phik")
            else:
                adv = ["robust_pearson"]
                if n_features < 20:
                    adv.append("mutual_info")

            for name in adv:
                try:
                    results[name] = self.advanced_strategies[name](numeric)
                except Exception as e:
                    print(f"[correlations] Skipped {name}: {e}")

            # Simple ensemble (abs-avg of Pearson, Spearman, best advanced present)
            ensemble_parts: List[pd.DataFrame] = []
            for key in (
                "pearson",
                "spearman",
                "distance",
                "mutual_info",
                "robust_pearson",
            ):
                if key in results:
                    ensemble_parts.append(results[key].abs())
            if len(ensemble_parts) >= 2:
                results["ensemble"] = sum(ensemble_parts) / float(len(ensemble_parts))

            # Optional long-form (sorted by ensemble if present, else Spearman)
            if self.return_longform:
                base = results.get("ensemble") or results.get("spearman")
                if isinstance(base, pd.DataFrame):
                    tri = []
                    cols = base.columns.tolist()
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            a, b = cols[i], cols[j]
                            row = {"feature_a": a, "feature_b": b}
                            for k, mat in results.items():
                                if k == "_metadata":
                                    continue
                                try:
                                    row[k] = float(mat.at[a, b])
                                except Exception:
                                    row[k] = np.nan
                            tri.append(row)
                    long_df = pd.DataFrame(tri)
                    sort_key = "ensemble" if "ensemble" in long_df else "spearman"
                    long_df = long_df.sort_values(
                        by=sort_key, ascending=False, na_position="last"
                    )
                    results["longform"] = long_df.reset_index(drop=True)

            # Metadata
            results["_metadata"] = {
                "n_features": n_features,
                "n_samples": n_samples,
                "methods_computed": [k for k in results.keys() if k != "_metadata"],
                "performance_tier": (
                    "fast"
                    if n_samples < self.fast_threshold
                    else "medium" if n_samples < self.medium_threshold else "large"
                ),
                "screen_abs_spearman": self.candidate_screen_abs_spearman,
                "random_state": self.random_state,
            }
        return results
