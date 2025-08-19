import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mutual_info_score

from .adaptive_mi import AdaptiveMI
from .distance_correlation import DistanceCorrelation
from .foreminer_aux import *
from .hsic import HSIC

# Optional deps
try:
    import phik

    HAS_PHIK = True
except Exception:
    HAS_PHIK = False

# -------------------------
# Utilities
# -------------------------
def _pairwise_align_np(x: np.ndarray, y: np.ndarray):
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def _safe_clip_quantiles(x: pd.Series, q_low=0.05, q_high=0.95) -> pd.Series:
    if x.empty:
        return x
    ql, qh = x.quantile([q_low, q_high])
    return x.clip(lower=ql, upper=qh)


# -------------------------
# Correlation Analyzer
# -------------------------
class CorrelationAnalyzer(AnalysisStrategy):
    """Modern correlation analysis with linear + nonlinear dependence measures."""

    @property
    def name(self) -> str:
        return "correlations"

    def __init__(
        self,
        fast_threshold: int = 1000,
        medium_threshold: int = 5000,
        large_sample_for_expensive: int = 2000,
        candidate_screen_abs_spearman: float = 0.10,
        mi_bins: int = 32,
        random_state: int = 42,
        return_longform: bool = False,
    ):
        self.fast_threshold = fast_threshold
        self.medium_threshold = medium_threshold
        self.large_sample_size = large_sample_for_expensive
        self.candidate_screen_abs_spearman = candidate_screen_abs_spearman
        self.mi_bins = mi_bins
        self.random_state = random_state
        self.return_longform = return_longform

    # -----------------------------
    # Smart subsampling
    # -----------------------------
    def _smart_subsample(
        self, df: pd.DataFrame, target_size: int = 2000
    ) -> pd.DataFrame:
        n = len(df)
        if n <= target_size:
            return df
        rng = np.random.default_rng(self.random_state)
        try:
            X = df.to_numpy(dtype=float)
            X = X - np.nanmean(X, axis=0, keepdims=True)
            X = np.nan_to_num(X, nan=0.0)
            pc1 = PCA(n_components=1, random_state=self.random_state).fit_transform(X)[
                :, 0
            ]
            s = pd.Series(pc1, index=df.index)
        except Exception:
            s = df.iloc[:, 0]
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
    def _compute_core_corrs(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        results = {}
        results["pearson"] = df.corr(method="pearson", min_periods=30)
        results["spearman"] = df.corr(method="spearman", min_periods=30)
        if len(df) <= 600:
            results["kendall"] = df.corr(method="kendall", min_periods=30)
        else:
            results["kendall"] = pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )
        return results

    # -----------------------------
    # Advanced correlations
    # -----------------------------
    def _mutual_info_correlation(self, df: pd.DataFrame, spearman_scr: pd.DataFrame) -> pd.DataFrame:
        ami = AdaptiveMI(
            subsample=min(getattr(self, "large_sample_size", 2000), 2000),
            spearman_gate=getattr(self, "candidate_screen_abs_spearman", 0.05),
            min_overlap=50,
            ks=(3, 5, 10),
            n_bins=getattr(self, "mi_bins", 16),
            random_state=getattr(self, "random_state", 42),
        )
        return ami.matrix(df, spearman_scr)

    def _distance_correlation(
        self, df: pd.DataFrame, pearson_scr: pd.DataFrame
    ) -> pd.DataFrame:
        dc = DistanceCorrelation(
            block_size=1024,  # increase if you have RAM; 2048–4096 is fine
            unbiased=False,  # True for small n and statistical purity
            pearson_gate=0.05,  # keep your gate
            max_n=1200,  # align with your sampler
            random_state=getattr(self, "random_state", 42),
            use_coreset=True,
        )
        return dc.matrix(df, pearson_scr)

    def _hsic_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Improved HSIC with subsampling and normalization using HSIC class."""
        cols = df.columns
        p = len(cols)
        mat = np.eye(p)

        # configure HSIC scorer (set options here)
        scorer = HSIC(
            kernel_x="rbf",
            kernel_y="rbf",
            estimator="biased",  # or "unbiased"
            normalize=True,
            use_numba=True,
        )

        S = self._smart_subsample(df, target_size=800)
        for i, c1 in enumerate(cols):
            for j in range(i + 1, p):
                x, y = _pairwise_align_np(S[c1].to_numpy(), S[cols[j]].to_numpy())
                if x.size < 50:
                    continue
                try:
                    v = scorer.score(x, y)
                    mat[i, j] = mat[j, i] = v
                except Exception:
                    continue
        return pd.DataFrame(mat, index=cols, columns=cols)

    def _chatterjee_correlation(
        self, df: pd.DataFrame, spearman_scr: pd.DataFrame
    ) -> pd.DataFrame:
        cols = df.columns
        p = len(cols)
        mat = np.eye(p)
        if len(df) > 2500:
            df = self._smart_subsample(df, target_size=1200)

        def fast_chatt(a: np.ndarray, b: np.ndarray) -> float:
            n = a.size
            if n < 50:
                return np.nan
            order = np.argsort(a, kind="mergesort")
            br = rankdata(b)[order]
            diffs = np.abs(np.diff(br))
            denom = n * n - 1.0
            if denom <= 0:
                return np.nan
            val = 1.0 - (3.0 * diffs.sum()) / denom
            return float(np.clip(val, 0.0, 1.0))

        for i, c1 in enumerate(cols):
            x_all = df[c1].to_numpy()
            for j in range(i + 1, p):
                if spearman_scr.iat[i, j] < self.candidate_screen_abs_spearman:
                    continue
                y_all = df[cols[j]].to_numpy()
                x, y = _pairwise_align_np(x_all, y_all)
                if x.size < 50:
                    continue
                try:
                    v = np.nanmean([fast_chatt(x, y), fast_chatt(y, x)])
                    if np.isfinite(v):
                        mat[i, j] = mat[j, i] = v
                except Exception:
                    continue
        return pd.DataFrame(mat, index=cols, columns=cols)

    def _robust_pearson_correlation(self, df: pd.DataFrame) -> pd.DataFrame:
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
        numeric = data.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return {}
        std = numeric.std(numeric_only=True)
        numeric = numeric.loc[:, std > 1e-10]
        n_samples, n_features = numeric.shape
        if n_features < 2:
            return {}
        results: Dict[str, Any] = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            core = self._compute_core_corrs(numeric)
            results.update(core)
            pearson_scr = core["pearson"].abs()
            spearman_scr = core["spearman"].abs()

            # Select advanced methods
            if n_samples < self.fast_threshold:
                adv = [
                    "mutual_info",
                    "distance",
                    "chatterjee",
                    "robust_pearson",
                    "hsic",
                ]
                if HAS_PHIK:
                    adv.append("phik")
            elif n_samples < self.medium_threshold:
                adv = ["mutual_info", "distance", "robust_pearson", "hsic"]
                if HAS_PHIK:
                    adv.append("phik")
            else:
                adv = ["robust_pearson"]
                if n_features < 20:
                    adv.extend(["mutual_info", "hsic"])

            for name in adv:
                try:
                    if name == "mutual_info":
                        results[name] = self._mutual_info_correlation(
                            numeric, spearman_scr
                        )
                    elif name == "distance":
                        results[name] = self._distance_correlation(numeric, pearson_scr)
                    elif name == "chatterjee":
                        results[name] = self._chatterjee_correlation(
                            numeric, spearman_scr
                        )
                    elif name == "robust_pearson":
                        results[name] = self._robust_pearson_correlation(numeric)
                    elif name == "phik":
                        results[name] = self._phik_correlation(numeric)
                    elif name == "hsic":
                        results[name] = self._hsic_correlation(numeric)
                except Exception as e:
                    print(f"[correlations] Skipped {name}: {e}")

            # Ensemble
            ensemble_parts: List[pd.DataFrame] = []
            for key in (
                "pearson",
                "spearman",
                "distance",
                "mutual_info",
                "robust_pearson",
                "hsic",
            ):
                if key in results:
                    ensemble_parts.append(results[key].abs())
            if len(ensemble_parts) >= 2:
                results["ensemble"] = sum(ensemble_parts) / float(len(ensemble_parts))

            # Longform
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
                    results["longform"] = long_df.sort_values(
                        by=sort_key, ascending=False, na_position="last"
                    ).reset_index(drop=True)

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
