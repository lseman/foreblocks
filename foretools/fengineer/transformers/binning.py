import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import KBinsDiscretizer

from .base import BaseFeatureTransformer


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

        config_strategies = getattr(self.config, "binning_strategies", None)
        if strategies is not None:
            self.strategies = strategies
        elif config_strategies:
            self.strategies = config_strategies
        else:
            # Safe default when config explicitly sets None
            self.strategies = ["auto"]

        self.auto_supervised = bool(getattr(self.config, "binning_auto_supervised", True))
        self.min_bin_fraction = float(getattr(self.config, "binning_min_bin_fraction", 0.01))

        self.binning_transformers_: Dict[str, Dict[str, Any]] = {}
        self.numerical_cols_: List[str] = []
        self.fill_values_: Dict[str, float] = {}
        self.is_fitted = False

    # ---------- helpers: bin-count rules -> edges ----------

    @staticmethod
    def _finite(x: np.ndarray) -> np.ndarray:
        return x[np.isfinite(x)]

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        n = max(1, len(y))
        k = y.nunique(dropna=True)
        return (k <= 20) or (k / n < 0.05)

    def _resolve_strategies_for_col(
        self, x: np.ndarray, y_col: Optional[np.ndarray]
    ) -> List[str]:
        strategies = [str(s).lower() for s in self.strategies]
        if "auto" not in strategies:
            return strategies
        # Replace auto with a robust per-column choice.
        out = [s for s in strategies if s != "auto"]
        x_f = self._finite(x)
        if x_f.size < 20:
            out.append("uniform")
            return out
        skew = stats.skew(x_f, bias=False) if x_f.size >= 8 else 0.0
        if (
            self.auto_supervised
            and y_col is not None
            and np.isfinite(y_col).sum() >= 50
        ):
            # For classification-like targets, prefer supervised MDLP.
            y_s = pd.Series(y_col)
            if self._is_classification_target(y_s):
                if getattr(self.config, 'use_woe', False) and len(y_s.unique()) == 2:
                    out.append("woe")
                else:
                    out.append("mdlp")
            else:
                out.append("quantile" if abs(float(skew)) > 1.0 else "fd")
        else:
            out.append("quantile" if abs(float(skew)) > 1.0 else "fd")
        return out

    @staticmethod
    def _clean_edges(edges: np.ndarray) -> np.ndarray:
        e = np.asarray(edges, dtype=float)
        e = e[np.isfinite(e)]
        if e.size < 2:
            return np.array([0.0, 1.0], dtype=float)
        e = np.unique(np.sort(e))
        if e.size < 2:
            return np.array([e[0], e[0] + 1.0], dtype=float)
        # Ensure strict numeric coverage with tiny padding.
        e[0] = e[0] - 1e-12
        e[-1] = e[-1] + 1e-12
        return e

    def _digitize_edges(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        b = np.digitize(x, edges) - 1
        return np.clip(b, 0, len(edges) - 2)

    def _enforce_min_support_edges(
        self, x: np.ndarray, edges: np.ndarray, min_count: int
    ) -> np.ndarray:
        """Merge weak bins by removing internal edges until support constraints are met."""
        edges = self._clean_edges(edges)
        x_f = self._finite(x)
        if x_f.size == 0 or len(edges) <= 2:
            return edges

        # Iteratively drop one edge from the weakest region.
        while len(edges) > 2:
            b = self._digitize_edges(x_f, edges)
            counts = np.bincount(b, minlength=len(edges) - 1)
            weak = np.where(counts < min_count)[0]
            if weak.size == 0:
                break
            i = int(weak[0])
            # remove left or right edge of weak bin (prefer lower-density side)
            if i == 0:
                drop_edge_idx = 1
            elif i == len(counts) - 1:
                drop_edge_idx = len(edges) - 2
            else:
                left_c = counts[i - 1]
                right_c = counts[i + 1]
                drop_edge_idx = i if left_c <= right_c else i + 1
            edges = np.delete(edges, drop_edge_idx)

        return self._clean_edges(edges)

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

    def _compute_woe_iv(self, x: np.ndarray, y: np.ndarray, edges: np.ndarray) -> Tuple[Dict[int, float], float]:
        """Compute Weight of Evidence and Information Value for a binned feature."""
        # Only for binary classification targets (0/1)
        binned = self._digitize_edges(x, edges)
        df = pd.DataFrame({'bin': binned, 'target': y})
        
        # Calculate event/non-event counts
        total_events = df['target'].sum()
        total_non_events = len(df) - total_events
        
        if total_events == 0 or total_non_events == 0:
            return {i: 0.0 for i in range(len(edges)-1)}, 0.0
            
        stats = df.groupby('bin')['target'].agg(['sum', 'count'])
        stats['non_sum'] = stats['count'] - stats['sum']
        
        # Avoid division by zero with small smoothing
        eps = 1e-6
        stats['dist_event'] = (stats['sum'] + eps) / (total_events + eps)
        stats['dist_non_event'] = (stats['non_sum'] + eps) / (total_non_events + eps)
        
        # WoE = ln(dist_event / dist_non_event)
        stats['woe'] = np.log(stats['dist_event'] / stats['dist_non_event'])
        
        # IV = sum((dist_event - dist_non_event) * WoE)
        iv = ((stats['dist_event'] - stats['dist_non_event']) * stats['woe']).sum()
        
        return stats['woe'].to_dict(), float(iv)

    # ---------- fit/transform ----------

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BinningTransformer":
        if not getattr(self.config, "create_binning", True):
            self.is_fitted = True
            return self

        self.numerical_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.fill_values_.clear()

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
            min_count = max(self.min_samples_per_bin, int(np.ceil(n * self.min_bin_fraction)))

            self.binning_transformers_.setdefault(col, {})
            self.fill_values_[col] = float(pd.to_numeric(X[col], errors="coerce").median())

            y_col = None
            if y is not None:
                y_col = pd.to_numeric(pd.Series(y).reindex(data.index), errors="coerce").values
            col_strategies = self._resolve_strategies_for_col(col_values, y_col)

            for strategy in col_strategies:
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
                        # enforce minimum support by adapting k down if needed
                        while k > 2:
                            b = disc.transform(col_values.reshape(-1, 1)).flatten().astype(int)
                            counts = np.bincount(b, minlength=k)
                            if counts.min() >= min_count:
                                break
                            k = max(2, k - 1)
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
                        while k > 2:
                            b = disc.transform(col_values.reshape(-1, 1)).flatten().astype(int)
                            counts = np.bincount(b, minlength=k)
                            if counts.min() >= min_count:
                                break
                            k = max(2, k - 1)
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
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }

                    elif strategy == "doane":
                        k = max(2, self._doane_bins(col_values))
                        k = int(np.clip(k, 2, dynamic_max_bins))
                        edges = self._equal_width_edges(col_values, k)
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }

                    elif strategy == "shimazaki":
                        k = max(
                            2, self._shimazaki_bins(col_values, *self.shimazaki_k_range)
                        )
                        k = int(np.clip(k, 2, dynamic_max_bins))
                        edges = self._equal_width_edges(col_values, k)
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
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
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
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
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }

                    elif strategy == "woe":
                        if y is None:
                            continue
                        y_arr = pd.Series(y).reindex(data.index).values
                        # For binary classification only
                        if len(np.unique(y_arr)) != 2:
                            continue
                        
                        # Use MDLP edges as base
                        edges = self._mdlp_edges(col_values, y_arr, max_bins=dynamic_max_bins)
                        edges = self._enforce_min_support_edges(col_values, edges, min_count)
                        woe_map, iv = self._compute_woe_iv(col_values, y_arr, edges)
                        
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "woe_map": woe_map,
                            "iv": iv,
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
            fill_val = self.fill_values_.get(col, float(col_data.median()))
            col_data = col_data.fillna(fill_val)
            X_col = col_data.values.reshape(-1, 1)

            for strategy, info in strat_info.items():
                try:
                    if "edges" in info:
                        edges = info["edges"]
                        binned = np.digitize(col_data.values, edges) - 1
                        binned = np.clip(binned, 0, len(edges) - 2)
                        
                        # If WoE map is available, replace bin index with WoE
                        if "woe_map" in info:
                            binned_vals = pd.Series(binned).map(info["woe_map"]).values
                            binned = binned_vals
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
