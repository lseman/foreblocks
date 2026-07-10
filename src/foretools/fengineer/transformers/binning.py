"""Adaptive binning transformer for numerical features.

Delegates strategy computation (FD, Doane, Shimazaki, K-Means, quantile,
uniform, MDLP, WoE) to the standalone ``binning_strategies`` module.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .aux import (
    BaseFeatureTransformer,
    require_fitted,
    doane_edges,
    fd_edges,
    kmeans_edges,
    quantile_transformer,
    shimazaki_edges,
    uniform_transformer,
    woe_edges_and_map,
)

if TYPE_CHECKING:
    pass


class BinningTransformer(BaseFeatureTransformer):
    """
    Adaptive binning with multiple strategies.

    Strategies (auto-selects when "auto" in list):
      - 'fd'           : Freedman–Diaconis (IQR-based)
      - 'doane'        : Doane's rule (skew-adjusted Sturges)
      - 'shimazaki'    : Shimazaki–Shinomoto optimal bin count
      - 'kmeans'       : 1D K-Means edges (handles multi-modality)
      - 'quantile'     : mass-balanced via KBinsDiscretizer
      - 'uniform'      : equal-width via KBinsDiscretizer
      - 'mdlp'         : supervised Fayyad–Irani MDL (requires y)
      - 'woe'          : WoE encoding (binary classification, uses MDLP)
    """

    def __init__(
        self,
        config: Any,
        min_samples_per_bin: int = 10,
        max_bins: int = 100,
        strategies: list[str] | None = None,
        kmeans_bins: int = 10,
        shimazaki_k_range: tuple[int, int] = (4, 128),
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
            self.strategies = ["auto"]

        self.auto_supervised = bool(
            getattr(self.config, "binning_auto_supervised", True)
        )
        self.min_bin_fraction = float(
            getattr(self.config, "binning_min_bin_fraction", 0.01)
        )

        self.binning_transformers_: dict[str, dict[str, Any]] = {}
        self.numerical_cols_: list[str] = []
        self.fill_values_: dict[str, float] = {}
        self.is_fitted = False

    @staticmethod
    def _is_classification_target(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        n = max(1, len(y))
        k = y.nunique(dropna=True)
        return (k <= 20) or (k / n < 0.05)

    def _resolve_strategies_for_col(
        self, x: np.ndarray, y_col: np.ndarray | None
    ) -> list[str]:
        from .aux import safe_skew

        strategies = [str(s).lower() for s in self.strategies]
        if "auto" not in strategies:
            return strategies

        out = [s for s in strategies if s != "auto"]
        x_f = x[np.isfinite(x)]
        if x_f.size < 20:
            out.append("uniform")
            return out

        skew = float(safe_skew(x_f, bias=False)) if x_f.size >= 8 else 0.0

        if (
            self.auto_supervised
            and y_col is not None
            and np.isfinite(y_col).sum() >= 50
        ):
            y_s = pd.Series(y_col)
            if self._is_classification_target(y_s):
                if getattr(self.config, "use_woe", False) and len(y_s.unique()) == 2:
                    out.append("woe")
                else:
                    out.append("mdlp")
            else:
                out.append("quantile" if abs(skew) > 1.0 else "fd")
        else:
            out.append("quantile" if abs(skew) > 1.0 else "fd")

        return out

    def _fit_strategy(
        self,
        col_values: np.ndarray,
        strategy: str,
        n: int,
        dynamic_max_bins: int,
        min_count: int,
    ) -> dict[str, Any] | None:
        """Fit a single strategy and return its config dict."""
        n_bins = int(getattr(self.config, "n_bins", 10))

        if strategy == "quantile":
            tr = quantile_transformer(
                col_values,
                n_bins=n_bins,
                min_count=min_count,
                subsample=min(20000, n),
            )
            if tr is None:
                return None
            binned = tr.transform(col_values.reshape(-1, 1)).flatten().astype(int)
            actual_bins = len(np.unique(binned))
            return {"transformer": tr, "n_bins": actual_bins}

        if strategy == "uniform":
            tr = uniform_transformer(
                col_values,
                n_bins=n_bins,
                min_count=min_count,
                subsample=min(20000, n),
            )
            if tr is None:
                return None
            binned = tr.transform(col_values.reshape(-1, 1)).flatten().astype(int)
            actual_bins = len(np.unique(binned))
            return {"transformer": tr, "n_bins": actual_bins}

        if strategy == "fd":
            edges = fd_edges(col_values, self.max_bins)
            edges = self._enforce_support(col_values, edges, min_count)
            return {"edges": edges, "n_bins": len(edges) - 1}

        if strategy == "doane":
            edges = doane_edges(col_values, self.max_bins)
            edges = self._enforce_support(col_values, edges, min_count)
            return {"edges": edges, "n_bins": len(edges) - 1}

        if strategy == "shimazaki":
            edges = shimazaki_edges(
                col_values,
                max_bins=self.max_bins,
                kmin=self.shimazaki_k_range[0],
                kmax=self.shimazaki_k_range[1],
            )
            edges = self._enforce_support(col_values, edges, min_count)
            return {"edges": edges, "n_bins": len(edges) - 1}

        if strategy == "kmeans":
            k = int(np.clip(n_bins, 2, dynamic_max_bins))
            edges = kmeans_edges(col_values, k, self.random_state)
            edges = self._enforce_support(col_values, edges, min_count)
            return {"edges": edges, "n_bins": len(edges) - 1}

        if strategy == "mdlp":
            return None  # handled in fit() with y

        if strategy == "woe":
            return None  # handled in fit() with y

        warnings.warn(f"Unknown binning strategy: {strategy}. Skipping.")
        return None

    @staticmethod
    def _enforce_support(
        x: np.ndarray, edges: np.ndarray, min_count: int
    ) -> np.ndarray:
        """Merge weak bins until support constraints are met."""
        from .aux import _enforce_min_support_edges

        return _enforce_min_support_edges(x, edges, min_count)

    # ── fit/transform ──────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "BinningTransformer":
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

            col_min, col_max = float(np.min(col_values)), float(np.max(col_values))
            if (col_max - col_min) < 1e-12:
                continue

            n_cap = max(1, n // max(self.min_samples_per_bin, 1))
            dynamic_max_bins = int(
                np.clip(min(self.max_bins, n_cap, n_unique), 2, self.max_bins)
            )
            min_count = max(
                self.min_samples_per_bin, int(np.ceil(n * self.min_bin_fraction))
            )

            self.binning_transformers_.setdefault(col, {})
            self.fill_values_[col] = float(
                pd.to_numeric(X[col], errors="coerce").median()
            )

            y_col = None
            if y is not None:
                y_col = pd.to_numeric(
                    pd.Series(y).reindex(data.index), errors="coerce"
                ).values

            col_strategies = self._resolve_strategies_for_col(col_values, y_col)

            for strategy in col_strategies:
                try:
                    # Special handling for supervised strategies
                    if strategy == "mdlp":
                        if y is None:
                            continue
                        y_arr = pd.Series(y).reindex(data.index).values
                        from .aux import mdlp_edges

                        edges = mdlp_edges(col_values, y_arr, max_bins=dynamic_max_bins)
                        edges = self._enforce_support(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "n_bins": len(edges) - 1,
                        }
                        continue

                    if strategy == "woe":
                        if y is None:
                            continue
                        y_arr = pd.Series(y).reindex(data.index).values
                        if len(np.unique(y_arr)) != 2:
                            continue
                        edges, woe_map, iv = woe_edges_and_map(
                            col_values, y_arr, max_bins=dynamic_max_bins
                        )
                        edges = self._enforce_support(col_values, edges, min_count)
                        self.binning_transformers_[col][strategy] = {
                            "edges": edges,
                            "woe_map": woe_map,
                            "iv": iv,
                            "n_bins": len(edges) - 1,
                        }
                        continue

                    result = self._fit_strategy(
                        col_values, strategy, n, dynamic_max_bins, min_count
                    )
                    if result is not None:
                        self.binning_transformers_[col][strategy] = result

                except Exception as e:
                    warnings.warn(f"Binning fit failed for {col} ({strategy}): {e}")
                    continue

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        if (
            not getattr(self.config, "create_binning", True)
            or not self.binning_transformers_
        ):
            return pd.DataFrame(index=X.index)

        features = {}
        for col, strat_info in self.binning_transformers_.items():
            if col not in X.columns:
                continue
            col_data = pd.to_numeric(X[col], errors="coerce").fillna(
                self.fill_values_.get(col, 0.0)
            )
            for strategy, info in strat_info.items():
                try:
                    if "edges" in info:
                        edges = info["edges"]
                        binned = np.digitize(col_data.values, edges) - 1
                        binned = np.clip(binned, 0, len(edges) - 2)
                        if "woe_map" in info:
                            binned = pd.Series(binned).map(info["woe_map"]).values
                    else:
                        transformer = info["transformer"]
                        binned = transformer.transform(col_data.values.reshape(-1, 1))
                        if hasattr(binned, "toarray"):
                            binned = binned.toarray().flatten()
                        else:
                            binned = np.asarray(binned).flatten()

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
