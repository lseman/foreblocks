"""Minimum Description Length Principle (MDLP) binning.

Supervised binning using the Fayyad-Irani MDL stopping criterion.
Recursively splits where information gain exceeds the MDL threshold.
"""

from __future__ import annotations

import warnings
from math import log2
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class MDLPTransformer(BaseFeatureTransformer):
    """
    MDLP (Fayyad-Irani) supervised binning for numerical features.

    Recursively splits features at points that maximize information gain
    while satisfying the MDL stopping criterion. Returns bin indices
    suitable for one-hot encoding or direct use.

    Parameters
    ----------
    config:
        FeatureConfig instance.
    max_bins:
        Maximum number of bins (default 50).
    min_samples_per_bin:
        Minimum samples per bin (default 10).
    """

    def __init__(self, config: Any, max_bins: int = 50, min_samples_per_bin: int = 10):
        super().__init__(config)
        self.max_bins = max_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.binning_configs_: dict[str, dict] = {}
        self.numerical_cols_: list[str] = []
        self.fill_values_: dict[str, float] = {}
        self.is_fitted = False

    @staticmethod
    def _entropy(labels: np.ndarray) -> float:
        _, cnt = np.unique(labels, return_counts=True)
        p = cnt / cnt.sum()
        return -float((p * np.log2(p + 1e-12)).sum())

    @staticmethod
    def _candidate_splits(x: np.ndarray, y: np.ndarray) -> list[float]:
        splits = []
        for i in range(1, len(x)):
            if x[i] != x[i - 1] and y[i] != y[i - 1]:
                splits.append((x[i - 1] + x[i]) / 2.0)
        return splits

    def _mdlp_edges(self, x: np.ndarray, y: np.ndarray, max_bins: int) -> np.ndarray:
        """Compute MDLP cut points."""
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size == 0:
            return np.array([0.0, 1.0])

        order = np.argsort(x)
        x, y = x[order], y[order]

        def mdl_stop(parent_y, left_y, right_y, N, k):
            H_parent = self._entropy(parent_y)
            H_left, H_right = self._entropy(left_y), self._entropy(right_y)
            Nl, Nr = len(left_y), len(right_y)
            gain = H_parent - (Nl / N) * H_left - (Nr / N) * H_right

            k_left = len(np.unique(left_y))
            k_right = len(np.unique(right_y))
            delta = log2(3**k - 2) - (
                k * H_parent - k_left * H_left - k_right * H_right
            )
            threshold = (log2(max(N - 1, 1)) + delta) / max(N, 1)
            return gain > threshold, gain

        def split_rec(xv, yv):
            if len(np.unique(yv)) <= 1 or xv.size < 2:
                return []
            N = xv.size
            k = len(np.unique(yv))
            splits = self._candidate_splits(xv, yv)
            if not splits:
                return []

            best_s, best_gain = None, -np.inf
            for s in splits:
                left_mask = xv <= s
                yl, yr = yv[left_mask], yv[~left_mask]
                if yl.size == 0 or yr.size == 0:
                    continue
                ok, gain = mdl_stop(yv, yl, yr, N, k)
                if ok and gain > best_gain:
                    best_gain, best_s = gain, s

            if best_s is None:
                return []
            left = xv <= best_s
            right = ~left
            return (
                split_rec(xv[left], yv[left])
                + [best_s]
                + split_rec(xv[right], yv[right])
            )

        cut_points = split_rec(x, y)
        cut_points = sorted(set(cut_points))

        if len(cut_points) + 1 > max_bins:
            step = max(1, int(np.ceil(len(cut_points) / (max_bins - 1))))
            cut_points = cut_points[::step][: max(0, max_bins - 1)]

        return np.array([x.min() - 1e-12] + cut_points + [x.max() + 1e-12], dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MDLPTransformer":
        if y is None:
            raise ValueError("MDLP requires a target y.")

        self.numerical_cols_ = self.get_numerical_cols(X)
        self.binning_configs_.clear()
        self.fill_values_.clear()

        y_arr = pd.to_numeric(y, errors="coerce").to_numpy()

        for col in self.numerical_cols_:
            if col not in X.columns:
                continue
            data = X[col].dropna()
            if data.empty or data.nunique(dropna=True) <= 1:
                continue

            self.fill_values_[col] = float(X[col].median(skipna=True))
            x_vals = pd.to_numeric(X[col], errors="coerce").to_numpy()
            y_reindexed = pd.Series(y_arr).reindex(X.index).to_numpy()

            try:
                edges = self._mdlp_edges(x_vals, y_reindexed, max_bins=self.max_bins)
                # Enforce min support
                valid_edges = self._enforce_min_support(x_vals, edges)
                self.binning_configs_[col] = {
                    "edges": valid_edges,
                    "n_bins": len(valid_edges) - 1,
                }
            except Exception as e:
                warnings.warn(f"MDLP failed for {col}: {e}")

        self.is_fitted = True
        return self

    def _enforce_min_support(self, x: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """Merge weak bins until support constraints are met."""
        edges = np.asarray(edges, dtype=float)
        x_f = x[np.isfinite(x)]
        if x_f.size == 0 or len(edges) <= 2:
            return edges

        while len(edges) > 2:
            b = np.digitize(x_f, edges) - 1
            b = np.clip(b, 0, len(edges) - 2)
            counts = np.bincount(b, minlength=len(edges) - 1)
            weak = np.where(counts < self.min_samples_per_bin)[0]
            if weak.size == 0:
                break
            i = weak[0]
            if i == 0:
                drop_idx = 1
            elif i == len(counts) - 1:
                drop_idx = len(edges) - 2
            else:
                left_c = counts[i - 1]
                right_c = counts[i + 1]
                drop_idx = i if left_c <= right_c else i + 1
            edges = np.delete(edges, drop_idx)

        return edges

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        features = {}
        for col, info in self.binning_configs_.items():
            if col not in X.columns:
                continue
            col_data = pd.to_numeric(X[col], errors="coerce")
            col_data = col_data.fillna(self.fill_values_.get(col, 0.0))
            edges = info["edges"]
            binned = np.digitize(col_data.values, edges) - 1
            binned = np.clip(binned, 0, len(edges) - 2)
            features[f"{col}_mdlp"] = binned.astype(int)
        return pd.DataFrame(features, index=X.index)

    def get_bin_edges(self, col: str) -> np.ndarray | None:
        """Return bin edges for a column."""
        info = self.binning_configs_.get(col)
        if info is None:
            return None
        return info["edges"]
