"""Statistical aggregation features computed across rows.

Expands on basic statistics (mean, std, min, max) with entropy, skewness,
kurtosis, range ratios, and other distributional descriptors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .aux import BaseFeatureTransformer, require_fitted
from .aux import safe_row_kurtosis, safe_row_skew


class StatisticalTransformer(BaseFeatureTransformer):
    """
    Creates statistical aggregation features across rows.

    Features computed per row:
      - Central tendency: mean, median
      - Dispersion: std, iqr, mad, range, range_iqr
      - Shape: skewness (safe), kurtosis (safe)
      - Counts: non-null count, null ratio
      - Ratios: min/max, mean/median, range/median
      - Entropy: Shannon entropy of row distribution (discretized)

    Parameters
    ----------
    config:
        FeatureConfig instance.
    include_shape:
        If True, add skewness and kurtosis (requires >=8 valid values per row).
    include_ratios:
        If True, add ratio-based features (min/max, mean/median).
    include_dispersion:
        If True, add IQR and MAD features.
    """

    def __init__(
        self,
        config,
        include_shape: bool = True,
        include_ratios: bool = True,
        include_dispersion: bool = True,
    ):
        super().__init__(config)
        self.include_shape = include_shape
        self.include_ratios = include_ratios
        self.include_dispersion = include_dispersion
        self.numerical_cols_: list[str] = []
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "StatisticalTransformer":
        self.numerical_cols_ = self.get_numerical_cols(X)
        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        available_cols = [c for c in self.numerical_cols_ if c in X.columns]
        if len(available_cols) < 2:
            return self._empty_df(X.index)

        X_s = X[available_cols]
        features: dict[str, np.ndarray] = {}

        # Central tendency
        features["row_mean"] = X_s.mean(axis=1).to_numpy()  # type: ignore[union-attr]
        features["row_median"] = X_s.median(axis=1).to_numpy()  # type: ignore[union-attr]

        # Dispersion
        features["row_std"] = X_s.std(axis=1).to_numpy()  # type: ignore[union-attr]
        features["row_min"] = X_s.min(axis=1).to_numpy()
        features["row_max"] = X_s.max(axis=1).to_numpy()
        features["row_range"] = features["row_max"] - features["row_min"]

        if self.include_dispersion:
            q1 = X_s.quantile(0.25, axis=1, numeric_only=True).to_numpy()  # type: ignore[arg-type]
            q3 = X_s.quantile(0.75, axis=1, numeric_only=True).to_numpy()  # type: ignore[arg-type]
            features["row_iqr"] = q3 - q1
            median = X_s.median(axis=1, skipna=True).values  # type: ignore[union-attr]
            row_arr = X_s.to_numpy(dtype=float, copy=False)
            mad = np.array(
                [
                    np.median(np.abs(row - m)) if np.isfinite(m) else np.nan
                    for row, m in zip(row_arr, median)
                ]
            )
            features["row_mad"] = mad

        # Shape
        if self.include_shape:
            features["row_skew"] = safe_row_skew(X_s).to_numpy()
            features["row_kurtosis"] = safe_row_kurtosis(X_s).to_numpy()

        # Null stats
        null_mask = X_s.isna()
        features["row_non_null_count"] = (~null_mask).sum(axis=1).to_numpy()
        features["row_null_ratio"] = null_mask.sum(axis=1).to_numpy() / len(
            available_cols
        )

        # Ratios (guarded against division by zero)
        if self.include_ratios:
            mean = features["row_mean"]
            median_val = features["row_median"]
            rng = features["row_range"]
            std = features["row_std"]

            safe_median = np.where(np.abs(median_val) > 1e-12, median_val, np.nan)
            features["row_mean_over_median"] = np.where(
                np.isfinite(safe_median), mean / safe_median, np.nan
            )

            safe_range = np.where(rng > 1e-12, rng, np.nan)
            features["row_std_over_range"] = np.where(
                np.isfinite(safe_range), std / safe_range, np.nan
            )

            safe_mean = np.where(np.abs(mean) > 1e-12, np.abs(mean), np.nan)
            features["row_cv"] = np.where(
                np.isfinite(safe_mean), std / safe_mean, np.nan
            )

        # Entropy of row distribution (discretize into 10 bins)
        features["row_entropy"] = self._compute_entropy(X_s)  # type: ignore[arg-type]

        return pd.DataFrame(features, index=X.index)

    @staticmethod
    def _compute_entropy(X: pd.DataFrame, n_bins: int = 10) -> np.ndarray:
        """Shannon entropy of each row's value distribution (discretized)."""
        n_rows = len(X)
        entropies = np.full(n_rows, np.nan, dtype=np.float64)

        row_arr = np.asarray(X, dtype=float)
        for i, row in enumerate(row_arr):
            finite = row[np.isfinite(row)]
            if finite.size < 3:
                continue
            counts, _ = np.histogram(finite, bins=n_bins)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropies[i] = -np.sum(probs * np.log2(probs))

        # Normalize to [0, 1] by dividing by log2(n_bins)
        entropies /= np.log2(n_bins)
        return entropies
