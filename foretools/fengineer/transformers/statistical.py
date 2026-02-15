from typing import Optional

import numpy as np
import pandas as pd

from .base import BaseFeatureTransformer


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


