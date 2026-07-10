"""Weight of Evidence (WoE) encoding for binary classification.

Computes per-category WoE values with smoothing, and optionally Information
Value (IV) for feature selection.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class WeightOfEvidenceTransformer(BaseFeatureTransformer):
    """
    Weight of Evidence encoding for binary classification.

    For each category, computes:
        WoE = ln(P(event|category) / P(non-event|category))

    With Laplace smoothing to handle zero counts.

    Parameters
    ----------
    config:
        FeatureConfig instance.
    smoothing:
        Laplace smoothing factor (default 1e-6).
    """

    def __init__(self, config: Any, smoothing: float = 1e-6):
        super().__init__(config)
        self.smoothing = smoothing
        self.woe_map_: dict[str, dict[str, float]] = {}
        self.iv_: dict[str, float] = {}
        self.global_event_rate_ = 0.0
        self.categorical_cols_: list[str] = []
        self.is_fitted = False

    def _is_binary(self, y: pd.Series) -> bool:
        unique = np.unique(pd.to_numeric(y, errors="coerce").dropna())
        return len(unique) == 2

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "WeightOfEvidenceTransformer":
        if y is None:
            raise ValueError("WoE requires a binary target y.")

        y_num = pd.to_numeric(y, errors="coerce")
        if not self._is_binary(y_num):
            raise ValueError("WoE requires a binary target (exactly 2 unique values).")

        cats = self.get_categorical_cols(X)
        self.categorical_cols_ = cats

        # Global event rate (assume higher value = event)
        valid = y_num.notna()
        if valid.sum() < 2:
            raise ValueError("Not enough valid target values for WoE.")

        y_valid = y_num[valid]
        event_val = y_valid.max()
        self.global_event_rate_ = float((y_valid == event_val).mean())

        for col in cats:
            if col not in X.columns:
                continue
            s = pd.Series(X[col]).astype(str).replace("nan", "MISSING")
            combined = pd.DataFrame({"cat": s, "y": y_num})
            valid_rows = combined.dropna()
            if len(valid_rows) < 5:
                continue

            # Event/non-event counts per category
            total_events = (valid_rows["y"] == event_val).sum()
            total_non_events = len(valid_rows) - total_events

            if total_events == 0 or total_non_events == 0:
                warnings.warn(f"Column {col}: all events or all non-events. Skipping.")
                continue

            grp = valid_rows.groupby("cat")["y"]
            counts = grp.agg(["sum", "count"])
            counts["non_sum"] = counts["count"] - counts["sum"]

            eps = self.smoothing
            dist_event = (counts["sum"] + eps) / (total_events + 2 * eps)
            dist_non_event = (counts["non_sum"] + eps) / (total_non_events + 2 * eps)

            woe = np.log(dist_event / dist_non_event)

            self.woe_map_[col] = woe.to_dict()
            self.iv_[col] = float(((dist_event - dist_non_event) * woe).sum())

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        features = {}
        prior = np.log(self.global_event_rate_ / (1 - self.global_event_rate_ + 1e-12))

        for col in self.categorical_cols_:
            if col not in X.columns or col not in self.woe_map_:
                continue
            s = pd.Series(X[col]).astype(str).replace("nan", "MISSING")
            woe_vals = s.map(self.woe_map_[col]).fillna(prior)
            features[f"{col}_woe"] = woe_vals.to_numpy()

        return pd.DataFrame(features, index=X.index)

    def get_feature_iv(self) -> pd.Series:
        """Return Information Value per feature."""
        return pd.Series(self.iv_, name="iv").sort_values(ascending=False)

    def get_woe_map(self, col: str) -> dict[str, float] | None:
        """Return the WoE mapping for a specific column."""
        return self.woe_map_.get(col)
