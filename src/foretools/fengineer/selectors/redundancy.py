"""Redundancy pruning via greedy correlation thresholding."""

from __future__ import annotations

import numpy as np
import pandas as pd


class RedundancyPruner:
    """Greedy correlation-based feature redundancy pruning.

    Given a *ranked* list of features, iteratively keeps each feature
    unless it is already ``threshold``-correlated (absolute Pearson)
    with something earlier in the ranking.

    Parameters
    ----------
    threshold:
        Correlation threshold above which a feature is considered redundant.
    pool_size:
        Only consider the top *n* features for pruning.  Features beyond
        the pool are kept as-is.
    """

    def __init__(
        self,
        threshold: float = 0.98,
        pool_size: int = 200,
    ) -> None:
        self.threshold = threshold
        self.pool_size = pool_size

    def prune(self, X: pd.DataFrame, ranked_features: list[str]) -> list[str]:
        """Remove redundant features from a ranked list.

        Returns a pruned list preserving the original ranking order.

        Parameters
        ----------
        X:
            DataFrame with the ranked features (any subset is fine,
            columns not present are silently skipped).
        ranked_features:
            Feature names in decreasing importance order.

        Returns
        -------
        list[str]
            Pruned feature list.
        """
        if len(ranked_features) < 2:
            return ranked_features

        pool = ranked_features[: self.pool_size]
        corr_thr = self.threshold
        Xp = X[pool]
        corr_matrix = Xp.corr().abs().fillna(0.0).values

        kept: list[str] = []
        kept_cols: list[int] = []
        for i, f in enumerate(pool):
            drop = False
            for j in kept_cols:
                if corr_matrix[i, j] >= corr_thr:
                    drop = True
                    break
            if not drop:
                kept.append(f)
                kept_cols.append(i)

        # Features outside pool are kept in original order
        tail = [f for f in ranked_features if f not in pool]
        return kept + tail
