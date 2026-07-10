"""Mutual-information-based feature selector with optional stability.

This selector computes MI scores (fast or stable via CV folds), applies a
threshold, and optionally prunes redundant features via correlation.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from foretools.aux.adaptive_mi import AdaptiveMI

from .base import FeatureSelectorABC
from .redundancy import RedundancyPruner

if TYPE_CHECKING:
    pass


class MISelector(FeatureSelectorABC):
    """Mutual-information-based feature selector with optional stability.

    Parameters
    ----------
    config:
        Configuration object with attribute access.
    scorer:
        AdaptiveMI scorer instance.  A new one is created if omitted.
    threshold:
        MI score threshold.  Features below this are dropped.
    use_stable:
        If ``True``, compute MI over CV folds and require a minimum
        selection frequency before counting a feature as relevant.
    cv:
        Number of CV folds for stable mode.
    min_freq:
        Minimum fraction of folds in which a feature must have non-zero MI
        to be retained (stable mode only).
    redundancy_pruner:
        Optional :class:`RedundancyPruner` for correlation-based pruning.
    min_features:
        Minimum number of features to keep (fallback if threshold selects none).
    max_features:
        Maximum number of features to keep after thresholding + pruning.
    task:
        ``"regression"`` or ``"classification"``.
    random_state:
        Random seed for CV splits.
    """

    def __init__(
        self,
        config: Any,
        scorer: AdaptiveMI | None = None,
        threshold: float = 0.01,
        use_stable: bool = True,
        cv: int = 5,
        min_freq: float = 0.5,
        redundancy_pruner: RedundancyPruner | None = None,
        min_features: int = 1,
        max_features: int | None = None,
        task: str = "regression",
        random_state: int = 42,
    ) -> None:
        self.config = config
        self.scorer = scorer or AdaptiveMI(random_state=random_state)
        self.threshold = threshold
        self.use_stable = use_stable
        self.cv = cv
        self.min_freq = min_freq
        self.redundancy_pruner = redundancy_pruner
        self.min_features = min_features
        self.max_features = max_features
        self.task = task
        self.random_state = random_state
        self._selected: list[str] = []
        self._scores: pd.Series | None = None

    # ── FeatureSelectorABC interface ────────────────────────────────────

    @property
    def selection_method(self) -> str:
        return "mi"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MISelector":
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            self._selected = []
            self._scores = pd.Series(dtype=float)
            return self

        X_clean, y_clean = MISelector._prepare(X[numerical_cols], y, self.task)
        if len(X_clean) < 10:
            self._selected = numerical_cols
            self._scores = pd.Series(0.0, index=numerical_cols)
            return self

        if self.use_stable:
            self._scores = self._stable_scores(X_clean, y_clean)
        else:
            self._scores = self._fast_scores(X_clean, y_clean)

        selected_mask = self._scores > self.threshold
        if selected_mask.any():
            self._selected = self._scores[selected_mask].index.tolist()
        else:
            n = max(1, self.min_features)
            self._selected = self._scores.head(n).index.tolist()

        if self.redundancy_pruner is not None and self._selected:
            self._selected = self.redundancy_pruner.prune(X_clean, self._selected)

        max_f = self.max_features or len(numerical_cols)
        if len(self._selected) > max_f:
            self._selected = self._selected[:max_f]

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        avail = [f for f in self._selected if f in X.columns]
        if not avail:
            return pd.DataFrame(index=X.index)
        return X[avail]

    def get_selected_features(self) -> list[str]:
        return list(self._selected)

    def get_scores(self) -> pd.Series | None:
        return self._scores

    # ── internals ───────────────────────────────────────────────────────

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        if y.dtype == "object" or str(y.dtype).startswith("category"):
            return True
        k = y.nunique(dropna=True)
        n = max(1, len(y))
        return (k <= 20) or (k / n < 0.05)

    @staticmethod
    def _prepare(
        X: pd.DataFrame, y: pd.Series, task: str
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Align indices, encode classification targets, drop invalid rows."""
        common = X.index.intersection(y.index)
        X_a = X.loc[common]
        y_a = y.loc[common]

        if task == "classification":
            if not pd.api.types.is_numeric_dtype(y_a):
                le = LabelEncoder()
                y_c = pd.Series(le.fit_transform(y_a), index=y_a.index)
            else:
                y_c = y_a.astype("int32")
        else:
            y_c = pd.to_numeric(y_a, errors="coerce")

        valid = y_c.notna() & np.isfinite(y_c)
        if not valid.any():
            return pd.DataFrame(), pd.Series(dtype=float)
        return X_a[valid].astype("float32", copy=False), y_c[valid]

    def _fast_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        scores = self.scorer.score_pairwise(X.values, y.values)
        return pd.Series(scores, index=X.columns).fillna(0.0).clip(lower=0.0)

    def _stable_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Compute MI over CV folds; require minimum frequency to retain."""
        n = len(X)
        cv = max(2, min(self.cv, max(2, n // 20)))
        if cv < 2:
            return self._fast_scores(X, y)

        if self._is_classification(y):
            splitter = StratifiedKFold(cv, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(cv, shuffle=True, random_state=self.random_state)

        names = list(X.columns)
        fold_scores: dict[str, list[float]] = {c: [] for c in names}
        freq: dict[str, int] = {c: 0 for c in names}

        for tr_idx, _ in splitter.split(X, y):
            Xf = X.iloc[tr_idx]
            yf = y.iloc[tr_idx]
            s = self._fast_scores(Xf, yf)
            for c, v in s.items():
                if np.isfinite(v):
                    fold_scores[c].append(float(v))
                    if v > 0:
                        freq[c] += 1

        min_freq = int(np.ceil(self.min_freq * cv))
        agg = {}
        for c in names:
            if not fold_scores[c] or freq[c] < min_freq:
                agg[c] = 0.0
            else:
                agg[c] = float(np.median(fold_scores[c]))

        return pd.Series(agg, index=names).fillna(0.0).clip(lower=0.0)
