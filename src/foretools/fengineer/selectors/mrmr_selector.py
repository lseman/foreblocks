"""mRMR (Minimum Redundancy Maximum Relevance) feature selector.

Wraps :class:`foretools.aux.adaptive_mrmr.AdaptiveMRMR` and exposes it
through the :class:`FeatureSelectorABC` interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from foretools.aux.adaptive_mi import AdaptiveMI
from foretools.aux.adaptive_mrmr import AdaptiveMRMR

from .base import FeatureSelectorABC
from .mi_selector import MISelector

if TYPE_CHECKING:
    pass


class MRMRSelector(FeatureSelectorABC):
    """mRMR selector backed by :class:`AdaptiveMRMR`.

    Parameters
    ----------
    config:
        Configuration object with attribute access.
    scorer:
        AdaptiveMI scorer instance.  A new one is created if omitted.
    criterion:
        mRMR criterion: ``"mid"`` (mutual information difference) or
        ``"mu"`` (mutual information union).
    candidate_pool:
        Number of top-MI features to consider during the incremental mRMR
        selection loop.
    redundancy_weight:
        Weight for the redundancy term in the mRMR score.
    redundancy_eps:
        Small epsilon to avoid division by zero in redundancy computation.
    use_raw_mi:
        If ``True``, use raw MI scores instead of the combined mRMR score.
    stable_relevance:
        If ``True``, use CV-fold-stable MI scores for relevance.
    cv:
        Number of CV folds for stable mode.
    min_freq:
        Minimum selection frequency across folds (stable mode).
    task:
        ``"regression"`` or ``"classification"``.
    random_state:
        Random seed.
    """

    def __init__(
        self,
        config: Any,
        scorer: AdaptiveMI | None = None,
        criterion: str = "mid",
        candidate_pool: int = 200,
        redundancy_weight: float = 1.0,
        redundancy_eps: float = 1e-8,
        use_raw_mi: bool = False,
        stable_relevance: bool = True,
        cv: int = 5,
        min_freq: float = 0.5,
        task: str = "regression",
        random_state: int = 42,
    ) -> None:
        self.config = config
        self.scorer = scorer or AdaptiveMI(random_state=random_state)
        self.criterion = criterion
        self.candidate_pool = candidate_pool
        self.redundancy_weight = redundancy_weight
        self.redundancy_eps = redundancy_eps
        self.use_raw_mi = use_raw_mi
        self.stable_relevance = stable_relevance
        self.cv = cv
        self.min_freq = min_freq
        self.task = task
        self.random_state = random_state
        self._selected: list[str] = []
        self._relevance: pd.Series | None = None
        self._selection: pd.Series | None = None
        self._mrmr: AdaptiveMRMR | None = None

    # ── FeatureSelectorABC interface ────────────────────────────────────

    @property
    def selection_method(self) -> str:
        return "mrmr"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MRMRSelector":
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not numerical_cols:
            return self

        X_clean, y_clean = MISelector._prepare(X[numerical_cols], y, self.task)

        self._mrmr = AdaptiveMRMR(
            scorer=self.scorer,
            criterion=self.criterion,
            candidate_pool=self.candidate_pool,
            redundancy_weight=self.redundancy_weight,
            redundancy_eps=self.redundancy_eps,
            use_raw_mi=self.use_raw_mi,
            stable_relevance=self.stable_relevance,
            cv=self.cv,
            min_freq=self.min_freq,
            task=self.task,
            random_state=self.random_state,
        )
        self._mrmr.fit(
            X_clean,
            y_clean,
            min_features=getattr(self.config, "min_features", 1),
            max_features=getattr(self.config, "max_features", X_clean.shape[1]),
            mi_threshold=getattr(self.config, "mi_threshold", 0.01),
            min_samples=getattr(self.config, "min_samples", 10),
        )
        self._selected = list(self._mrmr.selected_features_)
        self._relevance = self._mrmr.relevance_scores_
        self._selection = self._mrmr.selection_scores_
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        avail = [f for f in self._selected if f in X.columns]
        if not avail:
            return pd.DataFrame(index=X.index)
        return X[avail]

    def get_selected_features(self) -> list[str]:
        return list(self._selected)

    def get_scores(self) -> pd.Series | None:
        return self._selection

    def get_relevance_scores(self) -> pd.Series | None:
        return self._relevance
