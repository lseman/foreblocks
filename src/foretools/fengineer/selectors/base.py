"""Abstract base class for feature selectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    pass


class FeatureSelectorABC(ABC):
    """Abstract base class for all feature selectors.

    Every selector must implement :meth:`fit`, :meth:`transform`,
    :meth:`get_selected_features`, and expose a ``selection_method``
    string property.  A default :meth:`fit_transform` delegates to
    ``fit`` then ``transform``.
    """

    @property
    @abstractmethod
    def selection_method(self) -> str:
        """Human-readable name (e.g. ``"boruta"``, ``"rfecv"``, ``"mi"``)."""
        ...

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FeatureSelectorABC":
        """Fit the selector and determine feature importance."""
        ...

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return *X* with only the selected columns."""
        ...

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fit and transform in one call."""
        self.fit(X, y)
        return self.transform(X)

    @abstractmethod
    def get_selected_features(self) -> list[str]:
        """Return list of selected feature names."""
        ...

    def get_scores(self) -> pd.Series | None:
        """Return feature importance / relevance scores, or ``None``."""
        return None

    def get_feature_scores(self) -> pd.Series | None:
        """Alias of :meth:`get_scores` for backward compatibility."""
        return self.get_scores()

    def get_top_features(self, n: int = 10) -> list[str]:
        """Return the top *n* features by score."""
        scores = self.get_scores()
        if scores is None or scores.empty:
            return self.get_selected_features()[:n]
        return list(scores.head(n).index)
