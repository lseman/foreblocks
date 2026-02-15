from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from .config import FeatureConfig


class BaseFeatureTransformer(ABC):
    """Abstract base class for feature transformers."""

    def __init__(self, config: FeatureConfig):
        self.config = config
        self.is_fitted = False

    @abstractmethod
    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "BaseFeatureTransformer":
        """Fit the transformer."""
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data."""
        pass

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)


