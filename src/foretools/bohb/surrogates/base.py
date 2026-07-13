"""Abstract surrogate model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Surrogate(ABC):
    """Base class for surrogate models (KDE, GP, etc.)."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit surrogate to data.

        Args:
            X: Training inputs [n_samples, n_features]
            y: Training outputs [n_samples]
        """

    @abstractmethod
    def predict(
        self, X: np.ndarray, return_std: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Predict at test points.

        Args:
            X: Test inputs [n_samples, n_features]
            return_std: If True, return (mu, sigma); else just mu

        Returns:
            (mu, sigma) if return_std=True, else mu
        """

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Whether model has been fit to data."""
