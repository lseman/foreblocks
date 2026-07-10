"""Polynomial feature transformer.

Generates polynomial features (powers, roots, logs, reciprocals) for
numerical columns. Designed as a standalone transformer separate from
InteractionTransformer.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .aux import BaseFeatureTransformer, require_fitted

if TYPE_CHECKING:
    pass


class PolynomialTransformer(BaseFeatureTransformer):
    """
    Polynomial feature generator.

    Creates the following features per input column:
      - Squared (x²)
      - Cubed (x³)
      - Square root (√x, for x >= 0)
      - Reciprocal (1/x)
      - Log (log(1+x), for x > -1)

    Each transform is only created if the result has sufficient variance
    and is finite.

    Parameters
    ----------
    config:
        FeatureConfig instance.
    max_features:
        Maximum total polynomial features to create (default 50).
    min_variance:
        Minimum variance threshold for a polynomial feature (default 1e-6).
    """

    def __init__(
        self,
        config: Any,
        max_features: int = 50,
        min_variance: float = 1e-6,
    ):
        super().__init__(config)
        self.max_features = max_features
        self.min_variance = min_variance
        self.selected_features_: list[tuple[str, str]] = []  # (col, power_name)
        self.numerical_cols_: list[str] = []
        self.is_fitted = False

        self._transforms = {
            "squared": (True, 2.0),
            "cubed": (False, 3.0),
            "sqrt": (False, 0.5),
            "reciprocal": (False, -1.0),
            "log": (False, "log"),
        }

    def _apply_transform(
        self, arr: np.ndarray, power: float | str
    ) -> np.ndarray | None:
        """Apply a polynomial transform, returning None if invalid."""
        with np.errstate(all="ignore"):
            if power == 0.5:
                out = np.where(arr >= 0, np.sqrt(arr), np.nan)
            elif power == "log":
                out = np.where(arr > -1, np.log1p(arr), np.nan)
            elif power == -1.0:
                out = np.divide(
                    1.0,
                    arr,
                    out=np.full_like(arr, np.nan),
                    where=(np.abs(arr) > 1e-12) & np.isfinite(arr),
                )
            else:
                out = np.power(arr, power)
        return np.clip(out, -1e10, 1e10)

    def _clean(self, arr: np.ndarray) -> np.ndarray | None:
        """Sanitize an array: finite check, variance check."""
        arr = np.asarray(arr, dtype=np.float32)
        arr[~np.isfinite(arr)] = np.nan
        finite = np.isfinite(arr)
        if finite.sum() < 10:
            return None
        if np.nanvar(arr) < self.min_variance:
            return None
        return arr

    def fit(
        self, X: pd.DataFrame, y: pd.Series | None = None
    ) -> "PolynomialTransformer":
        self.numerical_cols_ = self.get_numerical_cols(X)
        self.selected_features_ = []
        candidates: list[tuple[float, str, str]] = []

        for col in self.numerical_cols_:
            if col not in X.columns:
                continue
            arr = self.coerce_numeric(X[col])
            if not self.check_variance(arr, self.min_variance):
                continue

            for name, (flag, power) in self._transforms.items():
                if not flag:
                    continue
                transformed = self._apply_transform(arr, power)
                cleaned = self._clean(transformed)
                if cleaned is not None:
                    variance = float(np.nanvar(cleaned))
                    candidates.append((variance, col, name))

        # Sort by variance descending, take top max_features
        candidates.sort(reverse=True, key=lambda x: x[0])
        self.selected_features_ = [
            (col, name) for _, col, name in candidates[: self.max_features]
        ]

        self.is_fitted = True
        return self

    @require_fitted
    def transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> pd.DataFrame:
        features: dict[str, np.ndarray] = {}
        for col, name in self.selected_features_:
            if col not in X.columns:
                continue
            arr = self.coerce_numeric(X[col])
            power = self._transforms[name][1]
            transformed = self._apply_transform(arr, power)
            cleaned = self._clean(transformed)
            if cleaned is not None:
                features[f"{col}_{name}"] = cleaned
        return (
            pd.DataFrame(features, index=X.index, dtype=np.float32)
            if features
            else self._empty_df(X.index)
        )
