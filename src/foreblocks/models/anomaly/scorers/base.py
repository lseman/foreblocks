"""Validation and training-only normalization for fitted anomaly scorers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


def positive_int(name: str, value: int, minimum: int = 1) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value < minimum
    ):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def positive_float(name: str, value: float, *, allow_zero: bool = False) -> float:
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(
            f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}."
        )
    return float(value)


def squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # Accumulate feature by feature: no [queries, reference, features] temporary,
    # and no cancellation from ||x||^2 + ||y||^2 - 2*x*y near identical points.
    result = np.zeros((len(x), len(y)))
    for j in range(x.shape[1]):
        result += (x[:, j, None] - y[None, :, j]) ** 2
    return result


class FittedScorer(ABC):
    """Finite matrix/window inputs, fixed preprocessing, larger = more anomalous."""

    def __init__(self, *, standardize: bool = True, seed: int | None = 42):
        self.standardize = standardize
        self.seed = seed
        self._fitted = False

    @staticmethod
    def _array(values: np.ndarray) -> np.ndarray:
        x = np.asarray(values, dtype=np.float64)
        if x.ndim not in (2, 3) or any(size == 0 for size in x.shape[1:]):
            raise ValueError(
                "Expected [samples, features] or [samples, time, channels]."
            )
        if not np.isfinite(x).all():
            raise ValueError("Input must contain only finite values.")
        return x

    def fit(self, values: np.ndarray) -> FittedScorer:
        x = self._array(values)
        if len(x) < 2:
            raise ValueError("At least two training samples are required.")
        self._fitted = False
        self.sample_shape_ = x.shape[1:]
        flat = x.reshape(len(x), -1)
        self.mean_ = flat.mean(axis=0) if self.standardize else np.zeros(flat.shape[1])
        scale = flat.std(axis=0) if self.standardize else np.ones(flat.shape[1])
        self.scale_ = np.where(scale > 1e-12, scale, 1.0)
        normalized = (flat - self.mean_) / self.scale_
        if not np.isfinite(normalized).all():
            raise ValueError("Input magnitude exceeds the supported numeric range.")
        self._fit(normalized)
        self.decision_scores_ = self._validated_scores(self._score(normalized), len(x))
        self._fitted = True
        return self

    @staticmethod
    def _validated_scores(scores, count):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape != (count,) or not np.isfinite(scores).all():
            raise ValueError("Scorer produced nonfinite or incorrectly shaped scores.")
        return scores

    def decision_function(self, values: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scorer is not fitted.")
        x = self._array(values)
        if x.shape[1:] != self.sample_shape_:
            raise ValueError(
                f"Expected sample shape {self.sample_shape_}, got {x.shape[1:]}."
            )
        if len(x) == 0:
            return np.empty(0, dtype=np.float64)
        flat = (x.reshape(len(x), -1) - self.mean_) / self.scale_
        return self._validated_scores(self._score(flat), len(x))

    @abstractmethod
    def _fit(self, x: np.ndarray) -> None: ...

    @abstractmethod
    def _score(self, x: np.ndarray) -> np.ndarray: ...
