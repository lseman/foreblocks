"""Native class-conditional novelty scores and split-calibration tail ranks.

References are fitted only on training embeddings. A separate known-class
calibration set supplies a finite-sample quantile and conservative tail p-values.
These values are not probabilities of physical failure or of an unseen class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from foreblocks.models.anomaly.scorers.empirical import COPOD, ECOD
from foreblocks.models.discharge.validation import label_vector


@dataclass
class ClassConditionalNovelty:
    method: Literal["mahalanobis", "ecod", "copod", "cosine"] = "mahalanobis"
    covariance_shrinkage: float | None = None
    covariance_regularization: float = 1e-6
    class_means_: dict = field(default_factory=dict, init=False, repr=False)
    precision_: np.ndarray | None = field(default=None, init=False, repr=False)
    class_scorers_: dict = field(default_factory=dict, init=False, repr=False)
    threshold_: float | None = field(default=None, init=False)
    calibration_scores_: np.ndarray | None = field(default=None, init=False, repr=False)

    @staticmethod
    def _matrix(embeddings):
        x = np.asarray(embeddings, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] == 0 or not np.isfinite(x).all():
            raise ValueError(
                "Expected finite [samples, embedding_features] embeddings."
            )
        return x

    def fit(
        self, embeddings: np.ndarray, labels: np.ndarray
    ) -> ClassConditionalNovelty:
        x = self._matrix(embeddings)
        labels = label_vector(labels, len(x))
        classes = np.unique(labels)
        if len(classes) < 2:
            raise ValueError("Need at least 2 classes to fit a class-conditional score")
        if self.method not in {"mahalanobis", "ecod", "copod", "cosine"}:
            raise ValueError(f"Unknown novelty method: {self.method!r}")
        shrinkage = self.covariance_shrinkage
        if shrinkage is not None and (
            not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1
        ):
            raise ValueError("covariance_shrinkage must be None or in [0, 1].")
        if (
            not np.isfinite(self.covariance_regularization)
            or self.covariance_regularization <= 0
        ):
            raise ValueError("covariance_regularization must be positive and finite.")
        means = {c: x[labels == c].mean(axis=0) for c in classes}
        precision, scorers = None, {}
        if self.method == "mahalanobis":
            residuals = np.concatenate([x[labels == c] - means[c] for c in classes])
            covariance = residuals.T @ residuals / len(residuals)
            dimension = x.shape[1]
            mu = float(np.trace(covariance) / dimension)
            if shrinkage is None:
                # OAS shrinkage, computed directly on the pooled residuals.
                alpha = float(np.mean(covariance**2))
                denominator = (len(residuals) + 1) * (alpha - mu**2 / dimension)
                shrinkage = (
                    min(1.0, (alpha + mu**2) / denominator) if denominator > 0 else 1.0
                )
            covariance = (1 - shrinkage) * covariance + shrinkage * mu * np.eye(
                dimension
            )
            covariance += (
                self.covariance_regularization * max(mu, 1.0) * np.eye(dimension)
            )
            precision = np.linalg.solve(covariance, np.eye(dimension))
            if not np.isfinite(precision).all():
                raise ValueError("Covariance exceeds supported numerical range.")
        elif self.method in {"ecod", "copod"}:
            cls = ECOD if self.method == "ecod" else COPOD
            scorers = {c: cls().fit(x[labels == c]) for c in classes}
        self.class_means_, self.precision_, self.class_scorers_ = (
            means,
            precision,
            scorers,
        )
        self.classes_ = classes
        self.n_features_in_ = x.shape[1]
        self.n_samples_fit_ = len(x)
        self.fitted_method_ = self.method
        self.threshold_ = None
        self.calibration_scores_ = None
        return self

    def score_per_class(self, embeddings: np.ndarray) -> np.ndarray:
        if not hasattr(self, "fitted_method_") or self.method != self.fitted_method_:
            raise RuntimeError("ClassConditionalNovelty is not fitted for this method")
        x = self._matrix(embeddings)
        if x.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} embedding features.")
        distances = []
        for c in self.classes_:
            mean = self.class_means_[c]
            if self.method == "mahalanobis":
                residual = x - mean
                score = np.maximum(
                    np.sum((residual @ self.precision_) * residual, axis=1), 0
                )
            elif self.method == "cosine":
                length, center_length = np.linalg.norm(x, axis=1), np.linalg.norm(mean)
                denominator = length * center_length
                similarity = np.divide(
                    x @ mean,
                    denominator,
                    out=np.zeros(len(x)),
                    where=denominator > 1e-12,
                )
                similarity[(length <= 1e-12) & (center_length <= 1e-12)] = 1.0
                score = 1 - np.clip(similarity, -1, 1)
            else:
                score = self.class_scorers_[c].decision_function(x)
            distances.append(score)
        result = np.column_stack(distances)
        if not np.isfinite(result).all():
            raise ValueError("Novelty scoring produced nonfinite distances.")
        return result

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        return self.score_per_class(embeddings).min(axis=1)

    def calibrate_threshold(
        self, calibration_scores: np.ndarray, quantile: float
    ) -> float:
        scores = np.asarray(calibration_scores, dtype=np.float64)
        if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
            raise ValueError("Calibration scores must be a nonempty finite vector.")
        if not np.isfinite(quantile) or not 0 < quantile <= 1:
            raise ValueError("quantile must be in (0, 1].")
        ordered = np.sort(scores)
        k = int(np.ceil((len(ordered) + 1) * quantile))
        self.threshold_ = float(ordered[k - 1]) if k <= len(ordered) else float(np.inf)
        self.calibration_scores_ = ordered
        return self.threshold_

    def p_values(self, embeddings: np.ndarray) -> np.ndarray:
        """Conservative upper-tail ranks, counting calibration ties as >= query."""
        if self.calibration_scores_ is None:
            raise RuntimeError("Call calibrate_threshold before p_values")
        scores = self.score(embeddings)
        n = len(self.calibration_scores_)
        below = np.searchsorted(self.calibration_scores_, scores, side="left")
        return (n - below + 1) / (n + 1)

    def is_unfamiliar(self, embeddings: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise RuntimeError("Call calibrate_threshold before is_unfamiliar")
        return self.score(embeddings) > self.threshold_
