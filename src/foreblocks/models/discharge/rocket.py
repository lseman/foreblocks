"""Rocket-family baseline for the discharge classifier.

The matched baseline from `datasets/_paper_review/model_proposal.md`'s
evaluation item 5, evaluated against the two-branch CNN in `classifier.py`.
The transforms themselves live in `foreblocks.features.rocket`; this module
adds the discharge input contract and class-conditional novelty flagging.
The default `"fused"` transform (`FusedRocket`, kept here under its original
name `RocketFeatures`) uses ROCKET kernels with MultiRocket's
PPV/MPV-on-signal-and-difference pooling; `"minirocket"`, `"multirocket"`
and `"rocket"` select the published methods.

References:
    ROCKET: https://arxiv.org/abs/1910.13051
    MiniRocket: https://arxiv.org/abs/2012.08791
    MultiRocket: https://arxiv.org/abs/2102.00457
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from foreblocks.features.rocket import FusedRocket, RocketClassifier, softmax_scores
from foreblocks.models.discharge.novelty import ClassConditionalNovelty
from foreblocks.models.discharge.preprocessing import normalize_context

RocketFeatures = FusedRocket


@dataclass
class RocketResult:
    predicted_class: np.ndarray
    class_probabilities: np.ndarray
    novelty_score: np.ndarray | None = None
    is_unfamiliar: np.ndarray | None = None
    novelty_pvalue: np.ndarray | None = None


class RocketDischargeClassifier:
    """Rocket features + `RidgeClassifierCV`, the standard ROCKET-family
    pairing, plus a `ClassConditionalNovelty` layer fit on the (scaled)
    Rocket features. Uses ECOD rather than the Mahalanobis default: with
    `num_kernels=3000` the feature space is 12,000-dimensional, too high for
    a shared covariance matrix to be well-conditioned against ~3,000
    training windows; ECOD scores each dimension's marginal tail instead of
    inverting a covariance matrix, so it degrades gracefully at this scale.

    `transformer` selects the Rocket-family transform (see
    `foreblocks.features.make_rocket_transform`). `num_kernels` sizes the
    `"fused"` and `"rocket"` kernel banks; `"minirocket"` and
    `"multirocket"` take their size from `transformer_params`.
    """

    def __init__(
        self,
        num_kernels: int = 2_000,
        seed: int | None = 42,
        device: str | None = None,
        novelty_method: str = "ecod",
        novelty_quantile: float = 0.99,
        transformer: str = "fused",
        transformer_params: dict | None = None,
    ):
        self.num_kernels = num_kernels
        self.seed = seed
        self.device = device
        self.novelty_method = novelty_method
        self.novelty_quantile = novelty_quantile
        self.transformer = transformer
        self.transformer_params = transformer_params
        self.model_: RocketClassifier | None = None
        self.novelty_: ClassConditionalNovelty | None = None
        self.classes_: list[str] | None = None

    def _transformer_params(self) -> dict:
        params = {"seed": self.seed}
        if self.transformer in {"fused", "rocket"}:
            params["num_kernels"] = self.num_kernels
        if self.transformer == "fused":
            params["device"] = self.device
        return {**params, **(self.transformer_params or {})}

    def fit(self, x: np.ndarray, y: np.ndarray) -> RocketDischargeClassifier:
        x = normalize_context(x)
        self.model_ = RocketClassifier(
            transformer=self.transformer,
            transformer_params=self._transformer_params(),
            normalize=False,
        ).fit(x, y)
        self.classes_ = list(self.model_.classes_)

        self.novelty_ = ClassConditionalNovelty(method=self.novelty_method)
        self.novelty_.fit(self.model_.features(x), y)
        return self

    @property
    def rocket_(self):
        return None if self.model_ is None else self.model_.transformer_

    @property
    def scaler_(self):
        return None if self.model_ is None else self.model_.scaler_

    @property
    def classifier_(self):
        return None if self.model_ is None else self.model_.classifier_

    def calibrate_novelty(
        self, x_calibration: np.ndarray, quantile: float | None = None
    ) -> float:
        if self.novelty_ is None:
            raise RuntimeError("Call fit before calibrate_novelty")
        scores = self.novelty_.score(self.model_.features(normalize_context(x_calibration)))
        return self.novelty_.calibrate_threshold(
            scores, self.novelty_quantile if quantile is None else quantile
        )

    def predict(self, x: np.ndarray) -> RocketResult:
        if self.model_ is None:
            raise RuntimeError("Call fit before predict")
        scaled = self.model_.features(normalize_context(x))
        predicted = self.model_.classifier_.predict(scaled)
        probabilities = softmax_scores(self.model_.classifier_.decision_function(scaled))

        novelty_score, is_unfamiliar = None, None
        if self.novelty_ is not None:
            novelty_score = self.novelty_.score(scaled)
            if self.novelty_.threshold_ is not None:
                is_unfamiliar = novelty_score > self.novelty_.threshold_

        return RocketResult(
            predicted_class=predicted,
            class_probabilities=probabilities,
            novelty_score=novelty_score,
            is_unfamiliar=is_unfamiliar,
            novelty_pvalue=(
                self.novelty_.p_values(scaled) if is_unfamiliar is not None else None
            ),
        )
