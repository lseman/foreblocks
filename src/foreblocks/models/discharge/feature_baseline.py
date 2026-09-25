"""Hand-engineered spectral/wavelet/pulse feature classifier.

The "feature classifier using spectral/envelope features" matched baseline
from `datasets/_paper_review/model_proposal.md`'s evaluation item 5. Feature
families follow the same design as the exploratory audit in
`datasets/pd_acoustic_anomaly_detection.ipynb` (`acoustic_features`), adapted
from that notebook's 25 ms windows to this classifier's 100 ms context. The
extractor is the generic `foreblocks.features.SignalFeatures`, configured
with the acoustic partial-discharge bands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from foreblocks.features.signal import SignalFeatures
from foreblocks.models.discharge.novelty import ClassConditionalNovelty
from foreblocks.models.discharge.validation import waveform_matrix

# Acoustic partial-discharge bands; the upper edge includes 128 kHz Nyquist
# at the 256 kHz sampling rate.
_BANDS_HZ = ((0, 20_000), (20_000, 50_000), (50_000, 80_000), (80_000, 128_001))
_BAND_NAMES = ("band_0_20k", "band_20_50k", "band_50_80k", "band_80_128k")


def _extractor(fs) -> SignalFeatures:
    return SignalFeatures(fs=fs, bands=_BANDS_HZ, band_names=_BAND_NAMES)


FEATURE_NAMES = list(_extractor(256_000).get_feature_names_out())


def extract_features(x: np.ndarray, fs: int) -> np.ndarray:
    """`x`: `[N, L]` raw (pre-normalization) waveform contexts. Returns
    `[N, len(FEATURE_NAMES)]`.
    """
    return _extractor(fs).transform(waveform_matrix(x, allow_empty=True))


@dataclass
class FeatureResult:
    predicted_class: np.ndarray
    class_probabilities: np.ndarray
    features: np.ndarray
    novelty_score: np.ndarray | None = None
    is_unfamiliar: np.ndarray | None = None
    novelty_pvalue: np.ndarray | None = None


class FeatureDischargeClassifier:
    """Hand-engineered features + `RandomForestClassifier`, the classical
    counterpart to the two-branch CNN and the Rocket baseline. Also fits a
    `ClassConditionalNovelty` layer on the (standardized) feature vectors --
    that module is format-agnostic, so the same rejection design from the
    CNN applies here without new code.
    """

    def __init__(
        self,
        fs: int = 256_000,
        n_estimators: int = 300,
        seed: int | None = 42,
        novelty_method: str = "mahalanobis",
        novelty_quantile: float = 0.99,
    ):
        self.fs = fs
        self.n_estimators = n_estimators
        self.seed = seed
        self.novelty_method = novelty_method
        self.novelty_quantile = novelty_quantile
        self.classifier_: RandomForestClassifier | None = None
        self.scaler_: StandardScaler | None = None
        self.novelty_: ClassConditionalNovelty | None = None
        self.classes_: list[str] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> FeatureDischargeClassifier:
        features = extract_features(x, self.fs)
        self.classifier_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.classifier_.fit(features, y)
        self.classes_ = list(self.classifier_.classes_)

        self.scaler_ = StandardScaler().fit(features)
        self.novelty_ = ClassConditionalNovelty(method=self.novelty_method)
        self.novelty_.fit(self.scaler_.transform(features), y)
        return self

    def calibrate_novelty(
        self, x_calibration: np.ndarray, quantile: float | None = None
    ) -> float:
        if self.novelty_ is None or self.scaler_ is None:
            raise RuntimeError("Call fit before calibrate_novelty")
        features = self.scaler_.transform(extract_features(x_calibration, self.fs))
        scores = self.novelty_.score(features)
        return self.novelty_.calibrate_threshold(
            scores, self.novelty_quantile if quantile is None else quantile
        )

    def predict(self, x: np.ndarray) -> FeatureResult:
        if self.classifier_ is None:
            raise RuntimeError("Call fit before predict")
        features = extract_features(x, self.fs)
        predicted = self.classifier_.predict(features)
        probabilities = self.classifier_.predict_proba(features)

        novelty_score, is_unfamiliar = None, None
        if self.novelty_ is not None:
            novelty_score = self.novelty_.score(self.scaler_.transform(features))
            if self.novelty_.threshold_ is not None:
                is_unfamiliar = novelty_score > self.novelty_.threshold_

        return FeatureResult(
            predicted_class=predicted,
            class_probabilities=probabilities,
            features=features,
            novelty_score=novelty_score,
            is_unfamiliar=is_unfamiliar,
            novelty_pvalue=(
                self.novelty_.p_values(self.scaler_.transform(features))
                if is_unfamiliar is not None
                else None
            ),
        )
