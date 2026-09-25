"""Structural guards for the Rocket-family and engineered-feature baselines."""

import numpy as np
import pytest
import torch

from foreblocks.models.discharge import (
    FeatureDischargeClassifier,
    RocketDischargeClassifier,
    RocketFeatures,
    extract_features,
)
from foreblocks.models.discharge.feature_baseline import FEATURE_NAMES

FS = 8_000


def _synthetic_dataset(n_per_class: int = 12, context_size: int = 800, seed: int = 0):
    rng = np.random.default_rng(seed)
    t = np.arange(context_size) / FS
    classes = {
        "corona": 0.3 * np.sin(2 * np.pi * 1500 * t),
        "dba": 1.0 * np.sin(2 * np.pi * 300 * t),
        "surface_discharge": 0.6 * np.sign(np.sin(2 * np.pi * 700 * t)),
    }
    x, y = [], []
    for label, base in classes.items():
        for _ in range(n_per_class):
            x.append(base + rng.normal(scale=0.05, size=context_size))
            y.append(label)
    order = rng.permutation(len(x))
    return np.array(x, dtype=np.float32)[order], np.array(y)[order]


def test_rocket_features_shape_and_determinism():
    x, _ = _synthetic_dataset(n_per_class=4)
    rocket = RocketFeatures(num_kernels=32, seed=0).fit(x.shape[-1])
    features_a = rocket.transform(x)
    features_b = rocket.transform(x)
    assert features_a.shape == (len(x), 32 * 4)
    np.testing.assert_allclose(features_a, features_b)  # deterministic given fixed kernels
    assert np.isfinite(features_a).all()


@pytest.mark.parametrize("backend, device", [("numba", "cpu"), ("triton", "cuda")])
def test_fused_rocket_backends_match_conv1d_reference(backend, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x, _ = _synthetic_dataset(n_per_class=4)
    x[0] = 0.0  # all-constant series: every PPV/MPV hits its bias-only value
    reference = RocketFeatures(num_kernels=64, seed=0, device="cpu", backend="torch")
    fused = RocketFeatures(num_kernels=64, seed=0, device=device, backend=backend)
    np.testing.assert_allclose(
        fused.fit(x.shape[-1]).transform(x),
        reference.fit(x.shape[-1]).transform(x),
        atol=1e-4,
    )


def test_rocket_classifier_fit_predict_round_trip():
    x, y = _synthetic_dataset()
    model = RocketDischargeClassifier(num_kernels=32, seed=0)
    model.fit(x, y)
    result = model.predict(x)
    assert result.predicted_class.shape == (len(x),)
    assert result.class_probabilities.shape == (len(x), 3)
    np.testing.assert_allclose(result.class_probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert set(model.classes_) == set(y.tolist())
    assert result.novelty_score is not None
    assert result.is_unfamiliar is None  # threshold not calibrated yet

    model.calibrate_novelty(x, quantile=0.9)
    result = model.predict(x)
    assert result.is_unfamiliar is not None
    assert result.is_unfamiliar.dtype == bool


def test_extract_features_shape_and_names_aligned():
    x, _ = _synthetic_dataset(n_per_class=3)
    features = extract_features(x, fs=FS)
    assert features.shape == (len(x), len(FEATURE_NAMES))
    assert np.isfinite(features).all()


def test_feature_classifier_fit_predict_round_trip():
    x, y = _synthetic_dataset()
    model = FeatureDischargeClassifier(fs=FS, n_estimators=20, seed=0)
    model.fit(x, y)
    result = model.predict(x)
    assert result.predicted_class.shape == (len(x),)
    assert result.class_probabilities.shape == (len(x), 3)
    np.testing.assert_allclose(result.class_probabilities.sum(axis=1), 1.0, atol=1e-5)
    assert result.features.shape == (len(x), len(FEATURE_NAMES))
    assert result.novelty_score is not None
    assert result.is_unfamiliar is None  # threshold not calibrated yet

    model.calibrate_novelty(x, quantile=0.9)
    result = model.predict(x)
    assert result.is_unfamiliar is not None
    assert result.is_unfamiliar.dtype == bool


@pytest.mark.parametrize("model_cls", [RocketDischargeClassifier, FeatureDischargeClassifier])
def test_baselines_separate_easy_synthetic_classes(model_cls):
    x, y = _synthetic_dataset(n_per_class=20)
    kwargs = dict(seed=0)
    if model_cls is RocketDischargeClassifier:
        kwargs["num_kernels"] = 64
    else:
        kwargs["fs"] = FS
        kwargs["n_estimators"] = 50
    model = model_cls(**kwargs)
    model.fit(x, y)
    result = model.predict(x)
    accuracy = (result.predicted_class == y).mean()
    assert accuracy > 0.9  # these synthetic classes are trivially separable
