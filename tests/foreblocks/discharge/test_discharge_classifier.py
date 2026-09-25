"""Structural and behavioral guards for the discharge classifier."""

import numpy as np
import pytest
import torch

from foreblocks.data.windowing import build_grouped_frames
from foreblocks.models import DischargeClassifier, DischargeClassifierConfig
from foreblocks.models.discharge import ClassConditionalNovelty
from foreblocks.models.discharge.backbones import (
    EnvelopeEncoder,
    WaveformEncoder,
    compute_envelope,
)

FS = 8_000  # small synthetic sampling rate keeps unit tests fast


def _config(**overrides) -> DischargeClassifierConfig:
    defaults = dict(
        fs=FS,
        context_ms=100.0,
        frame_ms=10.0,
        waveform_feature_dim=4,
        waveform_blocks=1,
        envelope_hidden_dim=4,
        envelope_levels=2,
        embedding_dim=8,
        epochs=2,
        batch_size=16,
        seed=0,
    )
    defaults.update(overrides)
    return DischargeClassifierConfig(**defaults)


def _synthetic_dataset(n_per_class: int = 24, context_size: int = 800, seed: int = 0):
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


def test_waveform_encoder_shape():
    encoder = WaveformEncoder(feature_dim=4, kernels=(3, 5), n_blocks=1)
    frames = torch.randn(6, 80)
    out = encoder(frames)
    assert out.shape == (6, encoder.output_dim)


def test_envelope_encoder_shape():
    encoder = EnvelopeEncoder(hidden_dim=4, n_levels=2)
    envelope = torch.randn(6, 50)
    out = encoder(envelope)
    assert out.shape == (6, encoder.output_dim)


def test_compute_envelope_shape_and_nonnegative():
    x = np.random.default_rng(0).normal(size=(5, 800)).astype(np.float32)
    envelope = compute_envelope(x, fs=FS, target_hz=1000)
    assert envelope.shape == (5, 100)
    assert np.all(envelope >= 0)


@pytest.mark.parametrize("frame_size,stride", [(10, None), (10, 5)])
def test_build_grouped_frames_shapes_and_groups(frame_size, stride):
    series = {"a": np.arange(35, dtype=np.float32), "b": np.arange(100, 122, dtype=np.float32)}
    frames, groups = build_grouped_frames(series, frame_size, stride)
    effective_stride = frame_size if stride is None else stride
    expected_a = (35 - frame_size) // effective_stride + 1
    expected_b = (22 - frame_size) // effective_stride + 1
    assert frames.shape == (expected_a + expected_b, frame_size, 1)
    assert (groups == "a").sum() == expected_a
    assert (groups == "b").sum() == expected_b
    np.testing.assert_array_equal(frames[0, :, 0], series["a"][:frame_size])
    with pytest.raises(ValueError):
        build_grouped_frames({"short": np.zeros(3)}, frame_size=10)


def test_discharge_classifier_fit_predict_round_trip():
    x, y = _synthetic_dataset()
    model = DischargeClassifier(_config())
    model.fit(x, y)
    result = model.predict(x)

    n_classes = len(set(y.tolist()))
    assert result.predicted_class.shape == (len(x),)
    assert result.class_probabilities.shape == (len(x), n_classes)
    np.testing.assert_allclose(
        result.class_probabilities.sum(axis=1), 1.0, atol=1e-5
    )
    assert result.embedding.shape == (len(x), model.config.embedding_dim)
    assert result.novelty_score is not None
    assert result.novelty_score.shape == (len(x),)
    # Threshold not calibrated yet: no hard flag, only a raw score.
    assert result.is_unfamiliar is None

    model.calibrate_novelty(x, quantile=0.9)
    result = model.predict(x)
    assert result.is_unfamiliar is not None
    assert result.is_unfamiliar.dtype == bool


def test_discharge_classifier_rejects_wrong_context_length():
    x, y = _synthetic_dataset(n_per_class=4, context_size=800)
    model = DischargeClassifier(_config())
    model.fit(x, y)
    with pytest.raises(ValueError):
        model.predict(x[:, :700])


@pytest.mark.parametrize("method", ["mahalanobis", "ecod"])
def test_class_conditional_novelty_separates_far_point(method):
    rng = np.random.default_rng(1)
    cluster_a = rng.normal(loc=0.0, scale=0.2, size=(40, 6))
    cluster_b = rng.normal(loc=5.0, scale=0.2, size=(40, 6))
    embeddings = np.concatenate([cluster_a, cluster_b], axis=0)
    labels = np.array(["a"] * 40 + ["b"] * 40)

    novelty = ClassConditionalNovelty(method=method).fit(embeddings, labels)
    inlier_score = novelty.score(np.array([[0.0] * 6]))[0]
    outlier_score = novelty.score(np.array([[50.0] * 6]))[0]
    assert outlier_score > inlier_score

    threshold = novelty.calibrate_threshold(novelty.score(embeddings), quantile=0.95)
    assert np.isfinite(threshold)
    assert novelty.is_unfamiliar(np.array([[50.0] * 6]))[0]
