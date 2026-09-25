"""Recording separation, numerical calibration, and attentive model regressions."""

import numpy as np
import pytest
import torch
from sklearn.covariance import OAS

from foreblocks.models.discharge import (
    ClassConditionalNovelty,
    DischargeClassifier,
    DischargeClassifierConfig,
    TemporalAttentionPool,
    compute_envelope,
    extract_features,
    split_training_indices,
)
from foreblocks.models.discharge.calibration import fit_temperature
from foreblocks.models.discharge.classifier import _context_features
from foreblocks.models.discharge.feature_baseline import FEATURE_NAMES
from foreblocks.models.discharge.preprocessing import normalize_context


def config(**kwargs):
    return DischargeClassifierConfig(
        **{
            "fs": 8000,
            "context_ms": 20,
            "frame_ms": 5,
            "waveform_feature_dim": 4,
            "waveform_kernels": (3, 5),
            "waveform_blocks": 1,
            "envelope_hidden_dim": 4,
            "envelope_levels": 1,
            "embedding_dim": 8,
            "epochs": 2,
            "batch_size": 8,
            "device": "cpu",
            "seed": 3,
            **kwargs,
        }
    )


def data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(36, 160)).astype(np.float32)
    y = np.repeat(["a", "b", "c"], 12)
    groups = np.repeat(np.arange(9), 4)
    return x, y, groups


def test_group_split_is_disjoint_reproducible_and_preserves_training_classes():
    x, y, groups = data()
    train, validation = split_training_indices(y, 0.3, groups=groups, seed=5)
    assert not set(groups[train]) & set(groups[validation])
    assert set(y[train]) == set(y)
    assert set(y[validation]) == set(y)
    np.testing.assert_array_equal(
        np.sort(np.concatenate([train, validation])), np.arange(len(y))
    )
    repeated = split_training_indices(y, 0.3, groups=groups, seed=5)
    np.testing.assert_array_equal(train, repeated[0])
    # Mixed-class groups also remain intact.
    mixed = np.tile(np.arange(4), 9)
    train, validation = split_training_indices(y, 0.25, groups=mixed)
    assert not set(mixed[train]) & set(mixed[validation])
    with pytest.raises(ValueError, match="recording groups"):
        split_training_indices(["a", "a", "b", "b"], groups=[0, 0, 1, 1])


def test_stratification_zero_split_and_invalid_splits():
    y = np.array(["rare", "common", "common", "common"])
    train, validation = split_training_indices(y, 0.5)
    assert 0 in train
    assert len(validation) == 1
    train, validation = split_training_indices(y, 0)
    assert not len(validation)
    np.testing.assert_array_equal(train, np.arange(4))
    for fraction in [-1, 1, float("nan")]:
        with pytest.raises(ValueError):
            split_training_indices(y, fraction)
    with pytest.raises(ValueError):
        split_training_indices(y, groups=[1, 2])


@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_training_reference_excludes_validation_and_inference_is_batch_invariant(
    pooling,
):
    x, y, groups = data()
    model = DischargeClassifier(config(frame_pooling=pooling)).fit(
        x, y, 0.3, groups=groups
    )
    assert model.novelty_.n_samples_fit_ == len(model.train_indices_)
    assert not set(groups[model.train_indices_]) & set(
        groups[model.validation_indices_]
    )
    expected = (
        _context_features(x[model.train_indices_]).astype(np.float64).mean(axis=0)
    )
    np.testing.assert_allclose(model.context_mean_, expected, rtol=1e-6)
    embeddings = model.embed(x[model.train_indices_])
    for label in model.classes_:
        np.testing.assert_allclose(
            model.novelty_.class_means_[label],
            embeddings[y[model.train_indices_] == label].mean(axis=0),
            rtol=1e-5,
            atol=1e-6,
        )
    result = model.predict(x[:5])
    model.config.batch_size = 1
    alone = model.predict(x[:5])
    np.testing.assert_allclose(result.embedding, alone.embedding, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        result.class_probabilities, alone.class_probabilities, rtol=1e-5, atol=1e-6
    )
    assert len(model.history_) == 2
    assert model.best_epoch_ is not None
    assert model.predict(np.empty((0, 160))).class_probabilities.shape == (0, 3)


def test_attention_uses_order_and_has_finite_gradients():
    torch.manual_seed(2)
    pool = TemporalAttentionPool(4)
    x = torch.randn(3, 6, 4, requires_grad=True)
    result = pool(x)
    assert result.shape == (3, 8)
    assert not torch.allclose(result, pool(x[:, [1, 5, 0, 4, 2, 3]]))
    result.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in pool.parameters()
    )
    assert torch.isfinite(pool(torch.ones(2, 1, 4))).all()


def nll(logits, y, temperature):
    scaled = logits / temperature
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    return np.mean(np.log(np.exp(shifted).sum(axis=1)) - shifted[np.arange(len(y)), y])


def test_temperature_improves_overconfident_logits_without_changing_argmax():
    logits = np.array([[8.0, -8.0], [8.0, -8.0], [-8.0, 8.0], [-8.0, 8.0]])
    y = np.array([0, 0, 1, 0])
    temperature = fit_temperature(logits, y)
    assert temperature > 1
    assert nll(logits, y, temperature) < nll(logits, y, 1)
    np.testing.assert_array_equal(
        logits.argmax(axis=1), (logits / temperature).argmax(axis=1)
    )
    assert fit_temperature(np.zeros((4, 2)), y) == 1
    with pytest.raises(ValueError):
        fit_temperature(logits, np.array([0, 1, 0, 2]))


def test_calibration_and_refit_reset():
    x, y, _ = data()
    torch.manual_seed(91)
    state = torch.random.get_rng_state().clone()
    model = DischargeClassifier(config()).fit(x, y, validation_split=0)
    assert torch.equal(state, torch.random.get_rng_state())
    assert len(model.validation_indices_) == 0
    assert all(item["validation_loss"] is None for item in model.history_)
    logits = model._infer(x)[1]
    temp = model.calibrate_probabilities(x, y)
    target = np.array([model.classes_.index(c) for c in y])
    assert nll(logits, target, temp) <= nll(logits, target, 1) + 1e-6
    model.calibrate_novelty(x, 0.9)
    result = model.predict(x)
    assert result.novelty_pvalue.shape == (len(x),)
    assert ((result.novelty_pvalue > 0) & (result.novelty_pvalue <= 1)).all()
    model.fit(x, y, validation_split=0)
    assert model.temperature_ == 1
    assert model.novelty_.threshold_ is None
    assert model.predict(x[:1]).novelty_pvalue is None
    with pytest.raises(ValueError, match="fitted classes"):
        model.calibrate_probabilities(x[:1], ["new"])


@pytest.mark.parametrize("method", ["mahalanobis", "ecod", "copod", "cosine"])
def test_novelty_methods_validation_and_reference_reset(method):
    train = np.array([[1.0, 0], [1.1, 0.1], [0, 1], [0.1, 1.1]])
    labels = np.array(["a", "a", "b", "b"])
    model = ClassConditionalNovelty(method=method).fit(train, labels)
    query = np.array([[1.0, 0], [-10.0, -10.0]])
    assert model.score(query)[1] > model.score(query)[0]
    assert model.score_per_class(query).shape == (2, 2)
    np.testing.assert_allclose(
        model.score(query), model.score_per_class(query).min(axis=1)
    )
    model.calibrate_threshold(model.score(train), 0.6)
    assert model.p_values(query)[1] <= model.p_values(query)[0]
    model.fit(train, labels)
    assert model.threshold_ is None and model.calibration_scores_ is None
    with pytest.raises(ValueError):
        model.score([[1, 2, 3]])
    with pytest.raises(ValueError):
        model.fit([[float("nan"), 1], [2, 3]], ["a", "b"])


def test_native_covariance_matches_oas_and_handles_zero_variance():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 8))
    y = np.repeat(["a", "b", "c"], 10)
    model = ClassConditionalNovelty().fit(x, y)
    residual = np.concatenate(
        [x[y == c] - x[y == c].mean(axis=0) for c in model.classes_]
    )
    covariance = OAS(assume_centered=True).fit(residual).covariance_
    mu = np.trace(covariance) / covariance.shape[0]
    expected = np.linalg.inv(covariance + 1e-6 * max(mu, 1) * np.eye(8))
    np.testing.assert_allclose(model.precision_, expected, rtol=1e-10, atol=1e-10)
    model.fit(np.zeros((4, 3)), ["a", "a", "b", "b"])
    scores = model.score([[0, 0, 0], [1, 1, 1]])
    assert scores[1] > scores[0] == 0


def test_conformal_rank_threshold_ties_and_small_sample_behavior():
    model = ClassConditionalNovelty().fit(
        [[0], [0.1], [1], [1.1]], ["a", "a", "b", "b"]
    )
    query = np.array([[0], [0.5], [10.0]])
    scores = model.score(query)
    calibration = np.array([scores[0], scores[0], scores[1], scores[2]])
    model.calibrate_threshold(calibration, 0.6)
    expected = np.array([(1 + np.sum(calibration >= score)) / 5 for score in scores])
    np.testing.assert_array_equal(model.p_values(query), expected)
    assert model.threshold_ == np.sort(calibration)[2]
    assert np.isinf(model.calibrate_threshold(calibration, 0.99))
    for invalid in [[], [np.nan], [[1, 2]]]:
        with pytest.raises(ValueError):
            model.calibrate_threshold(invalid, 0.9)
    for q in [0, -0.1, 1.1, np.nan]:
        with pytest.raises(ValueError):
            model.calibrate_threshold([1, 2], q)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"epochs": 0},
        {"fs": 0},
        {"frame_ms": 0},
        {"envelope_kernel_size": 4},
        {"waveform_kernels": (2,)},
        {"frame_pooling": "bad"},
        {"novelty_method": "bad"},
        {"dropout": 1},
        {"label_smoothing": -1},
        {"novelty_quantile": 0},
    ],
)
def test_config_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        config(**kwargs)


def test_inputs_gain_ablation_and_preprocessing_range():
    x, y, _ = data()
    model = DischargeClassifier(config(use_context_features=False)).fit(x, y, 0)
    np.testing.assert_allclose(
        model.embed(x[:3]), model.embed(10 * x[:3]), rtol=1e-5, atol=1e-5
    )
    assert np.isfinite(
        normalize_context(np.array([[1e30, -1e30]], dtype=np.float32))
    ).all()
    for bad in [x[0], x[:, None], x[:0], np.full_like(x, np.nan)]:
        with pytest.raises(ValueError):
            DischargeClassifier(config()).fit(bad, y)
    with pytest.raises(RuntimeError):
        DischargeClassifier(config()).embed(x)
    pulse = np.zeros((1, 800), dtype=np.float32)
    pulse[0, [100, 300, 500]] = 0.1
    features = extract_features(pulse, 8000)
    assert features[0, FEATURE_NAMES.index("pulse_rate_hz")] == 30
    for name in ["pulse_rate_hz", "pulse_occupancy", "pulse_gap_cv"]:
        index = FEATURE_NAMES.index(name)
        np.testing.assert_allclose(
            features[:, index], extract_features(pulse * 100, 8000)[:, index]
        )
    assert np.all(compute_envelope(pulse, 8000, 1000) >= 0)


def test_rational_envelope_rate_and_seeded_refit():
    cfg = DischargeClassifierConfig(fs=44_100, envelope_hz=2000)
    assert cfg.n_frames == 10
    x, y, _ = data()
    model = DischargeClassifier(config()).fit(x, y, 0)
    before = model.predict(x[:3]).class_probabilities
    model.fit(x, y, 0)
    np.testing.assert_array_equal(before, model.predict(x[:3]).class_probabilities)
