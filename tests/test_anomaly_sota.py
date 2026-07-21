"""Tests for SOTA anomaly detection modules: PatchMamba, iTransformer, calibration, online."""

import numpy as np
import pytest
import torch

from foreblocks.anomaly import (
    AnomalyDetectorConfig,
    BNAdaptiveWrapper,
    EMAStatistics,
    EnsembleScoreCombiner,
    ForeblocksAnomalyDetector,
    PatchMamba,
    PlattScaler,
    StreamingAnomalyDetector,
    TemperatureScaler,
    TENTAdapter,
    compute_confidence,
    fit_score_distribution,
    isotonic_calibrate,
    iTransformer,
)
from foreblocks.anomaly.models.state_space import PatchSSMBlock, S6Block

# ── PatchMamba tests ──


def test_patch_mamba_forward_preserves_shape():
    model = PatchMamba(
        n_features=2,
        window_size=8,
        patch_size=4,
        d_model=32,
        n_layers=2,
        d_state=8,
        dropout=0.0,
    )
    x = torch.randn(4, 8, 2)
    out = model(x)
    assert tuple(out.reconstruction.shape) == (4, 8, 2)
    assert out.per_token_scores.shape == (4,)


def test_patch_mamba_detector_smoke():
    rng = np.random.default_rng(42)
    series = rng.normal(size=(32, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="patch_mamba",
        window_size=8,
        epochs=2,
        batch_size=8,
        d_model=32,
        n_layers=2,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )
    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)
    assert result.scores.shape == (32,)
    assert result.labels.shape == (32,)


def test_s6_block_output_shape():
    block = S6Block(d_model=32, d_state=8, dropout=0.0)
    x = torch.randn(2, 16, 32)
    out = block(x)
    assert out.shape == (2, 16, 32)


# ── iTransformer tests ──


def test_i_transformer_forward_preserves_shape():
    model = iTransformer(
        n_features=3,
        window_size=8,
        d_model=32,
        n_layers=2,
        dropout=0.0,
    )
    x = torch.randn(4, 8, 3)
    out = model(x)
    assert tuple(out.reconstruction.shape) == (4, 8, 3)
    assert out.feature_scores.shape == (3,)  # per-feature scores


def test_i_transformer_detector_smoke():
    rng = np.random.default_rng(43)
    series = rng.normal(size=(32, 3)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="i_transformer",
        window_size=8,
        epochs=2,
        batch_size=8,
        d_model=32,
        n_layers=2,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )
    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)
    assert result.scores.shape == (32,)
    assert result.labels.shape == (32,)


# ── Calibration tests ──


def test_temperature_scaler_fit():
    scaler = TemperatureScaler(learnable=True)
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])
    temp = scaler.fit(scores, labels, lr=0.1, n_steps=50)
    assert 0.01 <= temp <= 10.0
    x = torch.tensor([1.0, 3.0])
    out = scaler(x)
    assert out.shape == (2,)


def test_platt_scaler_fit():
    scaler = PlattScaler()
    scores = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1])
    a, b = scaler.fit(scores, labels, lr=0.01, n_steps=100)
    x = torch.tensor([1.0, 3.0])
    probs = scaler(x)
    assert probs.shape == (2,)
    assert 0.0 <= probs.min() and probs.max() <= 1.0


def test_isotonic_calibrate_basic():
    raw = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    calibrated = isotonic_calibrate(raw, labels, increasing=True)
    assert len(calibrated) == 8
    assert all(0.0 <= c <= 1.0 for c in calibrated)
    # Monotonicity check
    for i in range(1, len(calibrated)):
        assert calibrated[i] >= calibrated[i - 1] - 1e-6


def test_compute_confidence():
    scores = np.array([0.1, 1.0, 3.0, 5.0, 7.0], dtype=np.float64)
    threshold = 3.0
    result = compute_confidence(scores, threshold)
    assert result.probabilities.shape == (5,)
    assert result.confidence.shape == (5,)
    assert result.uncertainty.shape == (5,)
    assert result.labels.shape == (5,)


def test_ensemble_combiner_equality():
    combiner = EnsembleScoreCombiner(n_detectors=3, strategy="equal")
    scores = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    # Without normalization: simple average
    combined = combiner.combine(scores, normalize=False)
    (n_samples,) = combined.shape
    assert n_samples == 2
    assert np.allclose(combined, np.array([2.0, 5.0]))


def test_ensemble_combiner_normalize():
    combiner = EnsembleScoreCombiner(n_detectors=2, strategy="equal")
    scores = np.array([[0.0, 100.0], [10.0, 110.0]], dtype=np.float64)
    normalized = combiner.normalize_scores(scores)
    # After z-score, both features should have mean ~0 and std ~1
    assert np.allclose(normalized[:, 0].mean(), 0.0, atol=1e-6)
    assert np.allclose(normalized[:, 1].mean(), 0.0, atol=1e-6)


def test_fit_score_distribution_mad():
    scores = np.random.default_rng(0).normal(0, 1, 1000).astype(np.float64)
    result = fit_score_distribution(scores, method="mad")
    assert "threshold" in result
    assert np.isfinite(result["threshold"])


def test_fit_score_distribution_gaussian():
    scores = np.random.default_rng(1).normal(5, 2, 500).astype(np.float64)
    result = fit_score_distribution(scores, method="gaussian")
    assert "mean" in result
    assert "std" in result
    assert result["threshold"] > result["mean"]


# ── Online/Streaming tests ──


def test_ema_statistics():
    ema = EMAStatistics(decay=0.9, max_len=100)
    for i in range(50):
        ema.update(float(i))  # type: ignore[call-overload]
    assert ema.mean > 0
    assert ema.std > 0
    thresh = ema.adaptive_threshold(z=2.0)
    assert thresh > ema.mean


def test_streaming_detector_smoke():
    # Test without underlying detector (uses default scoring)
    detector = StreamingAnomalyDetector(None, initial_threshold=1.0)
    for _ in range(20):
        sample = np.random.randn(5, 2).astype(np.float32)
        result = detector.score(sample)
        assert np.isfinite(result.score)
        val = result.is_anomaly
        assert val == True or val == False
        assert 0.0 <= result.confidence <= 1.0


# ── Block registry tests ──


def test_patch_mamba_block_registered():
    from foreblocks.anomaly.modes import list_blocks, resolve_block

    blocks = list_blocks()
    assert "patch_mamba" in blocks
    block = resolve_block("patch_mamba")
    assert block.block_type() == "patch_mamba"


def test_i_transformer_block_registered():
    from foreblocks.anomaly.modes import list_blocks, resolve_block

    blocks = list_blocks()
    assert "i_transformer" in blocks
    block = resolve_block("i_transformer")
    assert block.block_type() == "i_transformer"
