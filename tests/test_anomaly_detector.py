import numpy as np
import torch

from foreblocks.anomaly import (
    AnomalyDetectorConfig,
    AnomalyTransformer,
    DAGMM,
    ForeblocksAnomalyDetector,
    OmniAnomaly,
    TranAD,
    TranADDetector,
    TransformerVAE,
    build_sliding_windows,
    map_window_scores,
)
from foreblocks.anomaly.models.forecasting import TransformerForecaster
from foreblocks.anomaly.models.reconstruction import MLPVAE
from foreblocks.anomaly.models.representation import ContrastiveTransformerEncoder
from foreblocks.anomaly.models.tranad import TranAD as ModelTranAD
from foreblocks.ts_handler import outlier


def test_model_family_modules_are_importable():
    assert MLPVAE.__name__ == "MLPVAE"
    assert OmniAnomaly.__name__ == "OmniAnomaly"
    assert AnomalyTransformer.__name__ == "AnomalyTransformer"
    assert DAGMM.__name__ == "DAGMM"
    assert TransformerForecaster.__name__ == "TransformerForecaster"
    assert ContrastiveTransformerEncoder.__name__ == "ContrastiveTransformerEncoder"
    assert ModelTranAD.__name__ == "TranAD"


def test_transformer_vae_uses_foreblocks_layers_and_preserves_shape():
    model = TransformerVAE(
        n_features=2,
        window_size=6,
        d_model=32,
        latent_size=8,
        n_layers=1,
        dropout=0.0,
    )
    x = torch.randn(4, 6, 2)
    out = model(x)

    assert tuple(out.reconstruction.shape) == (4, 6, 2)
    assert tuple(out.mu.shape) == (4, 8)
    assert model.encoder.layers[0].__class__.__name__ == "TransformerEncoderLayer"


def test_outlier_tranad_imports_from_anomaly_package():
    assert outlier.TranAD is TranAD
    assert outlier.TranADDetector is TranADDetector


def test_window_score_mapping_end_alignment():
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    mapped = map_window_scores(scores, series_length=5, window_size=3)

    np.testing.assert_allclose(mapped[:2], [np.nan, np.nan])
    np.testing.assert_allclose(mapped[2:], [1.0, 2.0, 3.0])


def test_mlp_vae_detector_fit_predict_smoke():
    rng = np.random.default_rng(7)
    series = rng.normal(size=(32, 2)).astype(np.float32)
    series[20] += 5.0
    cfg = AnomalyDetectorConfig(
        model_type="mlp_vae",
        window_size=6,
        epochs=2,
        batch_size=8,
        hidden_size=32,
        latent_size=4,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (32,)
    assert result.labels.shape == (32,)
    assert result.window_scores.shape == (27, 2)
    assert np.isfinite(result.threshold)


def test_transformer_vae_detector_fit_predict_smoke():
    rng = np.random.default_rng(9)
    series = rng.normal(size=(20, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="transformer_vae",
        window_size=5,
        epochs=1,
        batch_size=5,
        d_model=32,
        latent_size=4,
        n_layers=1,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (20,)
    assert result.window_scores.shape == (16, 2)


def test_omni_anomaly_detector_fit_predict_smoke():
    rng = np.random.default_rng(21)
    series = rng.normal(size=(20, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="omni_anomaly",
        window_size=5,
        epochs=1,
        batch_size=5,
        hidden_size=16,
        latent_size=4,
        n_layers=1,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (20,)
    assert result.window_scores.shape == (16, 2)


def test_anomaly_transformer_detector_fit_predict_smoke():
    rng = np.random.default_rng(23)
    series = rng.normal(size=(20, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="anomaly_transformer",
        window_size=5,
        epochs=1,
        batch_size=5,
        d_model=16,
        n_layers=1,
        association_weight=0.01,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (20,)
    assert result.window_scores.shape == (16, 2)


def test_dagmm_detector_fit_predict_smoke():
    rng = np.random.default_rng(25)
    series = rng.normal(size=(20, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="dagmm",
        window_size=5,
        epochs=1,
        batch_size=5,
        hidden_size=16,
        latent_size=3,
        gmm_components=2,
        energy_weight=0.01,
        covariance_weight=0.001,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (20,)
    assert result.window_scores.shape == (16,)


def test_tranad_detector_scores_last_step():
    rng = np.random.default_rng(11)
    series = rng.normal(size=(18, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        model_type="tranad",
        window_size=4,
        epochs=1,
        batch_size=6,
        d_model=32,
        n_layers=1,
        contamination=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (18,)
    assert result.window_scores.shape == (15, 2)


def test_forecasting_mode_can_use_transformer_forecaster():
    rng = np.random.default_rng(13)
    series = rng.normal(size=(20, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        detection_mode="forecasting",
        model_type="transformer_vae",
        window_size=5,
        epochs=1,
        batch_size=5,
        d_model=32,
        n_layers=1,
        num_workers=0,
        use_mixed_precision=False,
    )
    detector = ForeblocksAnomalyDetector(cfg)

    result = detector.fit_predict(series, validation_split=0.0)

    assert detector.detection_mode == "forecasting"
    assert result.window_scores.shape == (16, 2)


def test_representation_mode_fit_predict_smoke():
    rng = np.random.default_rng(17)
    series = rng.normal(size=(18, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        detection_mode="representation",
        window_size=5,
        epochs=1,
        batch_size=6,
        d_model=32,
        projection_size=8,
        n_layers=1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (18,)
    assert result.window_scores.shape == (14, 1)


def test_hybrid_mode_fit_predict_smoke():
    rng = np.random.default_rng(19)
    series = rng.normal(size=(18, 2)).astype(np.float32)
    cfg = AnomalyDetectorConfig(
        detection_mode="hybrid",
        model_type="mlp_vae",
        window_size=5,
        epochs=1,
        batch_size=6,
        d_model=32,
        hidden_size=32,
        latent_size=4,
        projection_size=8,
        n_layers=1,
        representation_weight=0.1,
        num_workers=0,
        use_mixed_precision=False,
    )

    result = ForeblocksAnomalyDetector(cfg).fit_predict(series, validation_split=0.0)

    assert result.scores.shape == (18,)
    assert result.window_scores.shape == (14, 2)


def test_sliding_windows_support_univariate_series():
    series = np.arange(8, dtype=np.float32)
    windows = build_sliding_windows(series, 4)

    assert windows.shape == (5, 4, 1)
