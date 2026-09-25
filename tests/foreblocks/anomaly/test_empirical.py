import numpy as np
import pytest

from foreblocks.models.anomaly import (
    COPOD,
    ECOD,
    HBOS,
    ForeblocksAnomalyDetector,
    copod_score,
    ecod_score,
    hbos_score,
)


@pytest.mark.parametrize("cls", [ECOD, COPOD, HBOS])
def test_fitted_scores_are_batch_independent_and_detect_extremes(cls):
    rng = np.random.default_rng(31)
    train = rng.normal(size=(200, 3))
    queries = np.vstack(
        [np.zeros((1, 3)), np.full((1, 3), 20), rng.normal(size=(7, 3))]
    )
    model = cls().fit(train)
    scores = model.decision_function(queries)
    individual = np.concatenate([model.decision_function(row[None]) for row in queries])
    np.testing.assert_allclose(scores, individual)
    np.testing.assert_allclose(scores, model.feature_scores(queries).sum(axis=1))
    np.testing.assert_allclose(model.decision_scores_, model.decision_function(train))
    assert scores[1] > scores[0]
    assert np.isfinite(scores).all()


@pytest.mark.parametrize("cls", [ECOD, COPOD, HBOS])
def test_constants_ties_empty_queries_and_refit(cls):
    model = cls().fit(np.ones((1, 2)))
    np.testing.assert_allclose(model.decision_function(np.ones((3, 2))), 0)
    assert model.decision_function(np.zeros((1, 2)))[0] > 0
    assert model.decision_function(np.empty((0, 2))).shape == (0,)
    model.fit(np.zeros((4, 2)))
    np.testing.assert_allclose(model.decision_scores_, 0)
    assert model.decision_function(np.ones((1, 2)))[0] > 0


@pytest.mark.parametrize("cls", [ECOD, COPOD, HBOS])
def test_input_validation(cls):
    with pytest.raises(RuntimeError, match="not fitted"):
        cls().decision_function(np.ones((2, 2)))
    for bad in [np.ones(3), np.empty((0, 2)), np.empty((3, 0)), [[np.nan]], [[np.inf]]]:
        with pytest.raises(ValueError):
            cls().fit(bad)
    model = cls().fit(np.ones((3, 2, 2)))
    with pytest.raises(ValueError, match="sample shape"):
        model.decision_function(np.ones((3, 4)))


def test_ecod_and_copod_match_hand_computed_tied_empirical_tails():
    values = np.array([[0.0], [0.0], [1.0], [3.0]])  # positive skew
    left = -np.log([0.5, 0.5, 0.75, 1.0])
    right = -np.log([1.0, 1.0, 0.5, 0.25])
    np.testing.assert_allclose(
        ECOD().fit(values).decision_scores_, np.maximum(left, right)
    )
    np.testing.assert_allclose(
        COPOD().fit(values).decision_scores_, np.maximum((left + right) / 2, right)
    )
    # Symmetric marginals: ECOD includes both tails for zero skew.
    symmetric = np.array([[-1.0], [0.0], [1.0]])
    np.testing.assert_allclose(
        ECOD().fit(symmetric).decision_scores_, [np.log(3), 2 * np.log(1.5), np.log(3)]
    )
    mirrored = -values
    np.testing.assert_allclose(
        COPOD().fit(mirrored).decision_scores_, COPOD().fit(values).decision_scores_
    )


def test_hbos_bin_edges_density_and_parameters():
    model = HBOS(n_bins=2, alpha=0.1).fit(np.array([[0.0], [0.0], [1.0], [2.0]]))
    np.testing.assert_allclose(model.decision_function([[0], [1], [2]]), 0)
    np.testing.assert_allclose(model.decision_function([[-1], [3]]), np.log(11))
    for kwargs in [
        {"n_bins": 1},
        {"n_bins": 2.5},
        {"alpha": 0},
        {"alpha": float("nan")},
    ]:
        with pytest.raises(ValueError):
            HBOS(**kwargs)


@pytest.mark.parametrize(
    "cls,score", [(ECOD, ecod_score), (COPOD, copod_score), (HBOS, hbos_score)]
)
def test_convenience_functions_and_window_flattening(cls, score):
    windows = np.random.default_rng(9).normal(size=(30, 4, 2))
    query = windows[:3] + 3
    expected = cls().fit(windows.reshape(30, 8)).decision_function(query.reshape(3, 8))
    np.testing.assert_allclose(score(query, reference=windows), expected)
    np.testing.assert_allclose(score(windows), cls().fit(windows).decision_scores_)


@pytest.mark.parametrize("model_type", ["ecod", "copod", "hbos"])
@pytest.mark.parametrize("mode", ["auto", "classical"])
def test_detector_fit_predict_uses_training_reference(model_type, mode):
    train = np.random.default_rng(7).normal(size=(80, 2)).astype(np.float32)
    detector = ForeblocksAnomalyDetector(
        model_type=model_type,
        detection_mode=mode,
        window_size=4,
        batch_size=7,
        device="cpu",
    ).fit(train)
    queries = np.zeros((12, 2), dtype=np.float32)
    queries[-4:] = 30
    first = detector.predict(queries)
    detector.config.batch_size = 1
    second = detector.predict(queries)
    np.testing.assert_allclose(first.scores, second.scores)
    np.testing.assert_array_equal(first.labels, second.labels)
    assert first.scores.shape == (12,)
    assert first.window_scores.shape == (9,)
    assert np.isnan(first.scores[:3]).all()
    assert first.window_scores[-1] > first.window_scores[0]
    assert detector.detection_mode == "classical"


@pytest.mark.parametrize(
    "model_type,expected_mode",
    [
        ("patch_mamba", "patch_mamba"),
        ("i_transformer", "i_transformer"),
        ("isolation_forest", "classical"),
        ("lof", "classical"),
        ("pca_mahalanobis", "classical"),
        ("matrix_profile", "classical"),
        ("ebs", "statistical"),
        ("cusum", "statistical"),
        ("ewma", "statistical"),
        ("seasonal_hybrid", "statistical"),
        ("stl_residual", "statistical"),
    ],
)
def test_auto_mode_resolves_selected_model_family(model_type, expected_mode):
    assert (
        ForeblocksAnomalyDetector(model_type=model_type).detection_mode == expected_mode
    )


def test_i_transformer_scores_each_window_independently():
    import torch

    detector = ForeblocksAnomalyDetector(
        model_type="i_transformer",
        d_model=16,
        n_layers=1,
        n_heads=2,
        window_size=4,
        dropout=0.0,
        device="cpu",
    )
    model = detector.mode.build_model(detector.config, 2).eval()
    batch = torch.randn(5, 4, 2)
    scores = detector.mode.score_batch(model, batch, detector.config)
    alone = np.concatenate(
        [detector.mode.score_batch(model, row[None], detector.config) for row in batch]
    )
    assert scores.shape == (5,)
    np.testing.assert_allclose(scores, alone, rtol=1e-5, atol=1e-6)
