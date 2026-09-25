"""Native algorithms: numerical oracles, learning, and inference invariants."""

import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from foreblocks.models.anomaly import (
    INNE,
    LODA,
    NATIVE_MODELS,
    AutoEncoderScorer,
    DeepIsolationForest,
    DeepSVDD,
    ForeblocksAnomalyDetector,
    GaussianMixtureScorer,
    KNNScorer,
    VAEScorer,
)

SETTINGS = {
    "inne": {"n_estimators": 8, "max_samples": 16},
    "loda": {"n_projections": 12, "n_bins": 6},
    "knn": {"n_neighbors": 3, "chunk_size": 7},
    "gmm": {"n_components": 2, "max_iter": 30},
    "dif": {
        "n_ensemble": 4,
        "n_estimators": 4,
        "max_samples": 32,
        "hidden_sizes": (8, 4),
        "representation_dim": 6,
    },
    "autoencoder": {
        "epochs": 3,
        "hidden_sizes": (12,),
        "latent_dim": 3,
        "device": "cpu",
    },
    "vae": {"epochs": 3, "hidden_sizes": (12,), "latent_dim": 3, "device": "cpu"},
    "deep_svdd": {"epochs": 3, "hidden_sizes": (12,), "latent_dim": 3, "device": "cpu"},
}


@pytest.mark.parametrize("name", SETTINGS)
def test_native_fit_query_refit_and_determinism(name):
    rng = np.random.default_rng(42)
    train = rng.normal(size=(64, 3, 2))
    query = rng.normal(size=(5, 3, 2))
    cls = NATIVE_MODELS[name]
    scorer = cls(**SETTINGS[name], seed=3).fit(train)
    scores = scorer.decision_function(query)
    assert scores.shape == (5,)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scorer.decision_scores_, scorer.decision_function(train))
    alone = np.array([scorer.decision_function(row[None])[0] for row in query])
    np.testing.assert_allclose(scores, alone, rtol=2e-5, atol=2e-6)
    repeated = cls(**SETTINGS[name], seed=3).fit(train).decision_function(query)
    np.testing.assert_allclose(scores, repeated)
    assert scorer.decision_function(np.empty((0, 3, 2))).shape == (0,)
    with pytest.raises(ValueError, match="sample shape"):
        scorer.decision_function(np.ones((2, 6)))
    scorer.fit(rng.normal(size=(64, 4)))
    assert scorer.decision_function(np.ones((2, 4))).shape == (2,)


@pytest.mark.parametrize("name", SETTINGS)
def test_pipeline_uses_fitted_native_scorers(name):
    train = np.random.default_rng(9).normal(size=(70, 2)).astype(np.float32)
    detector = ForeblocksAnomalyDetector(
        model_type=name,
        scorer_kwargs=SETTINGS[name],
        window_size=3,
        batch_size=16,
        epochs=1,
        device="cpu",
    ).fit(train)
    assert detector.detection_mode == "native"
    result = detector.predict(train[:20])
    assert result.scores.shape == (20,)
    assert np.isnan(result.scores[:2]).all()
    assert np.isfinite(result.window_scores).all()
    assert np.isfinite(result.threshold)
    detector.config.batch_size = 1
    np.testing.assert_allclose(
        result.scores, detector.predict(train[:20]).scores, rtol=2e-5, atol=2e-6
    )


@pytest.mark.parametrize("name", SETTINGS)
def test_constants_validation_and_far_anomaly(name):
    cls = NATIVE_MODELS[name]
    scorer = cls(**SETTINGS[name])
    with pytest.raises(RuntimeError, match="not fitted"):
        scorer.decision_function([[1, 2]])
    for bad in [
        np.ones(3),
        np.ones((1, 2)),
        np.empty((2, 0)),
        [[np.nan], [1]],
        [[1], [np.inf]],
    ]:
        with pytest.raises(ValueError):
            scorer.fit(bad)
    scorer.fit(np.ones((32, 2)))
    assert np.isfinite(scorer.decision_function([[1, 1], [2, 2]])).all()
    scorer.fit(np.random.default_rng(3).normal(size=(100, 2)))
    scores = scorer.decision_function([[0, 0], [30, 30]])
    assert scores[1] > scores[0]


@pytest.mark.parametrize(
    "method,expected", [("largest", [1, 8]), ("mean", [1, 6.5]), ("median", [1, 6.5])]
)
def test_knn_matches_hand_calculated_distances(method, expected):
    model = KNNScorer(
        n_neighbors=2, method=method, chunk_size=1, standardize=False
    ).fit([[0], [2], [5]])
    np.testing.assert_allclose(model.decision_function([[1], [10]]), expected)
    if method == "largest":
        np.testing.assert_allclose(model.decision_scores_, [2, 2, 3])


def test_inne_hypersphere_local_radius_ratio():
    model = INNE(n_estimators=1, max_samples=3, standardize=False).fit([[0], [1], [10]])
    np.testing.assert_allclose(
        model.decision_function([[0], [0.5], [5], [10], [30]]), [0, 0, 8 / 9, 8 / 9, 1]
    )
    model.fit(np.ones((10, 1)))
    np.testing.assert_allclose(model.decision_function([[1], [2]]), [0, 1])


def test_loda_histogram_density_and_unseen_tail():
    model = LODA(n_projections=1, n_bins=2, standardize=False).fit([[0], [0], [1], [2]])
    # The randomly signed projection swaps left/right but both bins have count 2.
    # Query bin centers avoid sign-dependent assignments at an interior edge.
    sign = model.projections_[0, 0]
    counts, edges, n = model.histograms_[0]
    query = ((edges[:-1] + edges[1:]) / 2 / sign)[:, None]
    expected = -np.log((counts + 0.1) / (n + 0.2) / np.diff(edges))
    np.testing.assert_allclose(model.decision_function(query), expected)
    assert model.decision_function([[30]])[0] > expected.max()


def test_gmm_matches_single_gaussian_and_improves_likelihood():
    model = GaussianMixtureScorer(n_components=1, standardize=False).fit([[-1], [1]])
    np.testing.assert_allclose(model.means_, [[0]], atol=1e-10)
    np.testing.assert_allclose(model.variances_, [[1]], atol=1e-10)
    np.testing.assert_allclose(
        model.decision_function([[0], [2]]), 0.5 * np.log(2 * np.pi) + np.array([0, 2])
    )
    rng = np.random.default_rng(1)
    train = np.concatenate([rng.normal(-3, 0.3, (80, 2)), rng.normal(3, 0.5, (80, 2))])
    model = GaussianMixtureScorer(n_components=2).fit(train)
    assert (np.diff(model.lower_bounds_) >= -1e-8).all()
    assert model.converged_
    assert model.decision_function([[0, 0]])[0] > model.decision_function([[-3, -3]])[0]


@pytest.mark.parametrize("cls", [AutoEncoderScorer, VAEScorer, DeepSVDD])
def test_neural_learning_and_rng_isolation(cls):
    train = np.random.default_rng(1).normal(size=(96, 4))
    torch.manual_seed(123)
    state = torch.random.get_rng_state().clone()
    model = cls(
        epochs=12, hidden_sizes=(16,), latent_dim=2, batch_size=24, device="cpu"
    ).fit(train)
    assert torch.equal(state, torch.random.get_rng_state())
    assert model.loss_history_[-1] < model.loss_history_[0]
    scores = model.decision_function(train[:5])
    np.testing.assert_array_equal(scores, model.decision_function(train[:5]))
    if cls is DeepSVDD:
        assert all(
            module.bias is None
            for module in model.model_.modules()
            if isinstance(module, torch.nn.Linear)
        )
        assert torch.all(model.model_.center.abs() >= 0.1)


def test_soft_boundary_svdd_radius_and_serialization(tmp_path):
    import pickle

    train = np.random.default_rng(4).normal(size=(50, 3))
    model = DeepSVDD(
        epochs=3, warmup_epochs=0, objective="soft_boundary", nu=0.2, device="cpu"
    ).fit(train)
    assert model.model_.radius.item() > 0
    scores = model.decision_function(train)
    squared_distance = scores + model.model_.radius.item() ** 2
    assert np.isclose(
        model.model_.radius.item(),
        np.quantile(np.sqrt(squared_distance), 0.8),
        rtol=1e-5,
    )
    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_array_equal(scores, restored.decision_function(train))


@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (KNNScorer, {"n_neighbors": 0}),
        (KNNScorer, {"method": "bad"}),
        (INNE, {"max_samples": 1}),
        (LODA, {"alpha": 0}),
        (LODA, {"n_projections": 1.5}),
        (GaussianMixtureScorer, {"reg_covar": 0}),
        (DeepIsolationForest, {"n_ensemble": 0}),
        (DeepSVDD, {"nu": 0}),
        (AutoEncoderScorer, {"epochs": 0}),
        (VAEScorer, {"hidden_sizes": ()}),
    ],
)
def test_invalid_parameters(cls, kwargs):
    with pytest.raises(ValueError):
        cls(**kwargs)


def test_no_pyod_needed_even_when_import_is_forbidden():
    code = """
import importlib.abc
import sys
class BlockPyOD(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyod" or fullname.startswith("pyod."):
            raise AssertionError("PyOD import attempted")
sys.meta_path.insert(0, BlockPyOD())
from foreblocks.models.anomaly import ForeblocksAnomalyDetector
import numpy as np
model = ForeblocksAnomalyDetector(model_type="loda", window_size=2).fit(np.arange(20.))
assert np.isfinite(model.predict(np.arange(10.)).window_scores).all()
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env={**os.environ, "PYTHONPATH": "src"},
    )


def test_native_block_fit_and_bad_mode():
    from foreblocks.models.anomaly import AnomalyBlockStack, AnomalyDetectorConfig

    with pytest.raises(ValueError, match="require detection_mode"):
        ForeblocksAnomalyDetector(model_type="inne", detection_mode="classical")
    config = AnomalyDetectorConfig(model_type="knn", window_size=3)
    windows = np.random.default_rng(9).normal(size=(30, 3, 2))
    stack = AnomalyBlockStack(["native"])
    models = stack.build_models(config, n_features=2)
    stack.fit(models, windows, config, epochs=1)
    assert np.isfinite(models["native"].scorer.decision_function(windows)).all()
