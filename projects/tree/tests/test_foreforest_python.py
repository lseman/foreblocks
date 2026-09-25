"""Python-binding regressions for foreforest (skipped unless the module is built).

Run with the build directory on the path, e.g.
    PYTHONPATH=build pytest tests/test_foreforest_python.py
"""

import gc

import numpy as np
import pytest

ff = pytest.importorskip("foreforest")


def _binary_data(n=4000, p=12, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + 0.5 * X[:, 1] * X[:, 2] + rng.normal(0, 0.3, n) > 0).astype(np.float64)
    return X, y


def _gbdt(n_estimators=40, cuda=False):
    cfg = ff.ForeForestConfig()
    cfg.mode = ff.Mode.GBDT
    cfg.n_estimators = n_estimators
    cfg.objective = ff.Objective.BinaryLogloss
    cfg.enable_cuda_backend = cuda
    return cfg


def test_returned_arrays_own_their_memory():
    # Arrays used to point at freed C++ vectors: repeated calls and garbage
    # collection changed their contents.
    X, y = _binary_data()
    model = ff.ForeForest(_gbdt())
    model.fit_complete(X, y)
    first = model.predict(X)
    snapshot = first.copy()
    for _ in range(20):
        model.predict(X)
        model.predict_margin(X)
        model.feature_importance_gain()
        gc.collect()
    np.testing.assert_array_equal(first, snapshot)
    np.testing.assert_array_equal(model.predict(X), snapshot)
    assert ((first > 0) & (first < 1)).all()


def test_gbdt_learns_and_cuda_matches_cpu():
    X, y = _binary_data()
    cpu = ff.ForeForest(_gbdt(cuda=False))
    cpu.fit_complete(X, y)
    accuracy = ((cpu.predict(X) > 0.5) == y).mean()
    assert accuracy > 0.9
    gpu = ff.ForeForest(_gbdt(cuda=True))
    gpu.fit_complete(X, y)
    # Float32 GPU histograms may flip near-tied splits; predictions stay close.
    assert np.abs(gpu.predict(X) - cpu.predict(X)).mean() < 0.02


def test_isolation_forest_api():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 5))
    X[:30] += 6.0
    model = ff.IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(X)
    scores = model.anomaly_score(X)
    assert scores.shape == (3000,) and ((scores > 0) & (scores <= 1)).all()
    assert scores[:30].min() > np.median(scores[30:])
    np.testing.assert_allclose(model.score_samples(X), -scores)
    np.testing.assert_allclose(model.decision_function(X), -scores - model.offset)
    labels = model.predict(X)
    assert set(np.unique(labels)) <= {-1, 1}
    assert (labels[:30] == -1).mean() > 0.9
    again = ff.IsolationForest(n_estimators=100, contamination=0.01, random_state=0).fit(X)
    np.testing.assert_array_equal(again.anomaly_score(X), scores)
    extended = ff.IsolationForest(extension_level=-1, random_state=0).fit(X)
    assert extended.anomaly_score(X)[:30].mean() > extended.anomaly_score(X)[30:].mean()
    with pytest.raises(RuntimeError):
        ff.IsolationForest().score_samples(X)


def test_quantized_training_is_deterministic_and_accurate():
    X, y = _binary_data(n=6000)
    exact = ff.ForeForest(_gbdt())
    exact.fit_complete(X, y)
    runs = []
    for _ in range(2):
        cfg = _gbdt()
        cfg.quantized_gradients = True
        model = ff.ForeForest(cfg)
        model.fit_complete(X, y)
        runs.append(model.predict(X))
    np.testing.assert_array_equal(runs[0], runs[1])
    acc_exact = ((exact.predict(X) > 0.5) == y).mean()
    acc_quant = ((runs[0] > 0.5) == y).mean()
    assert acc_quant > acc_exact - 0.01
    cfg = _gbdt(cuda=True)
    cfg.quantized_gradients = True
    cfg.quantized_gradient_bits = 4
    model = ff.ForeForest(cfg)
    model.fit_complete(X, y)
    assert ((model.predict(X) > 0.5) == y).mean() > 0.85


def _cuda_available():
    try:
        cfg = _gbdt(n_estimators=1)
        cfg.device = ff.Device.CUDA
        X, y = _binary_data(n=200)
        ff.ForeForest(cfg).fit_complete(X, y)
        return True
    except RuntimeError:
        return False


@pytest.mark.skipif(not _cuda_available(), reason="no CUDA device")
def test_gpu_training_matches_cpu_and_is_deterministic():
    X, y = _binary_data(n=20000)
    X[np.random.default_rng(1).random(X.shape) < 0.05] = np.nan
    X_valid, y_valid = _binary_data(n=4000, seed=2)

    cpu = ff.ForeForest(_gbdt(n_estimators=60))
    cpu.fit_complete(X, y)

    def gpu_model(**overrides):
        cfg = _gbdt(n_estimators=60)
        cfg.device = ff.Device.CUDA
        for key, value in overrides.items():
            setattr(cfg, key, value)
        return ff.ForeForest(cfg)

    runs = []
    for _ in range(2):
        model = gpu_model()
        model.fit_complete(X, y)
        runs.append(model.predict(X))
    np.testing.assert_array_equal(runs[0], runs[1])
    # Same algorithm, float margins / fixed-point gradients on the device.
    assert np.abs(runs[0] - cpu.predict(X)).mean() < 0.01

    contrib = model.predict_contrib(X[:50])
    np.testing.assert_allclose(contrib.sum(axis=1), model.predict_margin(X[:50]), atol=1e-6)

    stopper = gpu_model(early_stopping_enabled=True, early_stopping_rounds=5, n_estimators=400,
                        learning_rate=0.3)
    stopper.fit_complete(X, y, X_valid, y_valid)
    assert 0 < stopper.best_iteration() <= stopper.size() <= 400

    dart = gpu_model(dart_enabled=True)  # unsupported on GPU: trains on the CPU path
    dart.fit_complete(X, y)
    assert ((dart.predict(X) > 0.5) == y).mean() > 0.85
