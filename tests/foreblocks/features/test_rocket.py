"""Rocket-family transforms: kernels against naive NumPy references, plus API
contracts and an end-to-end classification check."""

import numpy as np
import pytest
import torch
from sklearn.base import clone
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline

from foreblocks.features import (
    FusedRocket,
    MiniRocket,
    MultiRocket,
    Rocket,
    RocketClassifier,
)
from foreblocks.features.rocket._kernels import MINIROCKET_INDICES
from foreblocks.features.rocket.minirocket import golden_quantiles


def _panel(n=6, channels=1, length=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, channels, length)).astype(np.float32)
    return x[:, 0] if channels == 1 else x


def _synthetic_classes(n_per_class=20, length=200, channels=1, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(length) / 100
    bases = [np.sin(2 * np.pi * 3 * t), np.sign(np.sin(2 * np.pi * 7 * t)), np.sin(2 * np.pi * 15 * t)]
    x, y = [], []
    for label, base in enumerate(bases):
        for _ in range(n_per_class):
            x.append(np.stack([base] * channels) + rng.normal(scale=0.3, size=(channels, length)))
            y.append(label)
    x = np.asarray(x, dtype=np.float32)
    return (x[:, 0] if channels == 1 else x), np.asarray(y)


# ------------------------------------------------------------------ references


def _rocket_reference(model, x):
    x = x[:, None] if x.ndim == 2 else x
    out = np.zeros((len(x), model.n_features_out_))
    for k in range(len(model.lengths_)):
        klen, d, pad = model.lengths_[k], model.dilations_[k], model.paddings_[k]
        channels = model.channel_indices_[model.channel_offsets_[k] : model.channel_offsets_[k + 1]]
        w = model.weights_[model.weight_offsets_[k] : model.weight_offsets_[k + 1]].reshape(len(channels), klen)
        for n, series in enumerate(x):
            padded = np.pad(series[channels].astype(np.float64), ((0, 0), (pad, pad)))
            positions = padded.shape[1] - (klen - 1) * d
            conv = model.biases_[k] + np.array(
                [sum(w[c] @ padded[c, i : i + klen * d : d] for c in range(len(channels))) for i in range(positions)]
            )
            out[n, 2 * k] = (conv > 0).mean()
            out[n, 2 * k + 1] = conv.max()
    return out


def _conv9(series, dilation, taps):
    weights = np.full(9, -1.0)
    weights[list(taps)] = 2.0
    length = series.shape[-1]
    padded = np.pad(series.astype(np.float64), ((0, 0), (4 * dilation, 4 * dilation)))
    return sum(weights[j] * padded[:, j * dilation : j * dilation + length] for j in range(9)).sum(axis=0)


def _minirocket_reference(params, x, multi):
    x = x[:, None] if x.ndim == 2 else x
    n_features = len(params.biases)
    out = np.zeros((len(x), (4 if multi else 1) * n_features))
    for n, series in enumerate(x):
        f = 0
        for d, dilation in enumerate(params.dilations):
            for k, taps in enumerate(MINIROCKET_INDICES):
                comb = d * len(MINIROCKET_INDICES) + k
                channels = params.channel_indices[params.channel_offsets[comb] : params.channel_offsets[comb + 1]]
                conv = _conv9(series[channels], dilation, taps)
                pad = 4 * dilation
                if (d + k) % 2:
                    conv = conv[pad:-pad]
                for _ in range(params.features_per_dilation[d]):
                    z = conv - params.biases[f]
                    positive = z > 0
                    out[n, f] = positive.mean()
                    if multi:
                        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, positive.astype(int), 0])))[::2]
                        out[n, n_features + f] = z[positive].mean() if positive.any() else 0.0
                        out[n, 2 * n_features + f] = np.flatnonzero(positive).mean() if positive.any() else -1.0
                        out[n, 3 * n_features + f] = runs.max() if len(runs) else 0
                    f += 1
    return out


# ------------------------------------------------------------------- ROCKET


@pytest.mark.parametrize("channels", [1, 3])
def test_rocket_matches_naive_convolution(channels):
    x = _panel(n=3, channels=channels, length=60)
    model = Rocket(num_kernels=40, seed=0).fit(x)
    np.testing.assert_allclose(model.transform(x), _rocket_reference(model, x), rtol=1e-4, atol=1e-4)


def test_rocket_short_series_have_valid_outputs():
    x = _panel(n=4, length=8)
    features = Rocket(num_kernels=50, seed=1).fit_transform(x)
    assert features.shape == (4, 100)
    assert np.isfinite(features).all()
    assert ((features[:, ::2] >= 0) & (features[:, ::2] <= 1)).all()


# --------------------------------------------------------------- MiniRocket


@pytest.mark.parametrize("channels", [1, 4])
def test_minirocket_matches_naive_convolution(channels):
    x = _panel(n=3, channels=channels, length=80)
    model = MiniRocket(num_features=84 * 6, seed=0).fit(x)
    np.testing.assert_allclose(
        model.transform(x), _minirocket_reference(model.parameters_, x, multi=False), atol=1e-6
    )


def test_minirocket_biases_are_quantiles_of_training_convolutions():
    x = _panel(n=1, length=100)  # a single series: every combination samples it
    params = MiniRocket(num_features=84 * 4, seed=0).fit(x).parameters_
    quantiles = golden_quantiles(len(params.biases))
    f = 0
    for d, dilation in enumerate(params.dilations):
        for taps in MINIROCKET_INDICES:
            conv = _conv9(x[:1], dilation, taps)
            count = params.features_per_dilation[d]
            np.testing.assert_allclose(
                params.biases[f : f + count], np.quantile(conv, quantiles[f : f + count]), rtol=1e-5, atol=1e-5
            )
            f += count


def test_minirocket_feature_count_and_dilation_budget():
    x = _panel(n=5, length=500)
    model = MiniRocket(num_features=10_000, seed=0).fit(x)
    assert model.n_features_out_ == 84 * (10_000 // 84)
    assert model.transform(x).shape == (5, model.n_features_out_)
    assert len(model.parameters_.dilations) <= 32
    assert model.parameters_.dilations.max() <= (500 - 1) / 8


# -------------------------------------------------------------- MultiRocket


@pytest.mark.parametrize("channels", [1, 2])
def test_multirocket_pooling_matches_naive_reference(channels):
    x = _panel(n=3, channels=channels, length=70)
    model = MultiRocket(num_features=8 * 84 * 2, seed=0).fit(x)
    features = model.transform(x)
    diff = np.diff(x[:, None] if x.ndim == 2 else x, axis=-1)
    expected = np.concatenate(
        [
            _minirocket_reference(model.parameters_, x, multi=True),
            _minirocket_reference(model.diff_parameters_, diff, multi=True),
        ],
        axis=1,
    )
    assert features.shape == (3, model.n_features_out_)
    np.testing.assert_allclose(features, expected, rtol=1e-4, atol=1e-4)


# ----------------------------------------------------------- shared contracts


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Rocket(num_kernels=30, seed=3),
        lambda: MiniRocket(num_features=168, seed=3),
        lambda: MultiRocket(num_features=8 * 84, seed=3),
    ],
)
def test_transforms_are_seeded_and_validate_inputs(factory):
    x = _panel(n=4, channels=2, length=64)
    a = factory().fit(x).transform(x)
    b = clone(factory()).fit(x).transform(x)
    np.testing.assert_array_equal(a, b)
    assert a.dtype == np.float32 and np.isfinite(a).all()

    model = factory().fit(x)
    assert model.transform(x[:0]).shape == (0, a.shape[1])
    with pytest.raises(ValueError, match="channels"):
        model.transform(x[:, :1])
    bad = x.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        model.transform(bad)
    with pytest.raises(RuntimeError, match="fit"):
        factory().transform(x)


def test_minirocket_rejects_series_shorter_than_kernel():
    with pytest.raises(ValueError, match="length >= 9"):
        MiniRocket(num_features=84).fit(_panel(length=8))


@pytest.mark.parametrize("backend, device", [("numba", "cpu"), ("triton", "cuda")])
def test_fused_rocket_backends_match_conv1d_reference(backend, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    x = _panel(n=8, length=300)
    x[0] = 0.0  # all-constant series: every PPV/MPV hits its bias-only value
    reference = FusedRocket(num_kernels=64, seed=0, device="cpu", backend="torch").fit(x)
    fused = FusedRocket(num_kernels=64, seed=0, device=device, backend=backend).fit(x)
    np.testing.assert_allclose(fused.transform(x), reference.transform(x), atol=1e-4)


# ----------------------------------------------------------------- classifier


@pytest.mark.parametrize(
    "transformer, params, channels",
    [
        ("rocket", {"num_kernels": 200, "seed": 0}, 1),
        ("minirocket", {"num_features": 840, "seed": 0}, 1),
        ("multirocket", {"num_features": 8 * 84 * 2, "seed": 0}, 1),
        ("minirocket", {"num_features": 840, "seed": 0}, 3),
        ("fused", {"num_kernels": 100, "seed": 0, "device": "cpu"}, 1),
    ],
)
def test_rocket_classifier_separates_synthetic_classes(transformer, params, channels):
    x, y = _synthetic_classes(channels=channels)
    x_test, y_test = _synthetic_classes(channels=channels, seed=1)
    model = RocketClassifier(transformer=transformer, transformer_params=params).fit(x, y)
    assert (model.predict(x_test) == y_test).mean() > 0.9
    proba = model.predict_proba(x_test)
    assert proba.shape == (len(x_test), 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    np.testing.assert_array_equal(model.classes_[proba.argmax(axis=1)], model.predict(x_test))


def test_rocket_transforms_compose_with_sklearn_pipelines():
    x, y = _synthetic_classes()
    pipeline = make_pipeline(MiniRocket(num_features=840, seed=0), RidgeClassifierCV())
    assert (pipeline.fit(x, y).predict(x) == y).mean() > 0.9
    model = RocketClassifier(transformer=Rocket(num_kernels=100, seed=0)).fit(x, y)
    assert model.transformer_ is not model.transformer  # instances are cloned
