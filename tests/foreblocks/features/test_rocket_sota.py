"""Hydra, SelF-Rocket, kernel pruning (POCKET / S-ROCKET) and GMP pooling."""

import numpy as np
import pytest
from sklearn.base import clone
from test_rocket import _conv9, _minirocket_reference, _panel, _synthetic_classes

from foreblocks.features import (
    Hydra,
    MiniRocket,
    MultiRocket,
    Rocket,
    RocketClassifier,
    SelfRocket,
    SparseScaler,
    pocket_select,
    srocket_select,
)
from foreblocks.features.rocket._kernels import MINIROCKET_INDICES
from foreblocks.features.rocket.minirocket import MiniRocketParameters


def _hydra_reference(model, x):
    """Loop-based Hydra: per group and time step, the arg-max kernel gains its
    response and the arg-min kernel gains one count."""
    x = x[:, None] if x.ndim == 2 else x
    x = x.astype(np.float64)
    k, g_rep = model.k, model.groups_per_representation_
    rows = []
    for series in x:
        diff = np.diff(series, axis=-1)
        feats = []
        for i, (d, p) in enumerate(zip(model.dilations_, model.paddings_)):
            for r in range(model.n_representations_):
                source = series if r == 0 else diff
                w = model.weights_[i][r][:, 0].numpy().astype(np.float64)  # [k * g_rep, 9]
                counts_max = np.zeros((g_rep, k))
                counts_min = np.zeros((g_rep, k))
                for grp in range(g_rep):
                    if model.n_channels_in_ > 1:
                        signal = source[model.channel_indices_[i][r][grp].numpy()].sum(0)
                    else:
                        signal = source[0]
                    padded = np.pad(signal, (p, p))
                    length = len(padded) - 8 * d
                    z = np.stack(
                        [
                            np.array([w[grp * k + j] @ padded[t : t + 9 * d : d] for t in range(length)])
                            for j in range(k)
                        ]
                    )
                    for t in range(length):
                        counts_max[grp, z[:, t].argmax()] += z[:, t].max()
                        counts_min[grp, z[:, t].argmin()] += 1
                feats += [counts_max.ravel(), counts_min.ravel()]
        rows.append(np.concatenate(feats))
    return np.asarray(rows)


@pytest.mark.parametrize("channels", [1, 3])
def test_hydra_matches_loop_reference(channels):
    x = _panel(n=3, channels=channels, length=40)
    model = Hydra(k=4, g=8, seed=0).fit(x)
    features = model.transform(x)
    assert features.shape == (3, model.n_features_out_)
    np.testing.assert_allclose(features, _hydra_reference(model, x), rtol=1e-4, atol=1e-4)


def test_hydra_is_seeded_and_handles_short_series():
    x = _panel(n=4, length=64)
    np.testing.assert_array_equal(
        Hydra(seed=1).fit(x).transform(x), clone(Hydra(seed=1)).fit(x).transform(x)
    )
    short = _panel(n=2, length=6)
    assert Hydra(seed=0).fit(short).transform(short).shape[0] == 2


def test_sparse_scaler_keeps_zeros_and_damps_sparse_columns():
    x = np.array([[0.0, 4.0], [0.0, 9.0], [4.0, 16.0], [0.0, 1.0]])
    out = SparseScaler().fit(x).transform(x)
    assert out[0, 0] == 0 and out[1, 0] == 0
    root = np.sqrt(x)
    scale = root.std(axis=0, ddof=1) + (root == 0).mean(axis=0) ** 4 + 1e-8
    np.testing.assert_allclose(out[:, 1], (root[:, 1] - root[:, 1].mean()) / scale[1], rtol=1e-6)


def test_gmp_pooling_matches_naive_reference():
    x = _panel(n=2, length=60)[:, None]
    params = MiniRocketParameters(x, 168, 32, np.random.default_rng(0))
    all_five = params.transform(x, 5)
    n = len(params.biases)
    np.testing.assert_allclose(all_five[:, : 4 * n], params.transform(x, 4), atol=1e-6)
    np.testing.assert_allclose(
        all_five[:, : 4 * n], _minirocket_reference(params, x, multi=True), rtol=1e-4, atol=1e-4
    )
    expected = np.zeros((len(x), n))
    for s, series in enumerate(x):
        f = 0
        for d, dilation in enumerate(params.dilations):
            for k, taps in enumerate(MINIROCKET_INDICES):
                conv = _conv9(series, dilation, taps)
                if (d + k) % 2:
                    conv = conv[4 * dilation : -4 * dilation]
                for _ in range(params.features_per_dilation[d]):
                    expected[s, f] = conv.max() - params.biases[f]
                    f += 1
    np.testing.assert_allclose(all_five[:, 4 * n :], expected, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: Rocket(num_kernels=40, seed=0),
        lambda: MiniRocket(num_features=336, seed=0),
        lambda: MultiRocket(num_features=8 * 84 * 2, seed=0),
    ],
)
def test_select_kernels_computes_exactly_the_kept_columns(factory):
    x = _panel(n=5, channels=2, length=80)
    model = factory().fit(x)
    groups = model.kernel_groups_
    assert len(groups) == model.n_features_out_
    kept = np.unique(groups)[::3]
    pruned = model.select_kernels(kept)
    out = pruned.transform(x)
    assert out.shape == (5, pruned.n_features_out_)
    np.testing.assert_array_equal(out, model.transform(x)[:, np.isin(groups, kept)])


def _informative_features(seed=0):
    """120 samples, 50 kernels x 2 features; only kernels 0-4 carry the label."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, 120)
    x = rng.normal(size=(120, 100))
    for g in range(5):
        x[:, 2 * g] += 2.0 * (y == g % 3)
    return x, y, np.repeat(np.arange(50), 2)


@pytest.mark.parametrize("select", [pocket_select, srocket_select])
def test_pruning_selectors_keep_the_requested_informative_kernels(select):
    x, y, groups = _informative_features()
    kept = select(x, y, groups, keep=5)
    assert len(kept) == 5 and np.all(np.diff(kept) > 0)
    assert len(set(kept) & set(range(5))) >= 4
    assert len(select(x, y, groups, keep=0.2)) == 10


def test_srocket_is_seeded():
    x, y, groups = _informative_features()
    a = srocket_select(x, y, groups, keep=8, seed=3, generations=5)
    b = srocket_select(x, y, groups, keep=8, seed=3, generations=5)
    np.testing.assert_array_equal(a, b)


def test_selfrocket_selects_a_candidate_and_falls_back_to_default():
    x, y = _synthetic_classes(n_per_class=12, length=100)
    model = SelfRocket(num_features=168, seed=0).fit(x, y)
    assert model.selection_ in model.scores_
    assert len(model.scores_) == 15
    out = model.transform(x)
    assert out.shape == (len(x), model.n_features_out_)
    assert model.n_features_out_ in (168, 336)

    strict = SelfRocket(num_features=168, seed=0, vote_threshold=1.01, default=("base", "lspv"))
    assert strict.fit(x, y).selection_ == ("base", "lspv")
    with pytest.raises(ValueError, match="labels"):
        SelfRocket().fit(x)


@pytest.mark.parametrize(
    "transformer, params, kwargs",
    [
        ("hydra", {"seed": 0}, {}),
        ("multirocket-hydra", {"multirocket": {"num_features": 8 * 84, "seed": 0}, "hydra": {"seed": 0}}, {}),
        ("selfrocket", {"num_features": 168, "seed": 0}, {}),
        ("minirocket", {"num_features": 840, "seed": 0}, {"pruning": "pocket", "prune_keep": 0.3}),
        ("rocket", {"num_kernels": 200, "seed": 0}, {"pruning": "s-rocket", "prune_keep": 0.3,
                                                     "prune_params": {"seed": 0, "generations": 10}}),
        ("multirocket", {"num_features": 8 * 84, "seed": 0}, {"pruning": "pocket", "prune_keep": 20}),
    ],
)
def test_classifier_with_new_transforms_and_pruning(transformer, params, kwargs):
    x, y = _synthetic_classes()
    x_test, y_test = _synthetic_classes(seed=1)
    model = RocketClassifier(transformer=transformer, transformer_params=params, **kwargs).fit(x, y)
    assert (model.predict(x_test) == y_test).mean() > 0.9
    if "pruning" in kwargs:
        n_kernels = len(np.unique(model.transformer_.kernel_groups_))
        assert n_kernels == len(model.kept_kernels_)
        assert model.features(x_test).shape[1] == model.transformer_.n_features_out_


def test_pruning_rejects_unsupported_transforms():
    x, y = _synthetic_classes(n_per_class=5)
    with pytest.raises(ValueError, match="single Rocket"):
        RocketClassifier("hydra", {"seed": 0}, pruning="pocket").fit(x, y)
