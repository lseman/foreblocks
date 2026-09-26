"""Tests for OpenFE-style boosted features, leakage-safe target encoding and
related fengineer fixes."""

import warnings

import numpy as np
import pandas as pd
import pytest

from foretools.fengineer import (
    BoostedFeatureGenerator,
    CategoricalTransformer,
    FeatureConfig,
    FeatureEngineer,
    InteractionTransformer,
)


def _interaction_data(n: int = 2000, seed: int = 0, cls: bool = False):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    X["store"] = rng.choice([f"s{i}" for i in range(20)], n)
    scale = 1 + X["store"].str[1:].astype(int).to_numpy() / 10
    X["price"] = rng.lognormal(size=n) * scale
    f = 3 * X.x0 * X.x1 + 0.5 * X.x2 + rng.normal(scale=0.3, size=n)
    y = (f > np.median(f)).astype(int) if cls else f
    return X, pd.Series(y)


# ── BoostedFeatureGenerator ─────────────────────────────────────────────────


class TestBoostedFeatureGenerator:
    def test_finds_multiplicative_interaction(self) -> None:
        X, y = _interaction_data()
        gen = BoostedFeatureGenerator(FeatureConfig()).fit(X, y)
        assert gen.feature_names_, "expected at least one generated feature"
        assert gen.feature_names_[0] in {"x0__mul__x1", "x1__mul__x0"}
        assert gen.conditional_gains_[gen.feature_names_[0]] > 0.05

    def test_transform_schema_and_values(self) -> None:
        X, y = _interaction_data()
        gen = BoostedFeatureGenerator(FeatureConfig()).fit(X, y)
        out = gen.transform(X.iloc[:10])
        assert list(out.columns) == gen.feature_names_
        assert out.index.equals(X.index[:10])
        if "x0__mul__x1" in out.columns:
            np.testing.assert_allclose(
                out["x0__mul__x1"], (X.x0 * X.x1).iloc[:10], rtol=1e-5
            )

    def test_binary_and_multiclass_classification(self) -> None:
        X, y = _interaction_data(cls=True)
        gen = BoostedFeatureGenerator(FeatureConfig(task="classification")).fit(X, y)
        assert gen.task_ == "binary"
        assert np.isfinite(gen.base_loss_) and np.isfinite(gen.null_gain_)
        assert list(gen.transform(X).columns) == gen.feature_names_

        y3 = pd.Series(np.digitize(X.x0 * X.x1, [-0.5, 0.5]))
        gen3 = BoostedFeatureGenerator(FeatureConfig(task="classification")).fit(X, y3)
        assert gen3.task_ == "multiclass"
        assert gen3.transform(X).shape[0] == len(X)

    def test_group_features_handle_unseen_keys(self) -> None:
        X, y = _interaction_data()
        gen = BoostedFeatureGenerator(FeatureConfig())
        gen._group_table(X, "price", "store")
        X_new = X.iloc[:4].copy()
        X_new["store"] = ["s0", "unseen", None, "s1"]
        rank = gen._compute(X_new, ("grp", "rank", "price", "store"))
        dev = gen._compute(X_new, ("grp", "dev", "price", "store"))
        assert np.isnan(rank[1]) and np.isnan(dev[1])
        assert 0.0 < rank[0] <= 1.0 and np.isfinite(dev[0])

    def test_rank_matches_training_distribution(self) -> None:
        X = pd.DataFrame({"g": ["a"] * 4 + ["b"] * 4, "v": [1, 2, 3, 4, 10, 20, 30, 40.0]})
        gen = BoostedFeatureGenerator(FeatureConfig())
        gen._group_table(X, "v", "g")
        probe = pd.DataFrame({"g": ["a", "b", "a"], "v": [2.5, 40.0, 0.0]})
        np.testing.assert_allclose(
            gen._compute(probe, ("grp", "rank", "v", "g")), [0.5, 1.0, 0.0]
        )

    def test_pure_noise_target_selects_nothing(self) -> None:
        rng = np.random.default_rng(3)
        X = pd.DataFrame({f"x{i}": rng.normal(size=1500) for i in range(5)})
        y = pd.Series(rng.normal(size=1500))
        gen = BoostedFeatureGenerator(FeatureConfig()).fit(X, y)
        assert len(gen.feature_names_) <= 1

    def test_no_target_or_tiny_data_is_noop(self) -> None:
        X, y = _interaction_data(n=100)
        gen = BoostedFeatureGenerator(FeatureConfig())
        assert gen.fit(X, None).transform(X).shape == (len(X), 0)
        assert gen.fit(X, y).transform(X).shape == (len(X), 0)

    def test_enabled_for_tree_backend_in_pipeline(self) -> None:
        X, y = _interaction_data(n=1500)
        cfg = FeatureConfig(
            backend="tree",
            create_boosted=True,
            boosted={"max_features": 5},
            verbose=False,
        )
        fe = FeatureEngineer(cfg).fit(X, y)
        assert "boosted" in fe.transformers_
        assert fe.feature_stats_["boosted"] >= 1
        assert fe.transform(X.iloc[:5]).shape == (5, len(fe.output_features_))

    def test_config_routing(self) -> None:
        cfg = FeatureConfig(boosted_max_features=7, boosted={"n_rounds": 2})
        assert cfg.boosted.max_features == 7
        assert cfg.boosted.n_rounds == 2
        gen = BoostedFeatureGenerator(cfg, min_gain=0.01)
        assert gen.max_features == 7 and gen.n_rounds == 2 and gen.min_gain == 0.01


# ── Target encoding ─────────────────────────────────────────────────────────


class TestTargetEncoding:
    @pytest.mark.parametrize("strategy", ["target_kfold", "loo", "james-stein"])
    def test_train_encoding_does_not_leak_noise_target(self, strategy: str) -> None:
        rng = np.random.default_rng(0)
        n = 3000
        X = pd.DataFrame({"cat": rng.choice([f"k{i}" for i in range(600)], n)})
        y = pd.Series(rng.normal(size=n))
        tf = CategoricalTransformer(FeatureConfig(), strategies=(strategy,))
        tf.fit(X, y)
        enc = tf.transform(X, y=y).iloc[:, 0]
        assert abs(np.corrcoef(enc, y)[0, 1]) < 0.08

    def test_multiclass_one_vs_rest_columns(self) -> None:
        rng = np.random.default_rng(1)
        n = 900
        cat = rng.choice(["a", "b", "c"], n)
        y = pd.Series(np.where(cat == "a", "red", rng.choice(["green", "blue"], n)))
        X = pd.DataFrame({"cat": cat})
        tf = CategoricalTransformer(
            FeatureConfig(task="classification"), strategies=("target_kfold",)
        )
        tf.fit(X, y)
        inf = tf.transform(X)
        assert list(inf.columns) == ["cat_te_blue", "cat_te_green", "cat_te_red"]
        np.testing.assert_allclose(inf.sum(axis=1), 1.0, atol=1e-6)
        assert inf.loc[cat == "a", "cat_te_red"].min() > 0.9
        train = tf.transform(X, y=y)
        assert list(train.columns) == list(inf.columns)

    def test_numeric_ordinal_target_keeps_single_column(self) -> None:
        X = pd.DataFrame({"cat": list("abcabcabcabc")})
        y = pd.Series([1.0, 2.0, 3.0] * 4)
        tf = CategoricalTransformer(
            FeatureConfig(), strategies=("target_kfold",), smoothing_prior=0.0
        )
        tf.fit(X, y)
        out = tf.transform(X)
        assert list(out.columns) == ["cat_te"]
        np.testing.assert_allclose(out["cat_te"], y)

    def test_inference_uses_same_shrinkage_as_training(self) -> None:
        X = pd.DataFrame({"cat": ["a"] * 3 + ["b"] * 30})
        y = pd.Series([10.0, 12.0, 14.0] + [0.0] * 30)
        tf = CategoricalTransformer(
            FeatureConfig(), strategies=("james-stein",), n_splits=3
        )
        tf.fit(X, y)
        enc_a = tf.transform(pd.DataFrame({"cat": ["a"]}))["cat_js"].iloc[0]
        prior = y.mean()
        # Shrunk toward the prior, not the raw category mean (12.0).
        assert prior < enc_a < 12.0

    def test_auto_smoothing_and_config_wiring(self) -> None:
        cfg = FeatureConfig(categorical={"smoothing_prior": "auto", "n_splits": 3})
        tf = CategoricalTransformer(cfg)
        assert tf.smoothing_prior == "auto" and tf.n_splits == 3

        explicit = CategoricalTransformer(cfg, smoothing_prior=0.0, n_splits=2)
        assert explicit.smoothing_prior == 0.0 and explicit.n_splits == 2

    def test_target_strategy_without_y_falls_back_to_freq(self) -> None:
        X = pd.DataFrame({"cat": list("aab")})
        tf = CategoricalTransformer(FeatureConfig(), strategies=("target_kfold",))
        tf.fit(X, None)
        assert tf.col_info_["cat"]["strategy"] == "freq"
        assert list(tf.transform(X).columns) == ["cat_freq"]


# ── Interaction min/max fix ─────────────────────────────────────────────────


def test_interaction_min_max_ops_do_not_misuse_out_argument() -> None:
    tf = InteractionTransformer(FeatureConfig())
    a = np.array([1.0, 5.0, -2.0], dtype=np.float32)
    b = np.array([3.0, 2.0, -1.0], dtype=np.float32)
    meta = {"a_mean": 0.0, "b_mean": 0.0}
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        lo = tf.operations["min"][1](a, b, meta)
        hi = tf.operations["max"][1](a, b, meta)
    np.testing.assert_array_equal(lo, [1.0, 2.0, -2.0])
    np.testing.assert_array_equal(hi, [3.0, 5.0, -1.0])
