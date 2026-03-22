import warnings

import numpy as np
import pandas as pd

from foretools import AdaptiveMRMR
from foretools.fengineer import FeatureEngineer, FeatureConfig
from foretools.fengineer.filters import CorrelationFilter
from foretools.fengineer.selectors.feature_selector import FeatureSelector
from foretools.fengineer.transformers import (
    BinningTransformer,
    CategoricalTransformer,
    DateTimeTransformer,
    FourierTransformer,
    InteractionTransformer,
    MathematicalTransformer,
    RandomFourierFeaturesTransformer,
)

def test_sota_fengineer_tree_backend_prunes_heavy_expansions():
    rng = np.random.default_rng(42)
    n_samples = 120

    dates = pd.date_range("2024-01-01", periods=n_samples, freq="D")
    city = np.where(np.arange(n_samples) % 3 == 0, "sao-paulo", "rio")

    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n_samples),
            "num_b": rng.normal(loc=2.0, scale=0.5, size=n_samples),
            "num_c": rng.normal(loc=-1.0, scale=1.5, size=n_samples),
            "city": city,
            "event_time": dates,
        }
    )
    y = 0.7 * X["num_a"] - 0.2 * X["num_b"] + (X["city"] == "sao-paulo").astype(float)

    config = FeatureConfig(
        backend="tree",
        create_datetime=True,
        create_clustering=True,
        create_rff=True,  # should be disabled by tree backend
        create_fourier=True,  # should be disabled by tree backend
        use_quantile_transform=True,  # should be disabled by tree backend
    )

    fe = FeatureEngineer(config)
    fe.fit(X, y)
    X_trans = fe.transform(X)

    assert "datetime" in fe.transformers_
    assert "clustering" not in fe.transformers_
    assert "rff" not in fe.transformers_
    assert "fourier" not in fe.transformers_
    assert fe.final_scaler_ is None

    assert any(col.startswith("event_time_") for col in X_trans.columns)
    assert len(X_trans) == len(X)


def test_sota_fengineer_linear_backend_wires_clustering():
    rng = np.random.default_rng(7)
    n_samples = 100

    X = pd.DataFrame(
        {
            "num_a": rng.normal(size=n_samples),
            "num_b": rng.normal(loc=1.5, scale=0.7, size=n_samples),
            "num_c": rng.normal(loc=-0.5, scale=1.2, size=n_samples),
        }
    )
    y = X["num_a"] * 0.5 + X["num_b"] * 0.2

    config = FeatureConfig(
        backend="linear",
        create_clustering=True,
        create_datetime=False,
        create_math_features=False,
        create_interactions=False,
        create_categorical=False,
        create_binning=False,
        create_statistical=False,
        create_rff=False,
        use_quantile_transform=False,
    )

    fe = FeatureEngineer(config)
    fe.fit(X, y)
    X_trans = fe.transform(X)

    assert "clustering" in fe.transformers_
    assert any(col.startswith("kmeans_") or col.startswith("gmm_") for col in X_trans.columns)


def test_categorical_target_encoder_group_folds_avoid_same_group_leakage():
    X = pd.DataFrame(
        {
            "cat": ["same", "same", "same", "same"],
            "group_id": ["a", "a", "b", "b"],
        }
    )
    y = pd.Series([0.0, 0.0, 10.0, 10.0])

    config = FeatureConfig(
        cat_fold_strategy="group",
        cat_group_key="group_id",
        cat_target_noise_std=0.0,
    )
    tf = CategoricalTransformer(
        config, strategies=("target_kfold",), n_splits=2, smoothing_prior=0.0
    )
    tf.fit(X, y)
    Xt = tf.transform(X, y=y)

    vals = Xt["cat_te"].to_numpy()
    assert np.allclose(vals[:2], 10.0)
    assert np.allclose(vals[2:], 0.0)


def test_categorical_target_encoder_time_folds_only_use_past_rows():
    X = pd.DataFrame(
        {
            "cat": ["same"] * 6,
            "event_time": pd.date_range("2024-01-01", periods=6, freq="D"),
        }
    )
    y = pd.Series([0.0, 0.0, 0.0, 9.0, 9.0, 9.0])

    config = FeatureConfig(
        cat_fold_strategy="time",
        cat_time_col="event_time",
        cat_target_noise_std=0.0,
    )
    tf = CategoricalTransformer(
        config, strategies=("target_kfold",), n_splits=3, smoothing_prior=0.0
    )
    tf.fit(X, y)
    Xt = tf.transform(X, y=y)

    vals = Xt["cat_te"].to_numpy()
    assert vals[3] < vals[4] < vals[5]
    assert vals[3] < 1.0
    assert vals[5] < 9.0


def test_categorical_auto_strategy_respects_tree_backend():
    X = pd.DataFrame(
        {
            "small_cat": ["a", "b", "a", "c", "b", "c"],
            "medium_cat": [f"id_{i}" for i in [1, 2, 3, 4, 5, 6]],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1])

    config = FeatureConfig(
        backend="tree",
        target_encode_threshold=2,
        cat_tree_onehot_max_categories=3,
        cat_tree_ordinal_max_categories=10,
    )
    tf = CategoricalTransformer(config, strategies=("auto",))
    tf.fit(X, y)

    assert tf.col_info_["small_cat"]["strategy"] == "onehot"
    assert tf.col_info_["medium_cat"]["strategy"] == "ordinal"


def test_categorical_auto_strategy_prefers_target_encoding_for_linear_backend():
    X = pd.DataFrame(
        {
            "cat": [f"id_{i}" for i in [1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6]],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=float)

    config = FeatureConfig(
        backend="linear",
        target_encode_threshold=2,
    )
    tf = CategoricalTransformer(
        config, strategies=("auto",), target_min_samples=4, smoothing_prior=0.0
    )
    tf.fit(X, y)

    assert tf.col_info_["cat"]["strategy"] == "target_kfold"


def test_datetime_transformer_skips_plain_text_and_avoids_parse_warnings():
    X = pd.DataFrame(
        {
            "city": ["sao", "rio", "bh", None],
            "event_time": ["2024-01-01", "2024/01/02", "Jan 03 2024", None],
        }
    )

    tf = DateTimeTransformer(FeatureConfig())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf.fit(X)
        Xt = tf.transform(X)

    messages = [str(w.message) for w in caught]
    assert tf.datetime_cols_ == ["event_time"]
    assert not any("errors='ignore'" in msg for msg in messages)
    assert not any("Could not infer format" in msg for msg in messages)
    assert any(col.startswith("event_time_") for col in Xt.columns)
    assert not any(col.startswith("city_") for col in Xt.columns)


def test_binning_transformer_avoids_small_width_kbins_warnings():
    X = pd.DataFrame({"x": [0.0] * 60 + [1.0] * 60})
    y = pd.Series(np.r_[np.zeros(60), np.ones(60)])

    tf = BinningTransformer(
        FeatureConfig(n_bins=10, binning_strategies=["quantile", "uniform"])
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf.fit(X, y)

    messages = [str(w.message) for w in caught]
    assert not any("Bins whose width are too small" in msg for msg in messages)
    assert tf.binning_transformers_["x"]["quantile"]["n_bins"] == 2
    assert tf.binning_transformers_["x"]["uniform"]["n_bins"] == 2


def test_selector_auto_prefers_mrmr_for_linear_backend():
    selector = FeatureSelector(
        FeatureConfig(
            selector_method="auto",
            backend="linear",
            use_rfecv=True,
        )
    )

    assert selector._resolve_selector_method(n_features=120) == "mrmr"


def test_selector_auto_prefers_mi_for_tree_backend_when_rfecv_is_not_viable():
    selector = FeatureSelector(
        FeatureConfig(
            selector_method="auto",
            backend="tree",
            use_rfecv=True,
            selector_auto_rfecv_max_features=40,
        )
    )

    assert selector._resolve_selector_method(n_features=120) == "mi"


def test_selector_auto_can_use_rfecv_for_small_tree_feature_sets():
    selector = FeatureSelector(
        FeatureConfig(
            selector_method="auto",
            backend="gbdt",
            use_rfecv=True,
            selector_auto_tree_method="rfecv",
            selector_auto_rfecv_max_features=40,
        )
    )

    assert selector._resolve_selector_method(n_features=24) == "rfecv"


def test_selector_auto_defaults_to_lightweight_mi_for_small_tree_feature_sets():
    selector = FeatureSelector(
        FeatureConfig(
            selector_method="auto",
            backend="gbdt",
            use_rfecv=True,
            selector_auto_rfecv_max_features=40,
        )
    )

    assert selector._resolve_selector_method(n_features=24) == "mi"


def test_fourier_transformer_limits_source_columns_and_skips_generated_features():
    X = pd.DataFrame(
        {
            "event_time_day": np.arange(20, dtype=float),
            "event_time_weekday": np.tile(np.arange(5, dtype=float), 4),
            "num_a": np.linspace(0.0, 1.0, 20),
            "num_b": np.linspace(1.0, 2.0, 20),
            "row_mean": np.linspace(2.0, 3.0, 20),
            "num_a_bin": np.tile([0.0, 1.0], 10),
            "rff_cos_0": np.linspace(-1.0, 1.0, 20),
            "num_a__prod__num_b": np.linspace(0.0, 5.0, 20),
        }
    )

    tf = FourierTransformer(
        FeatureConfig(
            create_fourier=True,
            n_fourier_terms=2,
            fourier_max_source_features=3,
        )
    )
    tf.fit(X)
    Xt = tf.transform(X)

    assert len(tf.fourier_configs_) <= 3
    assert "event_time_day" in tf.fourier_configs_
    assert "event_time_weekday" in tf.fourier_configs_
    assert "row_mean" not in tf.fourier_configs_
    assert "num_a_bin" not in tf.fourier_configs_
    assert "rff_cos_0" not in tf.fourier_configs_
    assert "num_a__prod__num_b" not in tf.fourier_configs_
    assert Xt.shape[1] <= 3 * 2 * 2


def test_interaction_prescreen_skips_generated_feature_sources_by_default():
    X = pd.DataFrame(
        {
            "num_a": np.linspace(0.0, 1.0, 40),
            "num_b": np.linspace(1.0, 2.0, 40),
            "row_mean": np.linspace(2.0, 3.0, 40),
            "num_a_bin": np.tile([0.0, 1.0], 20),
            "rff_cos_0": np.linspace(-1.0, 1.0, 40),
            "num_a__prod__num_b": np.linspace(0.0, 5.0, 40),
        }
    )

    tf = InteractionTransformer(FeatureConfig())
    pair_cols = tf._prescreen_columns(X, y=None)

    assert "num_a" in pair_cols
    assert "num_b" in pair_cols
    assert "row_mean" not in pair_cols
    assert "num_a_bin" not in pair_cols
    assert "rff_cos_0" not in pair_cols
    assert "num_a__prod__num_b" not in pair_cols


def test_mathematical_transformer_logp_avoids_log1p_warning_spam():
    X = pd.DataFrame(
        {
            "num_a": np.array([-5.0, -2.0, -1.5, -1.0, -0.5, 0.0, 1.0, 2.0, 3.0, 4.0]),
            "num_b": np.linspace(-3.0, 3.0, 10),
        }
    )
    y = pd.Series(np.linspace(0.0, 1.0, 10))

    tf = MathematicalTransformer(FeatureConfig(create_math_features=True))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tf.fit(X, y)

    messages = [str(w.message) for w in caught]
    assert not any("divide by zero encountered in log1p" in msg for msg in messages)


def test_feature_selector_mrmr_penalizes_redundant_duplicates():
    rng = np.random.default_rng(123)
    n = 220

    x1 = rng.normal(size=n)
    x2 = x1.copy()
    x3 = rng.normal(size=n)
    noise = rng.normal(scale=0.05, size=n)
    y = pd.Series(1.3 * x1 + 1.1 * x3 + noise)
    X = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "x3": x3,
            "noise": rng.normal(size=n),
        }
    )

    selector = FeatureSelector(
        FeatureConfig(
            selector_method="mrmr",
            selector_stable_mi=False,
            mi_threshold=0.0,
            min_features=2,
            max_features=2,
            mrmr_candidate_pool=4,
        )
    )
    selector.fit(X, y)

    assert selector.selection_method_ == "mrmr"
    assert "x3" in selector.selected_features_
    assert not {"x1", "x2"}.issubset(selector.selected_features_)


def test_adaptive_mrmr_supports_raw_miq_and_precomputes_redundancy():
    X = pd.DataFrame(
        {
            "lead": np.array([0.0, 1.0, 2.0, 3.0]),
            "dup": np.array([10.0, 11.0, 12.0, 13.0]),
            "compl": np.array([20.0, 21.0, 22.0, 23.0]),
        }
    )
    y = pd.Series([0.0, 1.0, 0.0, 1.0])

    class StubAdaptiveMI:
        def __init__(self):
            self.pair_calls = 0
            self._name_by_values = {
                tuple(np.asarray(X[col], dtype=np.float32).tolist()): col for col in X.columns
            }

        def score_pairwise(self, X_values, y_values, return_raw_mi=False):
            assert return_raw_mi
            return np.array([0.9, 0.8, 0.6], dtype=float)

        def score(self, xa, xb, return_raw_mi=False):
            assert return_raw_mi
            self.pair_calls += 1
            a = self._name_by_values[tuple(np.asarray(xa, dtype=np.float32).tolist())]
            b = self._name_by_values[tuple(np.asarray(xb, dtype=np.float32).tolist())]
            pair = frozenset((a, b))
            mapping = {
                frozenset(("lead", "dup")): 0.95,
                frozenset(("lead", "compl")): 0.10,
                frozenset(("dup", "compl")): 0.20,
            }
            return mapping[pair]

    selector = AdaptiveMRMR(
        scorer=StubAdaptiveMI(),
        criterion="miq",
        candidate_pool=3,
        use_raw_mi=True,
        stable_relevance=False,
        random_state=7,
    )
    selector.fit(
        X,
        y,
        min_features=2,
        max_features=2,
        mi_threshold=0.0,
        min_samples=1,
    )

    assert selector.selected_features_ == ["lead", "compl"]
    assert selector.selection_scores_ is not None
    assert selector.redundancy_matrix_ is not None
    assert selector.scorer.pair_calls == 3


def test_feature_selector_mrmr_wires_new_adaptive_mrmr_modes():
    X = pd.DataFrame(
        {
            "lead": np.array([0.0, 1.0, 2.0, 3.0]),
            "dup": np.array([10.0, 11.0, 12.0, 13.0]),
            "compl": np.array([20.0, 21.0, 22.0, 23.0]),
        }
    )
    y = pd.Series([0.0, 1.0, 0.0, 1.0])

    class StubAdaptiveMI:
        def __init__(self):
            self.pair_calls = 0
            self._name_by_values = {
                tuple(np.asarray(X[col], dtype=np.float32).tolist()): col for col in X.columns
            }

        def score_pairwise(self, X_values, y_values, return_raw_mi=False):
            assert return_raw_mi
            return np.array([0.9, 0.8, 0.6], dtype=float)

        def score(self, xa, xb, return_raw_mi=False):
            assert return_raw_mi
            self.pair_calls += 1
            a = self._name_by_values[tuple(np.asarray(xa, dtype=np.float32).tolist())]
            b = self._name_by_values[tuple(np.asarray(xb, dtype=np.float32).tolist())]
            pair = frozenset((a, b))
            mapping = {
                frozenset(("lead", "dup")): 0.95,
                frozenset(("lead", "compl")): 0.10,
                frozenset(("dup", "compl")): 0.20,
            }
            return mapping[pair]

    selector = FeatureSelector(
        FeatureConfig(
            selector_method="mrmr",
            selector_stable_mi=False,
            mi_threshold=0.0,
            min_features=2,
            max_features=2,
            min_samples=1,
            mrmr_candidate_pool=3,
            mrmr_criterion="miq",
            mrmr_use_raw_mi=True,
        )
    )
    selector.ami_scorer = StubAdaptiveMI()
    selector.fit(X, y)

    assert selector.selection_method_ == "mrmr"
    assert selector.selected_features_ == ["lead", "compl"]
    assert selector.mrmr_selector_ is not None
    assert selector.mrmr_selector_.scorer.pair_calls == 3


def test_rff_mutual_info_uses_adaptive_mi_scorer():
    class StubAdaptiveMI:
        def __init__(self):
            self.called = False

        def score_pairwise(self, X, y):
            self.called = True
            assert X.shape[1] == 3
            return np.array([0.1, 0.9, 0.4], dtype=float)

    X = pd.DataFrame(
        {
            "weak": [0.1, 0.2, 0.3, 0.4, 0.5],
            "strong": [5, 4, 3, 2, 1],
            "mid": [1, 1, 2, 2, 3],
        }
    )
    y = pd.Series([0.0, 1.0, 0.0, 1.0, 0.0])

    tf = RandomFourierFeaturesTransformer(
        FeatureConfig(create_rff=True),
        max_features=2,
        feature_selection_method="mutual_info",
    )
    tf.ami_scorer = StubAdaptiveMI()

    selected = tf._select_features_target_aware(X, y, "mutual_info")

    assert tf.ami_scorer.called
    assert selected == ["strong", "mid"]


def test_correlation_filter_adaptive_mi_keeps_better_nonlinear_proxy():
    rng = np.random.default_rng(321)
    n = 320

    base = rng.uniform(-2.0, 2.0, size=n)
    nonlinear_a = base**2
    nonlinear_b = (np.abs(base) + 0.02 * rng.normal(size=n)) ** 2
    distractor = rng.normal(size=n)
    y = pd.Series(nonlinear_a + 0.03 * rng.normal(size=n))

    X = pd.DataFrame(
        {
            "nonlinear_a": nonlinear_a,
            "nonlinear_b": nonlinear_b,
            "distractor": distractor,
        }
    )

    filt = CorrelationFilter(
        threshold=0.8,
        method="target_corr",
        dependence_metric="adaptive_mi",
    )
    filt.fit(X, y)

    assert any(
        {"nonlinear_a", "nonlinear_b"} == {c1, c2}
        for c1, c2, _ in filt.correlation_pairs_
    )
    assert len({"nonlinear_a", "nonlinear_b"} & set(filt.features_to_drop_)) == 1
    assert "distractor" not in filt.features_to_drop_
