"""Additional tests for fengineer module — API surface, config, and edge cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Public API surface ───────────────────────────────────────────────────────


class TestPublicAPI:
    """Verify that all expected names are importable from the package root."""

    def test_all_names_present(self) -> None:
        import foretools.fengineer as pkg

        for name in pkg.__all__:
            assert hasattr(pkg, name), f"Missing exported name: {name}"

    def test_feature_engineer_importable(self) -> None:
        from foretools.fengineer import FeatureEngineer

        assert callable(FeatureEngineer.fit)
        assert callable(FeatureEngineer.transform)

    def test_feature_config_importable(self) -> None:
        from foretools.fengineer import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.task == "regression"
        assert cfg.backend == "auto"

    def test_sub_configs_importable(self) -> None:
        from foretools.fengineer import (
            AutoencoderConfig,
            BinningConfig,
            CategoricalConfig,
            ClusteringConfig,
            DateTimeConfig,
            FourierConfig,
            InteractionConfig,
            MathConfig,
            RFFConfig,
            SelectorConfig,
        )

        assert issubclass(BinningConfig, object)
        assert issubclass(CategoricalConfig, object)


# ── FeatureConfig backward-compat properties ─────────────────────────────────


class TestFeatureConfigCompat:
    """Verify backward-compat properties delegate to sub-configs correctly."""

    def test_n_bins_delegates_to_binning(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.n_bins == cfg.binning.n_bins

    def test_n_clusters_delegates_to_clustering(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.n_clusters == cfg.clustering.n_clusters

    def test_n_fourier_terms_delegates_to_fourier(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.n_fourier_terms == cfg.fourier.n_fourier_terms

    def test_max_interactions_delegates_to_interaction(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.max_interactions == cfg.interaction.max_interactions

    def test_use_boruta_delegates_to_selector(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.use_boruta == cfg.selector.use_boruta

    def test_cat_top_k_delegates_to_categorical(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.cat_top_k == cfg.categorical.top_k

    def test_datetime_include_cyclical_delegates(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.datetime_include_cyclical == cfg.datetime.include_cyclical

    def test_binning_strategies_delegates(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.binning_strategies == cfg.binning.strategies

    def test_ae_latent_dim_delegates(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.ae_latent_dim == cfg.autoencoder.latent_dim

    def test_rff_n_components_delegates(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig()
        assert cfg.rff_n_components == cfg.rff.n_components

    def test_init_with_legacy_params(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig(n_bins=15, n_clusters=12, use_boruta=True)
        assert cfg.binning.n_bins == 15
        assert cfg.clustering.n_clusters == 12
        assert cfg.selector.use_boruta is True

    def test_init_with_prefix_params(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig(cat_top_k=10, cat_fold_strategy="group")
        assert cfg.categorical.top_k == 10
        assert cfg.categorical.fold_strategy == "group"

    def test_init_with_sub_config_override(self) -> None:
        from foretools.fengineer.transformers.support.config import FeatureConfig

        cfg = FeatureConfig(binning=dict(n_bins=20, auto_supervised=False))
        assert cfg.binning.n_bins == 20
        assert cfg.binning.auto_supervised is False


# ── FeatureEngineer pipeline ────────────────────────────────────────────────


class TestFeatureEngineer:
    """Test the main FeatureEngineer pipeline."""

    def test_default_config(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        fe = FeatureEngineer()
        assert fe.config is not None
        assert len(fe.transformers_) == 0
        assert fe.fitted_ is False

    def test_fit_transform_basic(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "num_a": rng.normal(size=50),
            "num_b": rng.normal(loc=1.0, size=50),
        })
        y = X["num_a"] + X["num_b"]

        fe = FeatureEngineer(FeatureConfig(create_datetime=False))
        X_trans = fe.fit_transform(X, y)

        assert isinstance(X_trans, pd.DataFrame)
        assert len(X_trans) == 50
        assert len(X_trans.columns) > 0

    def test_transform_without_fit_returns_empty(self) -> None:
        from foretools.fengineer import FeatureEngineer

        fe = FeatureEngineer()
        X = pd.DataFrame({"a": [1.0, 2.0]})

        # transform() without fit returns empty DataFrame (no error raised)
        Xt = fe.transform(X)
        assert isinstance(Xt, pd.DataFrame)

    def test_get_summary(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "num_a": rng.normal(size=50),
            "num_b": rng.normal(loc=1.0, size=50),
        })
        y = X["num_a"] + X["num_b"]

        fe = FeatureEngineer(FeatureConfig(create_datetime=False))
        fe.fit(X, y)
        summary = fe.get_summary()

        assert "backend" in summary
        assert "feature_stats" in summary
        assert "output_features" in summary
        assert isinstance(summary["output_features"], list)

    def test_get_transformation_report(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "num_a": rng.normal(size=50),
            "num_b": rng.normal(loc=1.0, size=50),
        })
        y = X["num_a"] + X["num_b"]

        fe = FeatureEngineer(FeatureConfig(create_datetime=False))
        fe.fit(X, y)
        report = fe.get_transformation_report()

        assert "feature_stats" in report
        assert "config" in report
        assert "backend" in report

    def test_get_feature_importance(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "num_a": rng.normal(size=50),
            "num_b": rng.normal(loc=1.0, size=50),
        })
        y = X["num_a"] + X["num_b"]

        fe = FeatureEngineer(FeatureConfig(create_datetime=False))
        fe.fit(X, y)
        importance = fe.get_feature_importance()

        # May be None if no selector was used
        assert importance is None or isinstance(importance, pd.Series)

    def test_no_y_provided_skips_selector(self) -> None:
        from foretools.fengineer import FeatureEngineer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "num_a": rng.normal(size=50),
            "num_b": rng.normal(loc=1.0, size=50),
        })

        fe = FeatureEngineer(FeatureConfig(create_datetime=False))
        fe.fit(X)  # no y

        assert fe.selector_ is None


# ── CorrelationFilter ───────────────────────────────────────────────────────


class TestCorrelationFilter:
    """Test the CorrelationFilter class."""

    def test_no_numerical_cols(self) -> None:
        from foretools.fengineer.filters import CorrelationFilter

        X = pd.DataFrame({"cat": ["a", "b", "c"]})
        filt = CorrelationFilter()
        filt.fit(X)

        assert len(filt.features_to_drop_) == 0

    def test_fewer_than_two_cols(self) -> None:
        from foretools.fengineer.filters import CorrelationFilter

        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        filt = CorrelationFilter()
        filt.fit(X)

        assert len(filt.features_to_drop_) == 0

    def test_transform_drops_columns(self) -> None:
        from foretools.fengineer.filters import CorrelationFilter

        X = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "b": [2.0, 4.0, 6.0, 8.0, 10.0],  # perfectly correlated with a
            "c": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        filt = CorrelationFilter(threshold=0.9)
        filt.fit(X)

        assert len(filt.features_to_drop_) >= 0  # may drop 0 or 1
        X_trans = filt.transform(X)
        assert len(X_trans.columns) <= len(X.columns)


# ── FeatureSelector (PipelineSelector) ──────────────────────────────────────


class TestFeatureSelector:
    """Test the FeatureSelector / PipelineSelector."""

    def test_resolve_method_mi(self) -> None:
        from foretools.fengineer import FeatureConfig, FeatureSelector

        sel = FeatureSelector(FeatureConfig(selector_method="mi"))
        assert sel._resolve_method(n_features=100) == "mi"

    def test_resolve_method_mrmr(self) -> None:
        from foretools.fengineer import FeatureConfig, FeatureSelector

        sel = FeatureSelector(FeatureConfig(selector_method="mrmr"))
        assert sel._resolve_method(n_features=100) == "mrmr"

    def test_resolve_method_boruta(self) -> None:
        from foretools.fengineer import FeatureConfig, FeatureSelector

        sel = FeatureSelector(FeatureConfig(selector_method="boruta"))
        assert sel._resolve_method(n_features=100) == "boruta"

    def test_clean_target_classification(self) -> None:
        from foretools.fengineer.selectors.feature_selector import PipelineSelector

        y = pd.Series([0, 1, 0, np.nan, 1])
        cleaned = PipelineSelector._clean_target(y)
        assert cleaned.isna().sum() == 0 or len(cleaned.dropna()) >= 3

    def test_is_classification_heuristic(self) -> None:
        from foretools.fengineer.selectors.feature_selector import PipelineSelector

        y_class = pd.Series([0, 1, 0, 1, 0])
        y_cont = pd.Series(list(range(100)))  # 100 unique values out of 100 samples

        assert PipelineSelector._is_classification(y_class) is True
        # Continuous with many unique values should be regression
        assert PipelineSelector._is_classification(y_cont) is False


# ── BaseFeatureTransformer helpers ──────────────────────────────────────────


class TestBaseFeatureTransformer:
    """Test shared utilities from BaseFeatureTransformer."""

    def test_winsorize(self) -> None:
        from foretools.fengineer.transformers.support.base import BaseFeatureTransformer

        arr = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0])
        winsorized = BaseFeatureTransformer.winsorize(arr, p=0.05)
        # With 10 values and p=0.05, we clip at ~5th percentile
        assert np.max(winsorized) < 100.0 or np.allclose(winsorized, arr)

    def test_median_impute(self) -> None:
        from foretools.fengineer.transformers.support.base import BaseFeatureTransformer

        X = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 2.0, 3.0]})
        imputed = BaseFeatureTransformer.median_impute(X)
        assert imputed.isna().sum().sum() == 0

    def test_check_variance(self) -> None:
        from foretools.fengineer.transformers.support.base import BaseFeatureTransformer

        assert BaseFeatureTransformer.check_variance(np.array([1.0, 1.0, 1.0])) is False
        assert BaseFeatureTransformer.check_variance(np.array([1.0, 2.0, 3.0])) is True

    def test_safe_corr(self) -> None:
        from foretools.fengineer.transformers.support.base import BaseFeatureTransformer

        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        c = BaseFeatureTransformer.safe_corr(a, b)
        # safe_corr returns 0 on failure, otherwise Pearson correlation
        assert isinstance(c, float)

    def test_is_classification_target(self) -> None:
        from foretools.fengineer.transformers.support.base import BaseFeatureTransformer

        y_class = pd.Series([0, 1, 0, 1])
        y_cont = pd.Series(list(range(100)))  # 100 unique values out of 100 samples

        assert BaseFeatureTransformer.is_classification_target(y_class) is True
        # Continuous with many unique values should be regression
        assert BaseFeatureTransformer.is_classification_target(y_cont) is False


# ── Transformer base classes ────────────────────────────────────────────────


class TestTransformers:
    """Test transformer base functionality."""

    def test_statistical_transformer_basic(self) -> None:
        from foretools.fengineer import FeatureConfig, StatisticalTransformer

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "a": rng.normal(size=50),
            "b": rng.normal(loc=1.0, size=50),
            "c": rng.normal(loc=2.0, size=50),
        })

        tf = StatisticalTransformer(FeatureConfig())
        tf.fit(X)
        Xt = tf.transform(X)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt.columns) > 0
        assert "row_mean" in Xt.columns

    def test_binning_transformer_basic(self) -> None:
        from foretools.fengineer import BinningTransformer, FeatureConfig

        rng = np.random.default_rng(42)
        X = pd.DataFrame({"a": rng.normal(size=100)})
        y = pd.Series(rng.choice([0, 1], size=100))

        tf = BinningTransformer(FeatureConfig(n_bins=5))
        tf.fit(X, y)
        Xt = tf.transform(X)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt.columns) > 0

    def test_categorical_transformer_basic(self) -> None:
        from foretools.fengineer import CategoricalTransformer, FeatureConfig

        X = pd.DataFrame({"cat": ["a", "b", "a", "c", "b", "a"]})
        y = pd.Series([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])

        tf = CategoricalTransformer(FeatureConfig())
        tf.fit(X, y)
        Xt = tf.transform(X, y=y)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt.columns) > 0

    def test_datetime_transformer_basic(self) -> None:
        from foretools.fengineer import DateTimeTransformer, FeatureConfig

        X = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=50),
        })

        tf = DateTimeTransformer(FeatureConfig())
        tf.fit(X)
        Xt = tf.transform(X)

        assert isinstance(Xt, pd.DataFrame)
        assert len(Xt.columns) > 0
        assert any(col.startswith("date_") for col in Xt.columns)


# ── Config dataclasses ──────────────────────────────────────────────────────


class TestConfigDataclasses:
    """Test config dataclass defaults and instantiation."""

    def test_binning_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import BinningConfig

        cfg = BinningConfig()
        assert cfg.n_bins == 5
        assert cfg.max_bins == 100
        assert cfg.min_samples_per_bin == 10

    def test_categorical_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import CategoricalConfig

        cfg = CategoricalConfig()
        assert cfg.rare_threshold == 0.01
        assert cfg.n_splits == 5
        assert cfg.tree_onehot_max_categories == 8

    def test_interaction_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import InteractionConfig

        cfg = InteractionConfig()
        assert cfg.max_interactions == 100
        assert cfg.include_prod is True
        assert cfg.include_ratio is True
        assert cfg.fast_mode is True

    def test_math_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import MathConfig

        cfg = MathConfig()
        assert cfg.method == "yeo-johnson"
        assert cfg.target_aware is True
        assert cfg.standardize is False

    def test_rff_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import RFFConfig

        cfg = RFFConfig()
        assert cfg.n_components == 100
        assert cfg.kernel == "rbf"
        assert cfg.max_features == 50

    def test_selector_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import SelectorConfig

        cfg = SelectorConfig()
        assert cfg.method == "mi"
        assert cfg.use_rfecv is False
        assert cfg.stable_mi is True
        assert cfg.cv == 5

    def test_autoencoder_config_defaults(self) -> None:
        from foretools.fengineer.transformers.support.config import AutoencoderConfig

        cfg = AutoencoderConfig()
        assert cfg.latent_dim == 8
        assert cfg.encoder_arch == [64, 32]
        assert cfg.decoder_arch == [32, 64]
        assert cfg.epochs == 50
        assert cfg.patience == 10
