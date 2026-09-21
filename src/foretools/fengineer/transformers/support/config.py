"""Configuration dataclasses for feature engineering.

Structure
---------
Each transformer category has its own focused dataclass.
:class:`FeatureConfig` aggregates them and retains flat attributes
for backward compatibility with existing code.

Usage
-----
>>> from foretools.fengineer.transformers.config import BinningConfig
>>> cfg = BinningConfig(n_bins=15, strategies=["auto"])
>>> # or access via FeatureConfig
>>> full = FeatureConfig()
>>> full.binning.n_bins  # → 5 (default)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class BinningConfig:
    """Configuration for :class:`BinningTransformer`."""

    strategies: list[str] | None = None
    auto_supervised: bool = True
    min_bin_fraction: float = 0.01
    min_samples_per_bin: int = 10
    max_bins: int = 100
    kmeans_bins: int = 10
    shimazaki_k_range: tuple[int, int] = (4, 128)
    n_bins: int = 5


@dataclass
class CategoricalConfig:
    """Configuration for :class:`CategoricalTransformer`."""

    rare_threshold: float = 0.01
    target_encode_threshold: int = 10
    top_k: int | None = None
    use_woe: bool = False
    use_loo: bool = False
    use_james_stein: bool = False
    n_splits: int = 5
    target_min_samples: int = 100
    smoothing_prior: float = 10.0
    use_stratified_kfold: bool = True
    target_noise_std: float = 0.0
    fold_strategy: str = "auto"  # "auto" | "kfold" | "group" | "time"
    group_key: str | None = None
    time_col: str | None = None
    tree_onehot_max_categories: int = 8
    tree_ordinal_max_categories: int = 255


@dataclass
class InteractionConfig:
    """Configuration for :class:`InteractionTransformer`."""

    max_interactions: int = 100
    max_polynomials: int = 50
    max_pairs_screen: int = 800
    prescreen_topk: int = 32
    redundancy_corr: float = 0.985
    prune_redundancy: bool = False
    stability_selection: bool = True
    stability_min_freq: float = 0.5
    fast_mode: bool = True
    row_subsample: int = 5000
    n_jobs: int = -1
    prescreen_spearman: bool = False
    winsor_p: float = 0.001
    pair_corr_with_y: bool = True
    pair_max_per_feature: int = 32
    corr_avoid_redundancy: float = 0.995
    exclude_generated_sources: bool = True
    n_splits: int = 5
    min_selected_per_fold: int = 20
    importance_agg: str = "median"

    # Operations
    include_sum: bool = True
    include_diff: bool = True
    include_prod: bool = True
    include_ratio: bool = True
    include_norm_ratio: bool = True
    include_minmax: bool = True
    include_zdiff: bool = True
    include_logratio: bool = True
    include_rootprod: bool = True
    include_square: bool = True
    include_sqrt: bool = True
    include_cube: bool = False
    include_reciprocal: bool = False
    include_log: bool = False


@dataclass
class MathConfig:
    """Configuration for :class:`MathematicalTransformer`."""

    target_aware: bool = True
    target_weight: float = 0.25
    max_transforms_per_feature: int = 3
    method: str = "yeo-johnson"
    min_variance: float = 1e-6
    max_missing: float = 0.5
    improvement_threshold: float = 0.1
    standardize: bool = False
    winsor_p: float = 0.0


@dataclass
class RFFConfig:
    """Configuration for :class:`RandomFourierFeaturesTransformer`."""

    n_components: int = 100
    gamma: float | str = "auto"
    kernel: str = "rbf"
    max_features: int = 50
    feature_selection_method: str = "variance"
    handle_missing_features: str = "impute"


@dataclass
class ClusteringConfig:
    """Configuration for :class:`ClusteringTransformer`."""

    strategies: list[str] | None = None
    max_features: int = 20
    n_clusters: int = 8


@dataclass
class FourierConfig:
    """Configuration for :class:`FourierTransformer`."""

    max_source_features: int = 12
    exclude_generated_sources: bool = True
    n_fourier_terms: int = 3


@dataclass
class DateTimeConfig:
    """Configuration for :class:`DateTimeTransformer`."""

    include_cyclical: bool = True
    include_flags: bool = True
    include_elapsed: bool = True
    group_key: str | None = None
    country_holidays: str | None = None


@dataclass
class SelectorConfig:
    """Configuration for feature selection (used by PipelineSelector, MISelector, etc.)."""

    method: str = "mi"
    use_quantile_transform: bool = True
    mi_threshold: float = 0.001
    shap_threshold: float = 0.001
    mi_spearman_gate: float = 0.05
    mi_min_overlap: int = 50
    mi_bins: int = 16
    stable_mi: bool = True
    cv: int = 5
    min_freq: float = 0.5
    redundancy_prune: bool = True
    redundancy_threshold: float = 0.98
    redundancy_pool: int = 200
    mrmr_candidate_pool: int = 128
    mrmr_redundancy_weight: float = 1.0
    mrmr_criterion: str = "mid"
    mrmr_use_raw_mi: bool = False
    mrmr_redundancy_eps: float = 1e-8
    auto_linear_method: str = "mrmr"
    auto_tree_method: str = "mi"
    auto_neural_method: str = "mrmr"
    auto_rfecv_max_features: int = 80
    use_rfecv: bool = False
    rfecv_step: int | float = 0.1
    rfecv_cv: int = 5
    rfecv_min_features: int | None = None
    rfecv_max_features: int | None = None
    rfecv_patience: int = 5
    rfecv_use_ensemble: bool = True
    rfecv_stability_selection: bool = True
    use_boruta: bool = False


@dataclass
class AutoencoderConfig:
    """Configuration for :class:`AutoencoderTransformer`."""

    enabled: bool = False
    latent_dim: int = 8
    encoder_arch: list[int] = field(default_factory=lambda: [64, 32])
    decoder_arch: list[int] = field(default_factory=lambda: [32, 64])
    activation: str = "relu"
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    weight_decay: float = 1e-5
    use_bn: bool = True
    device: str = "auto"
    random_state: int = 42
    max_features: int = 100
    min_features: int = 4


# ────────────────────────────────────────────────────────────────────────────
# Main config (backward-compatible, aggregates sub-configs)
# ────────────────────────────────────────────────────────────────────────────


class FeatureConfig:
    """
    Main configuration class for feature engineering.

    Provides both flat attributes (for backward compatibility) and
    nested sub-configs (for cleaner access).

    Usage
    -----
    >>> cfg = FeatureConfig(task="classification")
    >>> cfg.task                          # flat: "classification"
    >>> cfg.binning.n_bins                # nested: 5
    >>> cfg.binning.n_bins = 15           # modify sub-config
    """

    # ── core ────────────────────────────────────────────────────────────

    task: str = "regression"
    backend: str = "auto"
    random_state: int = 42
    verbose: bool = True
    log_level: str = "INFO"

    # ── feature creation flags ─────────────────────────────────────────

    create_datetime: bool = True
    create_math_features: bool = True
    create_interactions: bool = True
    create_polynomials: bool = True
    create_categorical: bool = True
    create_binning: bool = True
    create_statistical: bool = True
    create_clustering: bool = True
    create_fourier: bool = False
    create_rff: bool = True

    # ── correlation ────────────────────────────────────────────────────

    corr_threshold: float = 0.95
    corr_filter_method: str = "variance"
    corr_dependence_metric: str = "pearson"

    # ── thresholds ─────────────────────────────────────────────────────

    rare_threshold: float = 0.01
    min_variance_threshold: float = 1e-6
    max_rows_score: int = 50000
    epsilon: float = 1e-8
    dtype_out: str = "float32"
    max_features: int = 500
    min_features: int = 1
    min_samples: int = 10

    # ── sub-configs ────────────────────────────────────────────────────

    binning: BinningConfig = BinningConfig()
    categorical: CategoricalConfig = CategoricalConfig()
    interaction: InteractionConfig = InteractionConfig()
    math: MathConfig = MathConfig()
    rff: RFFConfig = RFFConfig()
    clustering: ClusteringConfig = ClusteringConfig()
    fourier: FourierConfig = FourierConfig()
    datetime: DateTimeConfig = DateTimeConfig()
    selector: SelectorConfig = SelectorConfig()
    autoencoder: AutoencoderConfig = AutoencoderConfig()

    # ── init ───────────────────────────────────────────────────────────

    def __init__(self, **kwargs: Any) -> None:
        """
        Initialize with optional keyword arguments.

        Accepts both flat attributes (task, backend, etc.) and sub-config
        overrides (binning=dict(n_bins=15), categorical=dict(top_k=10), etc.).

        Also accepts legacy flat parameters like ``n_bins``, ``n_clusters``,
        ``n_fourier_terms``, ``max_interactions``, ``max_polynomials``, etc.
        which are routed to the appropriate sub-config.
        """
        # Copy sub-configs to avoid sharing mutable state across instances
        self.binning = BinningConfig(**self.binning.__dict__)
        self.categorical = CategoricalConfig(**self.categorical.__dict__)
        self.interaction = InteractionConfig(**self.interaction.__dict__)
        self.math = MathConfig(**self.math.__dict__)
        self.rff = RFFConfig(**self.rff.__dict__)
        self.clustering = ClusteringConfig(**self.clustering.__dict__)
        self.fourier = FourierConfig(**self.fourier.__dict__)
        self.datetime = DateTimeConfig(**self.datetime.__dict__)
        self.selector = SelectorConfig(**self.selector.__dict__)
        self.autoencoder = AutoencoderConfig(**self.autoencoder.__dict__)

        # Route legacy flat params to sub-configs
        legacy_routing = {
            "n_bins": ("binning", "n_bins"),
            "n_clusters": ("clustering", "n_clusters"),
            "n_fourier_terms": ("fourier", "n_fourier_terms"),
            "max_interactions": ("interaction", "max_interactions"),
            "max_polynomials": ("interaction", "max_polynomials"),
            "ae_latent_dim": ("autoencoder", "latent_dim"),
            "use_boruta": ("selector", "use_boruta"),
            "use_loo": ("categorical", "use_loo"),
            "use_woe": ("categorical", "use_woe"),
            "use_james_stein": ("categorical", "use_james_stein"),
            "use_rfecv": ("selector", "use_rfecv"),
            "use_quantile_transform": ("selector", "use_quantile_transform"),
        }

        compat_routing = {
            "target_encode_threshold": ("categorical", "target_encode_threshold"),
            "shap_threshold": ("selector", "shap_threshold"),
        }
        prefixes = {
            "cat_": "categorical",
            "binning_": "binning",
            "fourier_": "fourier",
            "datetime_": "datetime",
            "interaction_": "interaction",
            "selector_": "selector",
            "mrmr_": "selector",
            "rfecv_": "selector",
            "mi_": "selector",
            "ae_": "autoencoder",
        }
        for key in kwargs:
            for prefix, group in prefixes.items():
                if key.startswith(prefix):
                    attr = key[len(prefix):] if prefix not in {"mrmr_", "rfecv_", "mi_"} else key
                    compat_routing.setdefault(key, (group, attr))
                    break

        for key, value in kwargs.items():
            if isinstance(value, dict) and hasattr(self, key):
                # Sub-config override: merge dict into existing sub-config
                for k, v in value.items():
                    if hasattr(getattr(self, key), k):
                        setattr(getattr(self, key), k, v)
            elif key in legacy_routing:
                attr_group, attr_name = legacy_routing[key]
                target = getattr(self, attr_group)
                if hasattr(target, attr_name):
                    setattr(target, attr_name, value)
            elif key in compat_routing:
                attr_group, attr_name = compat_routing[key]
                target = getattr(self, attr_group)
                if hasattr(target, attr_name):
                    setattr(target, attr_name, value)
            elif hasattr(self, key):
                setattr(self, key, value)

    # ── backward-compat properties ─────────────────────────────────────

    @property
    def n_bins(self) -> int:
        return self.binning.n_bins

    @property
    def n_clusters(self) -> int:
        return self.clustering.n_clusters

    @property
    def n_fourier_terms(self) -> int:
        return self.fourier.n_fourier_terms

    @property
    def max_interactions(self) -> int:
        return self.interaction.max_interactions

    @property
    def max_polynomials(self) -> int:
        return self.interaction.max_polynomials

    @property
    def use_boruta(self) -> bool:
        return self.selector.use_boruta

    @property
    def use_loo(self) -> bool:
        return self.categorical.use_loo

    @property
    def use_woe(self) -> bool:
        return self.categorical.use_woe

    @property
    def use_james_stein(self) -> bool:
        return self.categorical.use_james_stein

    @property
    def selector_method(self) -> str:
        return self.selector.method

    @property
    def mi_threshold(self) -> float:
        return self.selector.mi_threshold

    @property
    def mi_spearman_gate(self) -> float:
        return self.selector.mi_spearman_gate

    @property
    def mi_min_overlap(self) -> int:
        return self.selector.mi_min_overlap

    @property
    def mi_bins(self) -> int:
        return self.selector.mi_bins

    @property
    def selector_stable_mi(self) -> bool:
        return self.selector.stable_mi

    @property
    def selector_cv(self) -> int:
        return self.selector.cv

    @property
    def selector_min_freq(self) -> float:
        return self.selector.min_freq

    @property
    def selector_redundancy_prune(self) -> bool:
        return self.selector.redundancy_prune

    @property
    def selector_redundancy_threshold(self) -> float:
        return self.selector.redundancy_threshold

    @property
    def selector_redundancy_pool(self) -> int:
        return self.selector.redundancy_pool

    @property
    def mrmr_candidate_pool(self) -> int:
        return self.selector.mrmr_candidate_pool

    @property
    def mrmr_redundancy_weight(self) -> float:
        return self.selector.mrmr_redundancy_weight

    @property
    def mrmr_criterion(self) -> str:
        return self.selector.mrmr_criterion

    @property
    def mrmr_use_raw_mi(self) -> bool:
        return self.selector.mrmr_use_raw_mi

    @property
    def mrmr_redundancy_eps(self) -> float:
        return self.selector.mrmr_redundancy_eps

    @property
    def selector_auto_linear_method(self) -> str:
        return self.selector.auto_linear_method

    @property
    def selector_auto_tree_method(self) -> str:
        return self.selector.auto_tree_method

    @property
    def selector_auto_neural_method(self) -> str:
        return self.selector.auto_neural_method

    @property
    def selector_auto_rfecv_max_features(self) -> int:
        return self.selector.auto_rfecv_max_features

    @property
    def rfecv_step(self) -> int | float:
        return self.selector.rfecv_step

    @property
    def rfecv_cv(self) -> int:
        return self.selector.rfecv_cv

    @property
    def rfecv_min_features(self) -> int | None:
        return self.selector.rfecv_min_features

    @property
    def rfecv_max_features(self) -> int | None:
        return self.selector.rfecv_max_features

    @property
    def rfecv_patience(self) -> int:
        return self.selector.rfecv_patience

    @property
    def rfecv_use_ensemble(self) -> bool:
        return self.selector.rfecv_use_ensemble

    @property
    def rfecv_stability_selection(self) -> bool:
        return self.selector.rfecv_stability_selection

    @property
    def use_rfecv(self) -> bool:
        return self.selector.use_rfecv

    @property
    def use_quantile_transform(self) -> bool:
        return self.selector.use_quantile_transform

    @property
    def shap_threshold(self) -> float:
        return self.selector.shap_threshold

    # Categorical flat compat
    @property
    def target_encode_threshold(self) -> int:
        return self.categorical.target_encode_threshold

    @property
    def cat_top_k(self) -> int | None:
        return self.categorical.top_k

    @property
    def cat_use_stratified_kfold(self) -> bool:
        return self.categorical.use_stratified_kfold

    @property
    def cat_target_noise_std(self) -> float:
        return self.categorical.target_noise_std

    @property
    def cat_fold_strategy(self) -> str:
        return self.categorical.fold_strategy

    @property
    def cat_group_key(self) -> str | None:
        return self.categorical.group_key

    @property
    def cat_time_col(self) -> str | None:
        return self.categorical.time_col

    @property
    def cat_tree_onehot_max_categories(self) -> int:
        return self.categorical.tree_onehot_max_categories

    @property
    def cat_tree_ordinal_max_categories(self) -> int:
        return self.categorical.tree_ordinal_max_categories

    # DateTime flat compat
    @property
    def datetime_include_cyclical(self) -> bool:
        return self.datetime.include_cyclical

    @property
    def datetime_include_flags(self) -> bool:
        return self.datetime.include_flags

    @property
    def datetime_include_elapsed(self) -> bool:
        return self.datetime.include_elapsed

    @property
    def datetime_group_key(self) -> str | None:
        return self.datetime.group_key

    @property
    def datetime_country_holidays(self) -> str | None:
        return self.datetime.country_holidays

    # Binning flat compat
    @property
    def binning_strategies(self) -> list[str] | None:
        return self.binning.strategies

    @property
    def binning_auto_supervised(self) -> bool:
        return self.binning.auto_supervised

    @property
    def binning_min_bin_fraction(self) -> float:
        return self.binning.min_bin_fraction

    # Interaction flat compat
    @property
    def scorer(self) -> str:
        return "mi"

    @property
    def n_splits(self) -> int:
        return self.interaction.n_splits

    @property
    def min_selected_per_fold(self) -> int:
        return self.interaction.min_selected_per_fold

    @property
    def importance_agg(self) -> str:
        return self.interaction.importance_agg

    @property
    def max_pairs_screen(self) -> int:
        return self.interaction.max_pairs_screen

    @property
    def interaction_prescreen_topk(self) -> int:
        return self.interaction.prescreen_topk

    @property
    def interaction_redundancy_corr(self) -> float:
        return self.interaction.redundancy_corr

    @property
    def interaction_prune_redundancy(self) -> bool:
        return self.interaction.prune_redundancy

    @property
    def interaction_stability_selection(self) -> bool:
        return self.interaction.stability_selection

    @property
    def interaction_stability_min_freq(self) -> float:
        return self.interaction.stability_min_freq

    @property
    def interaction_fast_mode(self) -> bool:
        return self.interaction.fast_mode

    @property
    def interaction_row_subsample(self) -> int:
        return self.interaction.row_subsample

    @property
    def interaction_n_jobs(self) -> int:
        return self.interaction.n_jobs

    @property
    def interaction_prescreen_spearman(self) -> bool:
        return self.interaction.prescreen_spearman

    @property
    def interaction_winsor_p(self) -> float:
        return self.interaction.winsor_p

    @property
    def pair_corr_with_y(self) -> bool:
        return self.interaction.pair_corr_with_y

    @property
    def pair_max_per_feature(self) -> int:
        return self.interaction.pair_max_per_feature

    @property
    def corr_avoid_redundancy(self) -> float:
        return self.interaction.corr_avoid_redundancy

    @property
    def interaction_exclude_generated_sources(self) -> bool:
        return self.interaction.exclude_generated_sources

    # Autoencoder flat compat
    @property
    def ae_latent_dim(self) -> int:
        return self.autoencoder.latent_dim

    @property
    def ae_encoder_arch(self) -> str:
        return ",".join(map(str, self.autoencoder.encoder_arch))

    @property
    def ae_decoder_arch(self) -> str:
        return ",".join(map(str, self.autoencoder.decoder_arch))

    @property
    def ae_activation(self) -> str:
        return self.autoencoder.activation

    @property
    def ae_dropout(self) -> float:
        return self.autoencoder.dropout

    @property
    def ae_learning_rate(self) -> float:
        return self.autoencoder.learning_rate

    @property
    def ae_batch_size(self) -> int:
        return self.autoencoder.batch_size

    @property
    def ae_epochs(self) -> int:
        return self.autoencoder.epochs

    @property
    def ae_patience(self) -> int:
        return self.autoencoder.patience

    @property
    def ae_max_features(self) -> int:
        return self.autoencoder.max_features

    @property
    def ae_min_features(self) -> int:
        return self.autoencoder.min_features

    @property
    def math_target_aware(self) -> bool:
        return self.math.target_aware

    @property
    def math_target_weight(self) -> float:
        return self.math.target_weight

    @property
    def math_max_transforms_per_feature(self) -> int:
        return self.math.max_transforms_per_feature

    @property
    def rff_n_components(self) -> int:
        return self.rff.n_components

    @property
    def rff_gamma(self) -> float | str:
        return self.rff.gamma

    @property
    def rff_kernel(self) -> str:
        return self.rff.kernel

    @property
    def rff_max_features(self) -> int:
        return self.rff.max_features

    @property
    def rff_handle_missing_features(self) -> str:
        return self.rff.handle_missing_features

    @property
    def clustering_strategies(self) -> list[str] | None:
        return self.clustering.strategies

    @property
    def clustering_max_features(self) -> int:
        return self.clustering.max_features

    @property
    def fourier_max_source_features(self) -> int:
        return self.fourier.max_source_features

    @property
    def fourier_exclude_generated_sources(self) -> bool:
        return self.fourier.exclude_generated_sources

    @property
    def autoencoder_epochs(self) -> int:
        return self.autoencoder.epochs

    @property
    def autoencoder_batch_size(self) -> int:
        return self.autoencoder.batch_size

    @property
    def autoencoder_lr(self) -> float:
        return self.autoencoder.learning_rate

    @property
    def autoencoder_device(self) -> str:
        return self.autoencoder.device

    @property
    def autoencoder_latent_ratio(self) -> float:
        return 0.25  # legacy
