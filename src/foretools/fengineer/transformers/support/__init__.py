"""Shared infrastructure: base class, config, strategies, safe stats."""

from .base import (
    BaseFeatureTransformer,
    cached_fit,
    require_fitted,
)
from .config import (
    AutoencoderConfig,
    BinningConfig,
    CategoricalConfig,
    ClusteringConfig,
    DateTimeConfig,
    FeatureConfig,
    FourierConfig,
    InteractionConfig,
    MathConfig,
    RFFConfig,
    SelectorConfig,
)
from .stats_safe import (
    safe_skew,
    safe_kurtosis,
    safe_row_skew,
    safe_row_kurtosis,
)
from .binning_strategies import (
    fd_edges,
    doane_edges,
    shimazaki_edges,
    kmeans_edges,
    quantile_transformer,
    uniform_transformer,
    mdlp_edges,
    woe_edges_and_map,
    _finite,
    _clean_edges,
    _digitize_edges,
    _enforce_min_support_edges,
)

__all__ = [
    # Base
    "BaseFeatureTransformer",
    "cached_fit",
    "require_fitted",
    # Config
    "FeatureConfig",
    "BinningConfig",
    "CategoricalConfig",
    "ClusteringConfig",
    "DateTimeConfig",
    "FourierConfig",
    "InteractionConfig",
    "MathConfig",
    "RFFConfig",
    "SelectorConfig",
    "AutoencoderConfig",
    # Stats safe
    "safe_skew",
    "safe_kurtosis",
    "safe_row_skew",
    "safe_row_kurtosis",
    # Binning strategies
    "fd_edges",
    "doane_edges",
    "shimazaki_edges",
    "kmeans_edges",
    "quantile_transformer",
    "uniform_transformer",
    "mdlp_edges",
    "woe_edges_and_map",
    "_finite",
    "_clean_edges",
    "_digitize_edges",
    "_enforce_min_support_edges",
]
