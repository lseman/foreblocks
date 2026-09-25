"""Feature engineering pipeline for tabular data.

This package provides a comprehensive, sklearn-compatible feature engineering
framework with:

**Transformers** — automated creation of new features from existing columns::

    - DateTimeTransformer  : cyclical, flag, and elapsed-time features
    - CategoricalTransformer  : target encoding, WOE, ordinal, one-hot
    - BinningTransformer  : quantile, k-means, and supervised binning
    - MathematicalTransformer  : variance-driven transform selection (Yeo-Johnson, Box-Cox)
    - InteractionTransformer  : pairwise interactions with redundancy pruning
    - FourierTransformer  : periodic pattern extraction
    - RandomFourierFeaturesTransformer  : RBF kernel approximation
    - ClusteringTransformer  : k-means / GMM cluster-distance features
    - StatisticalTransformer  : row-wise statistics (mean, std, entropy)
    - AutoencoderTransformer  : non-linear dimensionality via PyTorch

**Feature Selection** — multi-stage pipelines that try several selectors::

    - PipelineSelector (aka FeatureSelector)  : MI → mRMR → RFECV → Boruta
    - MISelector  : mutual-information screening with stability selection
    - MRMRSelector  : minimum-redundancy-maximum-relevance
    - BorutaSelector  : all-relevant feature selection via shadow features
    - AdvancedRFECV  : recursive elimination with CV and ensemble voting

**Filters** — correlation-based redundancy removal::

    - CorrelationFilter  : removes highly correlated pairs (Pearson or AdaptiveMI)

**Configuration** — nested dataclass hierarchy for clean parameter access::

    >>> from foretools.fengineer import FeatureConfig
    >>> cfg = FeatureConfig(backend="tree", create_interactions=True)
    >>> cfg.interaction.max_pairs_screen = 500   # nested access
    >>> cfg.n_bins = 15                            # flat compat property

Quick start
-----------
>>> from foretools.fengineer import FeatureEngineer, FeatureConfig
>>> config = FeatureConfig(backend="auto", create_interactions=True)
>>> fe = FeatureEngineer(config)
>>> X_transformed = fe.fit_transform(X_train, y_train)
>>> X_test = fe.transform(X_test)  # uses fitted state

"""

from __future__ import annotations

# Main pipeline
from .fengineer import FeatureEngineer

# Configuration
from .transformers.support.config import (
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

# Transformers (selected key ones for the public API)
from .transformers import (
    AutoencoderTransformer,
    BinningTransformer,
    CategoricalTransformer,
    ClusteringTransformer,
    DateTimeTransformer,
    FourierTransformer,
    InteractionTransformer,
    MathematicalTransformer,
    MDLPTransformer,
    PolynomialTransformer,
    RandomFourierFeaturesTransformer,
    StatisticalTransformer,
    WeightOfEvidenceTransformer,
)

# Feature selection
from .selectors import (
    AdvancedRFECV,
    BorutaSelector,
    FeatureSelector,
    MISelector,
    MRMRSelector,
    PipelineSelector,
    RedundancyPruner,
    RFECVConfig,
)

# Filters
from .filters.correlation import CorrelationFilter

__all__ = [
    # Configuration
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
    # Main pipeline
    "FeatureEngineer",
    # Transformers
    "DateTimeTransformer",
    "CategoricalTransformer",
    "BinningTransformer",
    "MathematicalTransformer",
    "InteractionTransformer",
    "PolynomialTransformer",
    "FourierTransformer",
    "RandomFourierFeaturesTransformer",
    "ClusteringTransformer",
    "StatisticalTransformer",
    "WeightOfEvidenceTransformer",
    "MDLPTransformer",
    "AutoencoderTransformer",
    # Feature selection
    "FeatureSelector",
    "PipelineSelector",
    "MISelector",
    "MRMRSelector",
    "BorutaSelector",
    "AdvancedRFECV",
    "RFECVConfig",
    "RedundancyPruner",
    # Filters
    "CorrelationFilter",
]
