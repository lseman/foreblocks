"""Feature engineering transformers.

Structure
---------
Each transformer lives in its own file under this package:
  - ``autoencoder.py`` — Non-linear embeddings via PyTorch
  - ``binning.py`` — Adaptive binning with multiple strategies
  - ``categorical.py`` — Categorical encoding strategies
  - ``clustering.py`` — Clustering/embedding features
  - ``datetime.py`` — Datetime feature extraction
  - ``fourier.py`` — Periodic pattern features
  - ``interaction.py`` — Interaction & polynomial generation
  - ``mathematical.py`` — Variance-driven transform selection
  - ``mdlp.py`` — MDLP supervised binning
  - ``polynomial.py`` — Standalone polynomial feature generator
  - ``rff.py`` — Random Fourier Features (RBF kernel approx)
  - ``statistical.py`` — Row-wise statistical features
  - ``woe.py`` — Weight of Evidence encoding (binary classification)

Infrastructure (in ``aux/``):
  - ``aux/base.py`` — ABC base class, shared utilities, decorators
  - ``aux/config.py`` — FeatureConfig with nested sub-configs
  - ``aux/stats_safe.py`` — Safe statistical functions (skew, kurtosis)
  - ``aux/binning_strategies.py`` — Standalone binning strategy functions

Usage
-----
>>> from foretools.fengineer.transformers import FeatureConfig
>>> from foretools.fengineer.transformers import (
...     StatisticalTransformer,
...     MathematicalTransformer,
...     CategoricalTransformer,
...     BinningTransformer,
...     DateTimeTransformer,
...     InteractionTransformer,
...     PolynomialTransformer,
...     RandomFourierFeaturesTransformer,
...     FourierTransformer,
...     ClusteringTransformer,
...     WeightOfEvidenceTransformer,
...     MDLPTransformer,
...     AutoencoderTransformer,
... )
>>> cfg = FeatureConfig()
"""

from .aux import (
    BaseFeatureTransformer,
    cached_fit,
    require_fitted,
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
from .autoencoder import AutoencoderTransformer
from .binning import BinningTransformer
from .categorical import CategoricalTransformer
from .clustering import ClusteringTransformer
from .datetime import DateTimeTransformer
from .fourier import FourierTransformer
from .interaction import InteractionTransformer
from .mathematical import MathematicalTransformer
from .mdlp import MDLPTransformer
from .polynomial import PolynomialTransformer
from .rff import RandomFourierFeaturesTransformer
from .statistical import StatisticalTransformer
from .woe import WeightOfEvidenceTransformer

__all__ = [
    # Configs
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
    # Base
    "BaseFeatureTransformer",
    "require_fitted",
    "cached_fit",
    # Transformers
    "StatisticalTransformer",
    "MathematicalTransformer",
    "CategoricalTransformer",
    "BinningTransformer",
    "DateTimeTransformer",
    "InteractionTransformer",
    "PolynomialTransformer",
    "RandomFourierFeaturesTransformer",
    "FourierTransformer",
    "ClusteringTransformer",
    "WeightOfEvidenceTransformer",
    "MDLPTransformer",
    "AutoencoderTransformer",
]
