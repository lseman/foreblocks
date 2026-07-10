"""Feature selection module.

Exports
-------
- ``FeatureSelector`` (alias for :class:`PipelineSelector`)
- ``PipelineSelector`` — multi-stage orchestrator
- ``MISelector`` — mutual-information-based selection
- ``MRMRSelector`` — mRMR selection
- ``BorutaSelector`` — Boruta all-relevant selection
- ``AdvancedRFECV`` — RFECV with ensemble voting
- ``RFECVConfig`` — dataclass for RFECV parameters
- ``FeatureSelectorABC`` — abstract base class
- ``RedundancyPruner`` — correlation-based pruning utility
"""

from .base import FeatureSelectorABC
from .boruta import BorutaSelector
from .feature_selector import (
    FeatureSelector,
    PipelineSelector,
)
from .mi_selector import MISelector
from .mrmr_selector import MRMRSelector
from .rfecv import AdvancedRFECV, RFECVConfig
from .redundancy import RedundancyPruner

__all__ = [
    "FeatureSelector",
    "PipelineSelector",
    "MISelector",
    "MRMRSelector",
    "BorutaSelector",
    "AdvancedRFECV",
    "RFECVConfig",
    "FeatureSelectorABC",
    "RedundancyPruner",
]
