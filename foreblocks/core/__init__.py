"""Core foreblocks abstractions and top-level model utilities.

This package exports the main core building blocks used by foreblocks
models, attention layers, and conformal prediction helpers.
"""

from foreblocks.core.att import AttentionLayer
from foreblocks.core.extend import DistilledForecastingModel
from foreblocks.core.model import BaseHead, ForecastingModel
from foreblocks.core.training.conformal import ConformalPredictionEngine
from foreblocks.modules.heads import HeadComposer, HeadSpec


__all__ = [
    "AttentionLayer",
    "ConformalPredictionEngine",
    "ForecastingModel",
    "DistilledForecastingModel",
    "BaseHead",
    "HeadComposer",
    "HeadSpec",
]
