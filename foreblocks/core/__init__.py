"""Core foreblocks abstractions and top-level model utilities.

This package exports the main core building blocks used by foreblocks
models, attention layers, and conformal prediction helpers.
"""

from foreblocks.training.conformal import ConformalPredictionEngine

from .att import AttentionLayer
from .extend import DistilledForecastingModel
from .heads import HeadComposer, HeadSpec
from .model import BaseHead, ForecastingModel

__all__ = [
    "AttentionLayer",
    "ConformalPredictionEngine",
    "ForecastingModel",
    "DistilledForecastingModel",
    "BaseHead",
    "HeadComposer",
    "HeadSpec",
]
