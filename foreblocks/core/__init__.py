"""foreblocks.core.

Foundational model abstractions and inference utilities.

Core exports the base ForecastingModel class (wraps backbone + head),
attention layer abstractions, and specialized inference modes (distillation,
conformal prediction). These are the primary interfaces for end-to-end
forecasting—most foreblocks applications start by composing a ForecastingModel
and a configuration.

Core API:
- ForecastingModel: base forecasting architecture (encoder-decoder backbone + heads)
- DistilledForecastingModel: model distillation for deployment
- AttentionLayer: attention interface and implementations
- ConformalPredictionEngine: uncertainty quantification via conformal prediction
- BaseHead, HeadComposer: output head composition
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
