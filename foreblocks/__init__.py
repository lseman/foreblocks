"""Public foreblocks API exports and lazy load helpers.

This module exposes the main foreblocks entry points for forecasting,
training, evaluation, and model composition.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from foreblocks.modules.blocks.enc_dec import GRUDecoder, GRUEncoder, LSTMDecoder, LSTMEncoder
from foreblocks.config import ModelConfig, TrainingConfig
from foreblocks.core.att import AttentionLayer
from foreblocks.data import TimeSeriesDataset, create_dataloaders
from foreblocks.models import ForecastingModel, GraphForecastingModel
from foreblocks.models.transformer.transformer import TransformerDecoder, TransformerEncoder

# from .pipeline import TimeSeriesSeq2Seq
from foreblocks.models.transformer.transformer_tuner import ModernTransformerTuner, TransformerTuner

if TYPE_CHECKING:
    from foreblocks.core.evaluation import ModelEvaluator
    from foreblocks.core.training import Trainer
    from foreblocks.ts_handler import TimeSeriesHandler

# Stable top-level public API
__all__ = [
    "ForecastingModel",
    "GraphForecastingModel",
    "Trainer",
    "ModelEvaluator",
    # "TimeSeriesSeq2Seq",
    "TimeSeriesHandler",
    "TimeSeriesDataset",
    "create_dataloaders",
    "ModelConfig",
    "TrainingConfig",
    # expert/low-level exports (kept for compatibility)
    "LSTMEncoder",
    "LSTMDecoder",
    "GRUEncoder",
    "GRUDecoder",
    "TransformerEncoder",
    "TransformerDecoder",
    "ModernTransformerTuner",
    "TransformerTuner",
    "AttentionLayer",
]


def __getattr__(name):
    lazy_exports = {
        "ModelEvaluator": (".core.evaluation", "ModelEvaluator"),
        "TimeSeriesHandler": (".ts_handler", "TimeSeriesHandler"),
        "Trainer": (".core.training", "Trainer"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
