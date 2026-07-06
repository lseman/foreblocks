"""foreblocks.

Top-level lazy-loading entry point for the Foreblocks time series forecasting framework.

Exposes the main public API — forecasting models, trainers, evaluators, data
handling, and model configuration — via lazy imports so the full modeling stack
is not loaded at package import time.

Core API:
- ForecastingModel: base forecasting model class
- GraphForecastingModel: graph-based time series forecasting
- Trainer: unified training loop with NAS, conformal prediction, and MoE logging
- ModelEvaluator: evaluation and metric computation
- TimeSeriesDataset: time series dataset wrapper
- create_dataloaders: DataLoader factory for training/evaluation
- TimeSeriesHandler: time series data handler
- ModelConfig, TrainingConfig: configuration dataclasses

"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.config import ModelConfig, TrainingConfig
    from foreblocks.core.att import AttentionLayer
    from foreblocks.core.evaluation import ModelEvaluator
    from foreblocks.core.training import Trainer
    from foreblocks.data import TimeSeriesDataset, create_dataloaders
    from foreblocks.models import ForecastingModel, GraphForecastingModel
    from foreblocks.models.transformer.transformer import (
        TransformerDecoder,
        TransformerEncoder,
    )
    from foreblocks.models.transformer.transformer_tuner import (
        ModernTransformerTuner,
        TransformerTuner,
    )
    from foreblocks.modules.blocks.enc_dec import (
        GRUDecoder,
        GRUEncoder,
        LSTMDecoder,
        LSTMEncoder,
    )
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
        "AttentionLayer": (".core.att", "AttentionLayer"),
        "ForecastingModel": (".models", "ForecastingModel"),
        "GraphForecastingModel": (".models", "GraphForecastingModel"),
        "GRUDecoder": (".modules.blocks.enc_dec", "GRUDecoder"),
        "GRUEncoder": (".modules.blocks.enc_dec", "GRUEncoder"),
        "LSTMDecoder": (".modules.blocks.enc_dec", "LSTMDecoder"),
        "LSTMEncoder": (".modules.blocks.enc_dec", "LSTMEncoder"),
        "ModernTransformerTuner": (
            ".models.transformer.transformer_tuner",
            "ModernTransformerTuner",
        ),
        "ModelConfig": (".config", "ModelConfig"),
        "ModelEvaluator": (".core.evaluation", "ModelEvaluator"),
        "TimeSeriesDataset": (".data", "TimeSeriesDataset"),
        "Trainer": (".core.training", "Trainer"),
        "TrainingConfig": (".config", "TrainingConfig"),
        "TransformerDecoder": (".models.transformer.transformer", "TransformerDecoder"),
        "TransformerEncoder": (".models.transformer.transformer", "TransformerEncoder"),
        "TransformerTuner": (
            ".models.transformer.transformer_tuner",
            "TransformerTuner",
        ),
        "TimeSeriesHandler": (".ts_handler", "TimeSeriesHandler"),
        "create_dataloaders": (".data", "create_dataloaders"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
