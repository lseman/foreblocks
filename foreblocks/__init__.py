from importlib import import_module
from typing import TYPE_CHECKING

from .blocks.enc_dec import GRUDecoder
from .blocks.enc_dec import GRUEncoder
from .blocks.enc_dec import LSTMDecoder
from .blocks.enc_dec import LSTMEncoder
from .config import ModelConfig
from .config import TrainingConfig
from .core.att import AttentionLayer
from .data import TimeSeriesDataset
from .data import create_dataloaders
from .models import ForecastingModel
from .models import GraphForecastingModel
from .tf.transformer import TransformerDecoder
from .tf.transformer import TransformerEncoder

# from .pipeline import TimeSeriesSeq2Seq
from .tf.transformer_tuner import ModernTransformerTuner
from .tf.transformer_tuner import TransformerTuner


if TYPE_CHECKING:
    from .evaluation import ModelEvaluator
    from .training import Trainer
    from .ts_handler import TimeSeriesHandler

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
        "ModelEvaluator": (".evaluation", "ModelEvaluator"),
        "TimeSeriesHandler": (".ts_handler", "TimeSeriesHandler"),
        "Trainer": (".training", "Trainer"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
