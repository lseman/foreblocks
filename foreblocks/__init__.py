from typing import TYPE_CHECKING

from .aux import ModelConfig, TimeSeriesDataset, TrainingConfig, create_dataloaders
from .blocks.enc_dec import (
    GRUDecoder,
    GRUEncoder,
    LSTMDecoder,
    LSTMEncoder,
)
from .core import ForecastingModel
from .core.att import AttentionLayer
from .evaluation import ModelEvaluator

# from .pipeline import TimeSeriesSeq2Seq
from .pre.preprocessing import TimeSeriesPreprocessor
from .tf.transformer import TransformerDecoder, TransformerEncoder

if TYPE_CHECKING:
    from .training import Trainer

# Stable top-level public API
__all__ = [
    "ForecastingModel",
    "Trainer",
    "ModelEvaluator",
    # "TimeSeriesSeq2Seq",
    "TimeSeriesPreprocessor",
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
    "AttentionLayer",
]


def __getattr__(name):
    if name == "Trainer":
        from .training import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
