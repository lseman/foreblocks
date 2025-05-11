# Import main components to make them available at the package level
from .core import ForecastingModel
from .enc_dec import (
    LSTMEncoder,
    LSTMDecoder,
    GRUEncoder,
    GRUDecoder,
    VariationalEncoderWrapper,
    LatentConditionedDecoder,
)
from .transformer import TransformerEncoder, TransformerDecoder
from .att import AttentionLayer
from .preprocessing import TimeSeriesPreprocessor
from .utils import TimeSeriesDataset, create_dataloaders, Trainer
from .pipeline import TimeSeriesSeq2Seq
from .aux import ModelConfig, TrainingConfig

# Define package metadata
__version__ = "0.1.0"
__author__ = "Laio O. Seman"
__email__ = "laioseman@gmail.com"

# Define what gets imported with "from foreblocks import *"
__all__ = [
    "ForecastingModel",
    "LSTMEncoder",
    "LSTMDecoder",
    "GRUEncoder",
    "GRUDecoder",
    "TransformerEncoder",
    "TransformerDecoder",
    "VariationalEncoderWrapper",
    "LatentConditionedDecoder",
    "AttentionLayer",
    "TimeSeriesPreprocessor",
    "TimeSeriesDataset",
    "Trainer",
    "create_dataloaders",
    "TimeSeriesSeq2Seq",
    "ModelConfig",
    "TrainingConfig",
    "blocks",
]
