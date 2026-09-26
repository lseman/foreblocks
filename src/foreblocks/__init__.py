"""Public Foreblocks models, training, data, and neural building blocks; loaded lazily."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.models.forecasting import ForecastingModel as ForecastingModel
    from foreblocks.models.graph import GraphForecastingModel as GraphForecastingModel
    from foreblocks.training.trainer import Trainer as Trainer
    from foreblocks.evaluation.model_evaluator import ModelEvaluator as ModelEvaluator
    from foreblocks.data import TimeSeriesDataset as TimeSeriesDataset
    from foreblocks.data import create_dataloaders as create_dataloaders
    from foreblocks.processing import TimeSeriesHandler as TimeSeriesHandler
    from foreblocks.models.config import ModelConfig as ModelConfig
    from foreblocks.training.config import TrainingConfig as TrainingConfig
    from foreblocks.nn.blocks.recurrent import LSTMEncoder as LSTMEncoder
    from foreblocks.nn.blocks.recurrent import LSTMDecoder as LSTMDecoder
    from foreblocks.nn.blocks.recurrent import GRUEncoder as GRUEncoder
    from foreblocks.nn.blocks.recurrent import GRUDecoder as GRUDecoder
    from foreblocks.nn.transformer.encoder import TransformerEncoder as TransformerEncoder
    from foreblocks.nn.transformer.decoder import TransformerDecoder as TransformerDecoder
    from foreblocks.tuning.transformer import TransformerTuner as TransformerTuner
    from foreblocks.nn.attention.layer import AttentionLayer as AttentionLayer

_EXPORTS = {
    'ForecastingModel': 'foreblocks.models.forecasting',
    'GraphForecastingModel': 'foreblocks.models.graph',
    'Trainer': 'foreblocks.training.trainer',
    'ModelEvaluator': 'foreblocks.evaluation.model_evaluator',
    'TimeSeriesDataset': 'foreblocks.data',
    'create_dataloaders': 'foreblocks.data',
    'TimeSeriesHandler': 'foreblocks.processing',
    'ModelConfig': 'foreblocks.models.config',
    'TrainingConfig': 'foreblocks.training.config',
    'LSTMEncoder': 'foreblocks.nn.blocks.recurrent',
    'LSTMDecoder': 'foreblocks.nn.blocks.recurrent',
    'GRUEncoder': 'foreblocks.nn.blocks.recurrent',
    'GRUDecoder': 'foreblocks.nn.blocks.recurrent',
    'TransformerEncoder': 'foreblocks.nn.transformer.encoder',
    'TransformerDecoder': 'foreblocks.nn.transformer.decoder',
    'TransformerTuner': 'foreblocks.tuning.transformer',
    'AttentionLayer': 'foreblocks.nn.attention.layer',
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
