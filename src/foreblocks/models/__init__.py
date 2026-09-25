"""Complete forecasting and anomaly models, composed from foreblocks.nn."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.models.forecasting import ForecastingModel as ForecastingModel
    from foreblocks.models.graph import GraphForecastingModel as GraphForecastingModel
    from foreblocks.models.forecasting import BaseHead as BaseHead
    from foreblocks.models.config import ModelConfig as ModelConfig
    from foreblocks.models.discharge import DischargeClassifier as DischargeClassifier
    from foreblocks.models.discharge import (
        DischargeClassifierConfig as DischargeClassifierConfig,
    )
    from foreblocks.models.discharge import DischargeResult as DischargeResult

_EXPORTS = {
    'ForecastingModel': 'foreblocks.models.forecasting',
    'GraphForecastingModel': 'foreblocks.models.graph',
    'BaseHead': 'foreblocks.models.forecasting',
    'ModelConfig': 'foreblocks.models.config',
    'DischargeClassifier': 'foreblocks.models.discharge',
    'DischargeClassifierConfig': 'foreblocks.models.discharge',
    'DischargeResult': 'foreblocks.models.discharge',
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
