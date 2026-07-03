"""foreblocks.models.

High-level model factories and specialized architectures.

Exports ForecastingModel (point forecasts), GraphForecastingModel (spatiotemporal
graphs), and related composition utilities. Import from here when building
end-to-end applications; import from foreblocks.core for lower-level
abstractions and subclassing.

Core API:
- ForecastingModel: point forecasting model factory
- GraphForecastingModel: spatiotemporal graph forecasting model
- BaseHead: base class for forecasting heads

"""

from foreblocks.models.forecasting import BaseHead, ForecastingModel
from foreblocks.models.graph_forecasting import GraphForecastingModel

__all__ = [
    "BaseHead",
    "ForecastingModel",
    "GraphForecastingModel",
]
