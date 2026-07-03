"""foreblocks.models.

High-level model factories and specialized architectures.

Models exports ForecastingModel (point forecasts), GraphForecastingModel
(spatiotemporal graphs), and related composition utilities. Import from
here when building end-to-end applications; import from foreblocks.core
for lower-level abstractions and subclassing.

from foreblocks.models.forecasting import BaseHead, ForecastingModel
from foreblocks.models.graph_forecasting import GraphForecastingModel


__all__ = [
    "BaseHead",
    "ForecastingModel",
    "GraphForecastingModel",
]
