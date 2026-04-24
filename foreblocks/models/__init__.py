"""Model-level composition APIs."""

from .forecasting import BaseHead, ForecastingModel
from .graph_forecasting import GraphForecastingModel


__all__ = [
    "BaseHead",
    "ForecastingModel",
    "GraphForecastingModel",
]
