"""foreblocks.models.forecasting.

End-to-end forecasting model factory and composition.

ForecastingModel combines a backbone (Mamba, Transformer, GraphNN) with output
heads to produce point forecasts, quantiles, or uncertainty estimates. This is
the primary user-facing API for training and inference—initialize a ForecastingModel
with a backbone type and configuration, then call .fit() and .predict().

Core API:
- ForecastingModel: head-based modular forecasting model for time series tasks
- BaseHead: base class for all heads in the forecasting model

"""

from foreblocks.core.model import BaseHead, ForecastingModel

__all__ = [
    "BaseHead",
    "ForecastingModel",
]
