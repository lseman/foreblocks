"""foreblocks.ts_handler.

Time-series preprocessing, filtering, imputation, and outlier removal.

Provides utilities for cleaning and transforming multivariate time series:
filtering (Savitzky-Golay, Kalman, lowess, Wiener), missing value
imputation (SAITS), outlier detection/removal, and EWT-based decomposition.

Core API:
- TimeSeriesHandler: unified preprocessing pipeline for time-series data

"""

from importlib import import_module

__all__ = [
    "TimeSeriesHandler",
]


def __getattr__(name):
    if name == "TimeSeriesHandler":
        module = import_module(".preprocessing", __name__)
        return module.TimeSeriesHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
