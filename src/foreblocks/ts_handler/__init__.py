"""foreblocks.ts_handler.

Time-series preprocessing, filtering, imputation, and outlier removal.

Provides utilities for cleaning and transforming multivariate time series:
core preprocessing (TimeSeriesHandler, imputation, outlier removal, windowing),
transforms (EWT, time features, log/scaling), and filtering.

Subpackages:
- core: preprocessing, imputation, outlier removal, windowing
- transforms: log/scaling transforms, EWT, time features
- tools: diagnostics, plotting, auto-configuration, pipeline
- auto_filter: automatic filter selection and tuning
- filters: filtering algorithms (Savitzky-Golay, Kalman, lowess, Wiener)

Core API:
- TimeSeriesHandler: unified preprocessing pipeline for time-series data

"""

from importlib import import_module

__all__ = [
    "TimeSeriesHandler",
]


def __getattr__(name):
    if name == "TimeSeriesHandler":
        module = import_module(".core.preprocessing", __name__)
        return module.TimeSeriesHandler
    # Back-compat: from foreblocks.ts_handler import preprocessing
    if name == "preprocessing":
        module = import_module(".core.preprocessing", __name__)
        return module
    # Back-compat: from foreblocks.ts_handler import outlier
    if name == "outlier":
        module = import_module(".core.outlier", __name__)
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
