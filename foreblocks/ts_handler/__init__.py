from importlib import import_module

__all__ = [
    "TimeSeriesHandler",
]


def __getattr__(name):
    if name == "TimeSeriesHandler":
        module = import_module(".preprocessing", __name__)
        return module.TimeSeriesHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
