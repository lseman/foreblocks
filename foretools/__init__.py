from importlib import import_module

__all__ = ["FeatureEngineer", "AdaptiveMI", "DistanceCorrelation", "HSIC"]


def __getattr__(name):
    lazy_exports = {
        "FeatureEngineer": (".fengineer.fengineer", "FeatureEngineer"),
        "AdaptiveMI": (".aux.adaptive_mi", "AdaptiveMI"),
        "DistanceCorrelation": (".aux.distance_correlation", "DistanceCorrelation"),
        "HSIC": (".aux.hsic", "HSIC"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
