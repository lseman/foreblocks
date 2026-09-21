from importlib import import_module


__all__ = [
    "FeatureEngineer",
    "AdaptiveMI",
    "AdaptiveMRMR",
    "DistanceCorrelation",
    "HSIC",
]


def __getattr__(name):
    lazy_exports = {
        "FeatureEngineer": (".fengineer.fengineer", "FeatureEngineer"),
        "AdaptiveMI": (".stats.adaptive_mi", "AdaptiveMI"),
        "AdaptiveMRMR": (".fengineer.selectors.adaptive_mrmr", "AdaptiveMRMR"),
        "DistanceCorrelation": (".stats.distance_correlation", "DistanceCorrelation"),
        "HSIC": (".stats.hsic", "HSIC"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
