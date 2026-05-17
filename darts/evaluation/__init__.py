"""
Evaluation sub-package: metrics computation, visualization, and analysis.
"""

from importlib import import_module


__all__ = [
    "StreamlinedDARTSAnalyzer",
    "compute_metrics",
    "compute_final_metrics",
    "evaluate_on_loader",
    "plot_training_curve",
    "plot_alpha_evolution",
    "plot_prediction",
    "batched_forecast",
]


def __getattr__(name):
    lazy_exports = {
        "StreamlinedDARTSAnalyzer": (".analyzer", "StreamlinedDARTSAnalyzer"),
        "compute_metrics": (".metrics", "compute_metrics"),
        "compute_final_metrics": (".metrics", "compute_final_metrics"),
        "evaluate_on_loader": (".metrics", "evaluate_on_loader"),
        "plot_training_curve": (".plotting", "plot_training_curve"),
        "plot_alpha_evolution": (".plotting", "plot_alpha_evolution"),
        "plot_prediction": (".plotting", "plot_prediction"),
        "batched_forecast": (".plotting", "batched_forecast"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
