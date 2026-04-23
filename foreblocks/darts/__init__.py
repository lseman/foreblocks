"""
ForeBlocks DARTS: Neural Architecture Search for Time Series Forecasting
"""

from importlib import import_module


__version__ = "1.0.0"
__author__ = "ForeBlocks Team"

__all__ = [
    # Core model components
    "TimeSeriesDARTS",
    "DARTSCell",
    # Trainer
    "DARTSTrainer",
    # Config
    "DARTSConfig",
    "DARTSSearchSpaceConfig",
    "DARTSTrainConfig",
    "FinalTrainConfig",
    "MultiFildelitySearchConfig",
    "AblationSearchConfig",
    "RobustPoolSearchConfig",
    # Evaluation
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
        "TimeSeriesDARTS": (".architecture", "TimeSeriesDARTS"),
        "DARTSCell": (".architecture", "DARTSCell"),
        "DARTSTrainer": (".trainer", "DARTSTrainer"),
        "DARTSConfig": (".config", "DARTSConfig"),
        "DARTSSearchSpaceConfig": (".config", "DARTSSearchSpaceConfig"),
        "DARTSTrainConfig": (".config", "DARTSTrainConfig"),
        "FinalTrainConfig": (".config", "FinalTrainConfig"),
        "MultiFildelitySearchConfig": (".config", "MultiFildelitySearchConfig"),
        "AblationSearchConfig": (".config", "AblationSearchConfig"),
        "RobustPoolSearchConfig": (".config", "RobustPoolSearchConfig"),
        "StreamlinedDARTSAnalyzer": (".evaluation", "StreamlinedDARTSAnalyzer"),
        "compute_metrics": (".evaluation", "compute_metrics"),
        "compute_final_metrics": (".evaluation", "compute_final_metrics"),
        "evaluate_on_loader": (".evaluation", "evaluate_on_loader"),
        "plot_training_curve": (".evaluation", "plot_training_curve"),
        "plot_alpha_evolution": (".evaluation", "plot_alpha_evolution"),
        "plot_prediction": (".evaluation", "plot_prediction"),
        "batched_forecast": (".evaluation", "batched_forecast"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
