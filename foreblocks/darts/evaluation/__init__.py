"""
Evaluation sub-package: metrics computation, visualization, and analysis.
"""

from .analyzer import StreamlinedDARTSAnalyzer
from .metrics import compute_final_metrics, compute_metrics, evaluate_on_loader
from .plotting import (
    batched_forecast,
    plot_alpha_evolution,
    plot_prediction,
    plot_training_curve,
)

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
