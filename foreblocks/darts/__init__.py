"""
ForeBlocks DARTS: Neural Architecture Search for Time Series Forecasting
"""

# Neural Building Blocks
from .architecture import DARTSCell, TimeSeriesDARTS

# Configuration dataclasses
from .config import (
    AblationSearchConfig,
    DARTSConfig,
    DARTSSearchSpaceConfig,
    DARTSTrainConfig,
    FinalTrainConfig,
    MultiFildelitySearchConfig,
    RobustPoolSearchConfig,
)

# Evaluation helpers (re-exported for convenience)
from .evaluation import (
    StreamlinedDARTSAnalyzer,
    batched_forecast,
    compute_final_metrics,
    compute_metrics,
    evaluate_on_loader,
    plot_alpha_evolution,
    plot_prediction,
    plot_training_curve,
)

# Trainer (thin orchestrator — delegates to focused sub-modules)
from .trainer import DARTSTrainer

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
