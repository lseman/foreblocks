from typing import TYPE_CHECKING

from .history import TrainingHistory
from .losses import LossComputer
from .nas import NASHelper, plot_alpha_evolution


__all__ = [
    "LossComputer",
    "NASHelper",
    "Trainer",
    "TrainingHistory",
    "plot_alpha_evolution",
]

if TYPE_CHECKING:
    from .trainer import Trainer


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
