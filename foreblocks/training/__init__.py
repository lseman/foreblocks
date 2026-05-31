from typing import TYPE_CHECKING

from .batch_io import (
    loader_len,
    move_batch_to_device,
    to_device,
    unpack_batch,
)
from .history import TrainingHistory
from .losses import LossComputer
from .nas import NASHelper, plot_alpha_evolution
from .training_loop import (
    backward_step,
    evaluate,
    forward_pass,
    train_epoch,
)
from .conformal_trainer import (
    calibrate_conformal,
    compute_coverage,
    predict_with_intervals,
    update_conformal,
)


__all__ = [
    # Core
    "LossComputer",
    "NASHelper",
    "Trainer",
    "TrainingHistory",
    "plot_alpha_evolution",
    # Batch I/O
    "loader_len",
    "move_batch_to_device",
    "to_device",
    "unpack_batch",
    # Training loop
    "backward_step",
    "evaluate",
    "forward_pass",
    "train_epoch",
    # Conformal
    "calibrate_conformal",
    "compute_coverage",
    "predict_with_intervals",
    "update_conformal",
]

if TYPE_CHECKING:
    from .trainer import Trainer


def __getattr__(name):
    if name == "Trainer":
        from .trainer import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
