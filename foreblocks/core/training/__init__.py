from typing import TYPE_CHECKING

from foreblocks.core.training.batch_io import (
    loader_len,
    move_batch_to_device,
    to_device,
    unpack_batch,
)
from foreblocks.core.training.history import TrainingHistory
from foreblocks.core.training.losses import LossComputer
from foreblocks.core.training.nas import NASHelper, plot_alpha_evolution
from foreblocks.core.training.training_loop import (
    backward_step,
    evaluate,
    forward_pass,
    train_epoch,
)
from foreblocks.core.training.conformal_trainer import (
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
    from foreblocks.core.training.trainer import Trainer


def __getattr__(name):
    if name == "Trainer":
        from foreblocks.core.training.trainer import Trainer as _Trainer

        return _Trainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
