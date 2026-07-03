"""foreblocks.core.training.

Unified training orchestration with NAS, conformal prediction, and checkpointing.

Provides the ``Trainer`` class — the main entry point for full training loops with
optional validation, early stopping, and MLTracker logging. Also exposes loss
computation, batch I/O, conformal calibration, and training-loop primitives.

Core API:
- Trainer: unified training loop with NAS, conformal prediction, and MoE logging
- LossComputer: multi-component loss computation (task, distillation, regularization)
- NASHelper: architecture parameter detection and alpha optimization management
- TrainingHistory: epoch-level metric tracking and serialization
- train_epoch, evaluate, forward_pass, backward_step: training-loop primitives

"""

from typing import TYPE_CHECKING

from foreblocks.core.training.batch_io import (
    loader_len,
    move_batch_to_device,
    to_device,
    unpack_batch,
)
from foreblocks.core.training.conformal_trainer import (
    calibrate_conformal,
    compute_coverage,
    predict_with_intervals,
    update_conformal,
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
