"""
DARTS bilevel training loop.

Extracted from ``DARTSTrainer`` so the logic lives in one focused module.
The public entry-point is :func:`train_darts_model`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler

from ..evaluation.metrics import compute_final_metrics
from ..training.helpers import (
    ArchitectureRegularizer,
    BilevelOptimizer,
    RegularizationType,
    TemperatureScheduler,
)
from ..utils.training import (
    autocast_ctx,
    build_arch_param_groups,
    capture_progressive_state,
    create_progress_bar,
    restore_progressive_state,
    split_arch_and_model_params,
    unpack_forecasting_batch,
)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def _dynamic_arch_update_freq(
    epoch: int, epochs: int, warmup_epochs: int, base_freq: int
) -> int:
    """Return the arch-update frequency adapted to the current training phase.

    Architecture gradients are noisiest while model weights are still
    unstable (early training), so updates should be *less* frequent then
    and *more* frequent in the late commitment phase.

    Phase           progress (post-warmup)   returned frequency
    Early                  < 40 %            ``base_freq + 2``
    Mid             40 – 70 %                ``base_freq``
    Late                   > 70 %            ``max(1, base_freq − 1)``
    """
    if epoch <= warmup_epochs:
        return base_freq
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    if progress < 0.40:
        return base_freq + 2
    if progress < 0.70:
        return base_freq
    return max(1, base_freq - 1)


def _dynamic_inner_arch_iters(epoch: int, epochs: int, warmup_epochs: int) -> int:
    """Return the number of arch inner-loop gradient steps for this epoch.

    More iterations in the late phase give the architecture more gradient
    signal once model weights have converged.

    Phase   progress    iterations
    Early    < 40 %        1
    Mid     40–70 %        2
    Late     > 70 %        3
    """
    if epoch <= warmup_epochs:
        return 1
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    if progress < 0.40:
        return 1
    if progress < 0.70:
        return 2
    return 3


