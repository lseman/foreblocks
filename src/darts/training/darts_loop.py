"""
DARTS bilevel training loop.

Extracted from ``DARTSTrainer`` so the logic lives in one focused module.
The public entry-point is :func:`train_darts_model`.

The training loop now supports multiple DARTS variants via the
``engine`` parameter:

- ``DARTS``      – original bilevel optimisation (Gumbel-Softmax + Adam)
- ``GD_DARTS``   – straight-through softmax; no Gumbel noise
- ``R_DARTS``    – AdamW + gradient-norm balancing for arch params
- ``PC_DARTS``   – permutation-consistent weight sharing across edges
- ``BI_DARTS``   – bidirectional (forward + backward) cell training

See :class:`~darts.config.DARTSVariant` for full documentation.
"""

from __future__ import annotations

# Re-export all public functions from submodules
from .dynamic_scheduling import _dynamic_arch_update_freq, _dynamic_inner_arch_iters
from .edge_regularization import (
    _add_edge_diversity_reg,
    _add_edge_sharpening,
    _extract_edge_probs,
)
from .perturbation_hessian import (
    _apply_darts_pt_perturbation,
    _restore_model_params,
    compute_implicit_arch_gradient_correction,
    finite_difference_hessian_penalty,
)
from .training_loop import (
    _run_model_training_epoch,
    _run_validation_epoch,
    train_darts_model,
)
from .utils import _log_arch_gradients, _maybe_prune, _safe_load_state


__all__ = [
    "train_darts_model",
    "_run_model_training_epoch",
    "_run_validation_epoch",
    "_add_edge_diversity_reg",
    "_add_edge_sharpening",
    "_extract_edge_probs",
    "_apply_darts_pt_perturbation",
    "_restore_model_params",
    "compute_implicit_arch_gradient_correction",
    "finite_difference_hessian_penalty",
    "_dynamic_arch_update_freq",
    "_dynamic_inner_arch_iters",
    "_maybe_prune",
    "_safe_load_state",
    "_log_arch_gradients",
]
