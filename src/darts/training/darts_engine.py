"""
DARTS Engine — configurable multi-variant NAS training.

This module provides a unified entry-point that:

1. Configures the MixedOp / cell forward-pass according to the chosen
   ``DARTSVariant`` (GDAS, GD-DARTS, R-DARTS, PC-DARTS, Bi-DARTS).
2. Injects variant-specific gradient-balance logic into the bilevel
   optimisation loop.
3. Handles bidirectional (Bi-DARTS) training passes.

The engine is *drop-in* compatible with the existing
``darts.training.darts_loop.train_darts_model`` — callers simply pass
a ``DARTSEngineConfig`` instead of a long kwarg list.

Usage::

    from darts.config import DARTSEngineConfig, DARTSVariant

    engine_cfg = DARTSEngineConfig(
        variant=DARTSVariant.R_DARTS,
        r_darts=R_DARTSEngineConfig(balance_gradient_norms=True),
    )

    results = trainer.train_darts_model(
        model, train_loader, val_loader,
        engine=engine_cfg,  # new keyword
        epochs=50,
    )
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# MixedOp variant hooks
# ---------------------------------------------------------------------------


def configure_mixed_op_for_variant(
    model: nn.Module,
    variant: str,
    engine_cfg: Any,
    epoch: int,
    total_epochs: int,
) -> None:
    """Configure MixedOp / DARTSCell attributes per variant.

    This is called at the *start* of each epoch inside the training loop.
    It modifies the model in-place so that the forward pass behaves
    according to the selected variant.

    Args:
        model: The TimeSeriesDARTS / DARTSCell model.
        variant: One of "darts", "gd_darts", "r_darts", "pc_darts", "bi_darts".
        engine_cfg: The DARTSEngineConfig (provides variant-specific params).
        epoch: Current epoch (0-indexed).
        total_epochs: Total number of epochs.
    """
    if variant == "gd_darts":
        _configure_gd_darts(model, engine_cfg)
    elif variant == "pc_darts":
        _configure_pc_darts(model, engine_cfg)
    elif variant == "r_darts":
        _configure_r_darts(model, engine_cfg)
    elif variant == "bi_darts":
        _configure_bi_darts(model, engine_cfg)


def _configure_gd_darts(model: nn.Module, engine_cfg: Any) -> None:
    """GD-DARTS: disable Gumbel-Softmax, use pure straight-through softmax."""
    for m in model.modules():
        if hasattr(m, "use_gumbel"):
            m.use_gumbel = False
        if hasattr(m, "op_gdas"):
            m.op_gdas = False
        if hasattr(m, "pc_darts_enabled"):
            m.pc_darts_enabled = False
    # Set a low fixed temperature for the commitment phase.
    for m in model.modules():
        if hasattr(m, "op_temperature"):
            m.op_temperature = engine_cfg.gd_darts.commitment_temperature


def _configure_pc_darts(model: nn.Module, engine_cfg: Any) -> None:
    """PC-DARTS: enable partial-channel execution and edge normalization."""
    for m in model.modules():
        if hasattr(m, "op_gdas"):
            m.op_gdas = False  # PC-DARTS uses dense forward, not GDAS
        if hasattr(m, "pc_darts_enabled"):
            m.pc_darts_enabled = bool(engine_cfg.pc_darts.enable_partial_channels)
    # Normalize competing incoming edges as described by PC-DARTS. The legacy
    # permutation regularizer remains opt-in and independent of this behavior.
    for cell in getattr(model, "cells", []):
        if not hasattr(cell, "edges"):
            continue
        cell.pc_darts_enabled = bool(engine_cfg.pc_darts.enable_edge_normalization)
        for edge in cell.edges:
            if hasattr(edge, "perm_l2_weight"):
                edge.perm_l2_weight = engine_cfg.pc_darts.perm_l2_weight


def _configure_r_darts(model: nn.Module, engine_cfg: Any) -> None:
    """R-DARTS: enable gradient-norm balancing."""
    for m in model.modules():
        if hasattr(m, "op_gdas"):
            m.op_gdas = True  # R-DARTS works well with GDAS
        if hasattr(m, "pc_darts_enabled"):
            m.pc_darts_enabled = False
    # Set gradient-balance warmup on all cells.
    for cell in getattr(model, "cells", []):
        if hasattr(cell, "norm_balance_warmup"):
            cell.norm_balance_warmup = engine_cfg.r_darts.norm_balance_warmup
        if hasattr(cell, "balance_gradient_norms"):
            cell.balance_gradient_norms = engine_cfg.r_darts.balance_gradient_norms


def _configure_bi_darts(model: nn.Module, engine_cfg: Any) -> None:
    """Bi-DARTS: enable bidirectional training."""
    for m in model.modules():
        if hasattr(m, "op_gdas"):
            m.op_gdas = False  # Bi-DARTS uses dense forward pass
        if hasattr(m, "pc_darts_enabled"):
            m.pc_darts_enabled = False
    # Set bidirectional training flags.
    for cell in getattr(model, "cells", []):
        if hasattr(cell, "bidirectional_training"):
            cell.bidirectional_training = True
        if hasattr(cell, "backward_loss_weight"):
            cell.backward_loss_weight = engine_cfg.bi_darts.backward_loss_weight
        if hasattr(cell, "backward_passes"):
            cell.backward_passes = engine_cfg.bi_darts.backward_passes


# ---------------------------------------------------------------------------
# Gradient-norm balancing (R-DARTS)
# ---------------------------------------------------------------------------


def compute_gradient_norm_balance(
    model: nn.Module,
    arch_grads: list[torch.Tensor],
    model_grads: list[torch.Tensor],
    warmup_epochs: int,
    epoch: int,
) -> float:
    """Compute the gradient-norm balance factor for R-DARTS.

    Returns a scalar multiplier ``s`` such that the arch-gradient update
    is scaled by ``s`` to match the weight-gradient scale.

    Formula::

        s = ||∇_w L_arch|| / ||∇_α L_arch||

    When the arch gradient is much larger than the weight gradient (common
    in early training), ``s < 1`` and the arch update is scaled down to
    prevent the architecture from converging prematurely.

    Args:
        model: The DARTS model.
        arch_grads: Architecture parameter gradients (after backward).
        model_grads: Model parameter gradients (after backward).
        warmup_epochs: Warmup period during which balancing is disabled.
        epoch: Current epoch (0-indexed).

    Returns:
        Gradient-norm balance factor (≥ 0.0).
    """
    if epoch < warmup_epochs:
        return 1.0

    arch_grads = [g for g in arch_grads if g is not None and g.numel() > 0]
    model_grads = [g for g in model_grads if g is not None and g.numel() > 0]
    if not arch_grads or not model_grads:
        return 1.0

    # Compute weight gradient norm.
    weight_norm_sq = torch.tensor(0.0, device=model_grads[0].device if model_grads else "cpu")
    for g in model_grads:
        weight_norm_sq += g.detach().pow(2).sum()
    weight_norm = weight_norm_sq.sqrt().clamp_min(1e-12)

    # Compute arch gradient norm.
    arch_norm_sq = torch.tensor(0.0, device=model_grads[0].device if model_grads else "cpu")
    for g in arch_grads:
        arch_norm_sq += g.detach().pow(2).sum()
    arch_norm = arch_norm_sq.sqrt().clamp_min(1e-12)

    # Balance factor: scale arch update to match weight scale.
    return float(weight_norm / arch_norm)


# ---------------------------------------------------------------------------
# Optional legacy permutation-consistency regularizer
# ---------------------------------------------------------------------------


def apply_permutation_consistency(
    edge_weights: dict[str, torch.Tensor],
    perm_l2_weight: float = 1e-4,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Apply permutation-consistency correction to edge weights.

    When two edges share the same underlying operation, their weights
    should be permuted to match — swapping edge order should not change
    the forward output.

    Args:
        edge_weights: Dict of {edge_name: weight_tensor}.
        perm_l2_weight: L2 regularizer on the permutation matrix.

    Returns:
        (corrected_weights, perm_l2_loss) where corrected_weights are
        the permuted weight tensors and perm_l2_loss is a small scalar
        representing the permutation regularizer.
    """
    if not edge_weights or len(edge_weights) < 2:
        return edge_weights, torch.tensor(0.0)

    first = next(iter(edge_weights.values()))
    pair_losses: list[torch.Tensor] = []
    names = sorted(edge_weights)
    for i, name_i in enumerate(names):
        flat_i = edge_weights[name_i].reshape(-1)
        for name_j in names[i + 1 :]:
            flat_j = edge_weights[name_j].reshape(-1)
            if flat_i.numel() != flat_j.numel():
                continue
            aligned_i = torch.sort(flat_i, stable=True).values
            aligned_j = torch.sort(flat_j, stable=True).values
            pair_losses.append((aligned_i - aligned_j).square().mean())

    if not pair_losses:
        return dict(edge_weights), first.new_tensor(0.0)
    loss = torch.stack(pair_losses).mean() * float(perm_l2_weight)
    return dict(edge_weights), loss


def model_permutation_consistency_loss(
    model: nn.Module, perm_l2_weight: float = 1e-4
) -> torch.Tensor:
    """Compute differentiable consistency loss across compatible cell edges."""
    edge_weights: dict[str, torch.Tensor] = {}
    for cell_idx, cell in enumerate(getattr(model, "cells", [])):
        for edge_idx, edge in enumerate(getattr(cell, "edges", [])):
            if hasattr(edge, "get_alphas"):
                try:
                    weights = edge.get_alphas(detach=False)
                except TypeError:
                    weights = edge.get_alphas()
            else:
                weights = getattr(edge, "alphas", None)
            if isinstance(weights, torch.Tensor) and weights.numel() > 1:
                edge_weights[f"cell{cell_idx}.edge{edge_idx}"] = weights

    if not edge_weights:
        parameter = next(model.parameters(), None)
        return torch.tensor(0.0, device=parameter.device if parameter is not None else "cpu")
    return apply_permutation_consistency(edge_weights, perm_l2_weight)[1]


# ---------------------------------------------------------------------------
# Bidirectional forward pass (Bi-DARTS)
# ---------------------------------------------------------------------------


def forward_bidirectional(
    model: nn.Module,
    x: torch.Tensor,
    backward_loss_weight: float = 0.5,
    backward_passes: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Run a bidirectional forward pass through the model.

    The forward pass computes predictions normally.  The backward pass
    reverses the input sequence and runs the model again, producing
    a second set of predictions that are combined with the forward
    predictions.

    Args:
        model: The DARTS model (must support reverse-sequence input).
        x: Input tensor [B, L, C].
        backward_loss_weight: Weight for the backward-pass loss.
        backward_passes: Number of backward passes.

    Returns:
        Combined output tensor.
    """
    # Forward pass.
    kwargs = model_kwargs or {}
    forward_out = model(x, **kwargs)

    # Backward pass: reverse the sequence dimension.
    x_rev = x.flip(dims=[1])
    reversed_kwargs = {
        name: value.flip(dims=[1])
        if isinstance(value, torch.Tensor) and value.dim() >= 2
        else value
        for name, value in kwargs.items()
    }
    backward_outputs = [
        model(x_rev, **reversed_kwargs).flip(dims=[1])
        for _ in range(max(int(backward_passes), 1))
    ]
    backward_out = torch.stack(backward_outputs).mean(dim=0)

    # Combine forward and backward outputs.
    backward_weight = max(float(backward_loss_weight), 0.0)
    combined = (forward_out + backward_weight * backward_out) / (
        1.0 + backward_weight
    )

    return combined


def compute_backward_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: callable,
    backward_loss_weight: float = 0.5,
    backward_passes: int = 1,
    model_kwargs: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute the backward-pass loss for Bi-DARTS.

    The backward pass runs the model on the reversed input sequence
    and compares the reversed output against the reversed target.

    Args:
        model: The DARTS model.
        x: Input tensor [B, L, C].
        y: Target tensor [B, L, C].
        loss_fn: Loss function.
        backward_loss_weight: Weight for the backward-pass loss.

    Returns:
        Combined loss (forward + backward).
    """
    # Forward loss.
    kwargs = model_kwargs or {}
    forward_out = model(x, **kwargs)
    forward_loss = loss_fn(forward_out, y)

    # Backward loss: reverse both input and target.
    x_rev = x.flip(dims=[1])
    y_rev = y.flip(dims=[1])
    reversed_kwargs = {
        name: value.flip(dims=[1])
        if isinstance(value, torch.Tensor) and value.dim() >= 2
        else value
        for name, value in kwargs.items()
    }
    backward_losses = []
    for _ in range(max(int(backward_passes), 1)):
        backward_out = model(x_rev, **reversed_kwargs)
        backward_losses.append(loss_fn(backward_out, y_rev))
    backward_loss = torch.stack(backward_losses).mean()

    # Combined loss.
    return forward_loss + backward_loss_weight * backward_loss


# ---------------------------------------------------------------------------
# Engine factory — build a complete training configuration
# ---------------------------------------------------------------------------


def build_engine_config(
    variant: str = "r_darts",
    **overrides: Any,
) -> dict[str, Any]:
    """Build a variant-specific engine configuration dict.

    This is a convenience factory that creates the appropriate engine
    config for the given variant, then merges any overrides.

    Args:
        variant: One of "darts", "gd_darts", "r_darts", "pc_darts", "bi_darts".
        **overrides: Additional keyword arguments to override defaults.

    Returns:
        Engine config dict ready to be passed to the training loop.
    """
    config = {
        "darts": {},
        "gd_darts": {
            "replace_gumbel_softmax": True,
            "commitment_temperature": 0.1,
        },
        "r_darts": {
            "use_adamw_arch": True,
            "balance_gradient_norms": True,
            "arch_grad_scale": 1.0,
            "norm_balance_warmup": 2,
        },
        "pc_darts": {
            "enable_partial_channels": True,
            "enable_edge_normalization": True,
            "enable_permutation_consistency": False,
            "perm_l2_weight": 1e-4,
        },
        "bi_darts": {
            "bidirectional_training": True,
            "backward_loss_weight": 0.5,
            "backward_passes": 1,
        },
    }

    result = config.get(variant, {})
    result.update(overrides)
    return result
