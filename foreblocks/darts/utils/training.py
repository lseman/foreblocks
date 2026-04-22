"""
Training-time utility functions shared across DARTS modules.

Covers loss functions, mixed-precision context managers, progress bars,
and parameter-group construction helpers.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

_LOSS_REGISTRY: dict[str, Any] = {
    "huber": lambda p, t: F.huber_loss(p, t, delta=0.1),
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "smooth_l1": F.smooth_l1_loss,
}


def get_loss_function(loss_type: str):
    """
    Return a loss callable by name.

    Supported names: ``"huber"`` (default, delta=0.1), ``"mse"``,
    ``"mae"``, ``"smooth_l1"``.

    Args:
        loss_type: One of the supported loss names.

    Returns:
        Callable ``(predictions, targets) -> scalar Tensor``.
    """
    return _LOSS_REGISTRY.get(loss_type, _LOSS_REGISTRY["huber"])


def register_loss(name: str, fn) -> None:
    """Register a custom loss function under *name*."""
    _LOSS_REGISTRY[name] = fn


# ---------------------------------------------------------------------------
# Mixed-precision
# ---------------------------------------------------------------------------


@contextmanager
def autocast_ctx(device: str, enabled: bool = True):
    """
    Context manager wrapping :class:`torch.amp.autocast`.

    Args:
        device:  Device string (e.g., ``"cuda:0"`` or ``"cpu"``).
        enabled: Whether to activate mixed precision.

    Yields:
        Nothing – use as ``with autocast_ctx(device, enabled=use_amp):``.
    """
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    with autocast(device_type=device_type, enabled=enabled):
        yield


# ---------------------------------------------------------------------------
# Progress bars
# ---------------------------------------------------------------------------


def create_progress_bar(iterable, desc: str, *, leave: bool = True, **kwargs):
    """
    Thin wrapper around ``tqdm`` with sensible defaults.

    Args:
        iterable: The iterable to wrap.
        desc:     Progress bar description.
        leave:    Whether to keep the bar after completion.
        **kwargs: Forwarded to ``tqdm``.

    Returns:
        A ``tqdm`` progress bar wrapping *iterable*.
    """
    kwargs.setdefault("unit", "batch")
    return tqdm(iterable, desc=desc, leave=leave, **kwargs)


def unpack_forecasting_batch(
    batch,
    device: str,
    *,
    include_decoder_targets: bool = False,
    teacher_forcing_ratio: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | float]]:
    """
    Parse a forecasting batch into model inputs, targets, and model kwargs.

    Supported batch layouts:
    - ``(x, y)``
    - ``(x, y, x_future)``
    - ``(x, y, {"x_future": ..., ...})``

    The second tensor ``y`` is treated as the supervised forecast target. When
    ``include_decoder_targets`` is enabled and shapes are compatible, ``y`` is
    also exposed to the model as ``decoder_targets`` for teacher forcing. Known
    future covariates remain a separate optional input ``x_future`` to avoid
    leaking labels into evaluation.
    """
    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
        raise ValueError("Expected forecasting batch to contain at least (x, y).")

    x = batch[0].to(device, non_blocking=True)
    y = batch[1].to(device, non_blocking=True)
    model_kwargs: dict[str, torch.Tensor | float] = {}

    x_future = None
    if len(batch) >= 3:
        extra = batch[2]
        if isinstance(extra, dict):
            extra_value = extra.get("x_future", None)
            if extra_value is None:
                extra_value = extra.get("future_covariates", None)
            if extra_value is None:
                extra_value = extra.get("dec", None)
            extra = extra_value
        if isinstance(extra, torch.Tensor):
            candidate = extra.to(device, non_blocking=True)
            if (
                candidate.dim() == 3
                and candidate.size(0) == x.size(0)
                and candidate.size(-1) == x.size(-1)
                and (y.dim() != 3 or candidate.size(1) >= y.size(1))
            ):
                x_future = candidate

    if x_future is not None:
        model_kwargs["x_future"] = x_future

    if (
        include_decoder_targets
        and isinstance(y, torch.Tensor)
        and y.dim() == 3
        and y.size(-1) == x.size(-1)
    ):
        model_kwargs["decoder_targets"] = y

    if teacher_forcing_ratio is not None:
        model_kwargs["teacher_forcing_ratio"] = float(teacher_forcing_ratio)

    return x, y, model_kwargs


# ---------------------------------------------------------------------------
# Parameter group splitting
# ---------------------------------------------------------------------------


def split_arch_and_model_params(
    model: nn.Module,
    alpha_tracker=None,
) -> tuple[list, list, list, list]:
    """
    Separate architecture (alpha) parameters from regular model parameters.

    Args:
        model:         The DARTS model.
        alpha_tracker: Optional :class:`AlphaTracker` instance used to
                       pick up component alpha sources the name-heuristic
                       might miss.

    Returns:
        Four lists: ``(arch_params, model_params,
                       edge_arch_params, component_arch_params)``.
    """
    arch_params: list[torch.Tensor] = []
    model_params: list[torch.Tensor] = []
    edge_arch_params: list[torch.Tensor] = []
    component_arch_params: list[torch.Tensor] = []
    arch_param_ids = set()

    _arch_names = ("alphas", "arch_", "alpha_", "norm_alpha")

    for name, param in model.named_parameters():
        if any(k in name for k in _arch_names):
            if id(param) in arch_param_ids:
                continue
            arch_params.append(param)
            arch_param_ids.add(id(param))
            if "cells." in name and "edges." in name:
                edge_arch_params.append(param)
            else:
                component_arch_params.append(param)
        else:
            model_params.append(param)

    if alpha_tracker is not None:
        for source in alpha_tracker.component_alpha_sources(model):
            tensor = source["alpha"]
            if not isinstance(tensor, nn.Parameter):
                continue
            if id(tensor) in arch_param_ids:
                continue
            arch_params.append(tensor)
            arch_param_ids.add(id(tensor))
            component_arch_params.append(tensor)

    return arch_params, model_params, edge_arch_params, component_arch_params


def build_arch_param_groups(
    edge_arch_params: list[torch.Tensor],
    component_arch_params: list[torch.Tensor],
    arch_params: list[torch.Tensor],
    arch_learning_rate: float,
) -> list[dict[str, Any]]:
    """
    Build optimizer parameter groups for architecture parameters.

    Edge-level parameters use a 1.5× learning-rate boost to compensate for
    their comparatively small gradient signal.

    Args:
        edge_arch_params:      Parameters tied to cell edges.
        component_arch_params: Parameters for the encoder/decoder
                               architecture.
        arch_params:           Fallback list used when both specific lists
                               are empty.
        arch_learning_rate:    Base learning rate for architecture params.

    Returns:
        List of optimizer group dicts (each with ``"params"`` and ``"lr"``).
    """
    groups = []
    if edge_arch_params:
        groups.append({"params": edge_arch_params, "lr": arch_learning_rate * 1.5})
    if component_arch_params:
        groups.append({"params": component_arch_params, "lr": arch_learning_rate})
    if not groups:
        groups = [{"params": arch_params, "lr": arch_learning_rate}]
    return groups


# ---------------------------------------------------------------------------
# Model-weight utilities
# ---------------------------------------------------------------------------


def reset_model_parameters(model: nn.Module) -> int:
    """
    Call ``reset_parameters()`` on every sub-module that exposes it.

    Args:
        model: Any :class:`nn.Module`.

    Returns:
        Number of modules whose parameters were successfully reset.
    """
    count = 0
    for module in model.modules():
        fn = getattr(module, "reset_parameters", None)
        if callable(fn):
            try:
                fn()
                count += 1
            except Exception:
                continue
    return count


def capture_progressive_state(model: nn.Module) -> dict[str, Any] | None:
    """
    Snapshot the ``progressive_stage`` of each DARTS cell.

    This is needed so the best checkpoint can be restored after progressive-
    shrinking changes the cell configuration during training.

    Returns:
        A dict or ``None`` if the model has no ``cells`` attribute.
    """
    if not hasattr(model, "cells"):
        return None
    return {
        "cells": [
            {"progressive_stage": getattr(cell, "progressive_stage", None)}
            for cell in model.cells
        ]
    }


def restore_progressive_state(
    model: nn.Module, state: dict[str, Any] | None
) -> None:
    """
    Restore the ``progressive_stage`` snapshot captured by
    :func:`capture_progressive_state`.

    Args:
        model: The DARTS model.
        state: Snapshot dict (may be ``None``, in which case this is a no-op).
    """
    if state is None or not hasattr(model, "cells"):
        return
    cells = list(model.cells)
    for idx, cell in enumerate(cells):
        if idx >= len(state["cells"]):
            break
        stage = state["cells"][idx].get("progressive_stage")
        if stage is None:
            continue
        if hasattr(cell, "set_progressive_stage"):
            try:
                cell.set_progressive_stage(stage)
            except Exception:
                continue
        else:
            cell.progressive_stage = stage
