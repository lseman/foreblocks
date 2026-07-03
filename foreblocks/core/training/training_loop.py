"""foreblocks.core.training.training_loop.

Core training-loop primitives: forward/backward passes, epoch-level training,
evaluation, and NAS alpha optimization steps.

Extracted from the monolithic Trainer to enable composability. The forward pass
handles three model signatures via graceful fallback and distillation-aware models
with teacher output retrieval. The backward pass supports gradient accumulation
and AMP with step-level schedulers.

Core API:
- train_epoch: train for one epoch with optional NAS warmup
- evaluate: evaluate model on a DataLoader and return mean loss
- forward_pass: run a forward pass with signature fallback and distillation support
- backward_step: execute backward pass with gradient accumulation and clipping

"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from foreblocks.core.training.batch_io import (
    loader_len,
    move_batch_to_device,
    to_device,
    unpack_batch,
)

# ====================================================================
# Forward / backward helpers
# ====================================================================


def forward_pass(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor | None,
    time_feat: torch.Tensor | None,
    current_epoch: int,
    batch_idx: int,
    moe_log: Any,
    moe_meta_builder: Any,
    graph_kwargs: dict[str, Any] | None,
    amp_context: Any,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run a forward pass and return ``(outputs, aux_dict)``.

    Handles three model signatures via graceful fallback:
    1. ``model(X, y, time_feat, epoch, meta=…)``  – full signature
    2. ``model(X, y, time_feat, epoch)``           – no meta
    3. ``model(X)``                                 – bare

    Also handles distillation-aware models that implement
    ``get_distillation_info()`` and ``return_teacher_outputs``.
    """
    aux: dict[str, Any] = {}
    graph_kwargs = graph_kwargs or {}
    meta = (
        moe_meta_builder(X, y, time_feat, current_epoch, batch_idx)
        if moe_log is not None
        else None
    )

    # ── Distillation path ────────────────────────────────────────────
    if hasattr(model, "get_distillation_info"):
        distill_info = model.get_distillation_info()
        if distill_info.get("distillation_enabled", False):
            try:
                result = model(
                    X,
                    y,
                    time_feat,
                    current_epoch,
                    return_teacher_outputs=True,
                    meta=meta,
                )
            except TypeError:
                result = model(
                    X,
                    y,
                    time_feat,
                    current_epoch,
                    return_teacher_outputs=True,
                )
            if isinstance(result, tuple) and len(result) == 2:
                outputs, aux["teacher_outputs"] = result[0], result[1]
            else:
                outputs = result[0] if isinstance(result, tuple) else result
            return outputs, aux

    # ── Graph kwargs path ────────────────────────────────────────────
    if graph_kwargs:
        result = model(X, **graph_kwargs)
        outputs = result[0] if isinstance(result, tuple) else result
        return outputs, aux

    # ── Standard paths (graceful signature fallback) ─────────────────
    try:
        result = model(X, y, time_feat, current_epoch, meta=meta)
    except TypeError:
        try:
            result = model(X, y, time_feat, current_epoch)
        except TypeError:
            result = model(X)

    outputs = result[0] if isinstance(result, tuple) else result
    return outputs, aux


def backward_step(
    loss: torch.Tensor,
    model: nn.Module,
    config: Any,
    optimizer: torch.optim.Optimizer,
    batch_idx: int,
    total_batches: int,
    scaler: Any,
    amp_enabled: bool,
    scheduler: Any = None,
) -> None:
    """Execute the backward pass with gradient accumulation and clipping.

    If AMP is enabled the loss is scaled before ``.backward()`` and the
    optimizer is stepped via ``scaler.step()``.  Otherwise standard
    ``loss.backward()`` + ``optimizer.step()`` is used.  The optimizer
    is only stepped when the gradient-accumulation boundary is reached
    or the last batch is processed.

    If a step-level scheduler (e.g., WarmupCosineLR) is provided, it is
    stepped immediately after the optimizer step.
    """
    loss = loss / config.gradient_accumulation_steps

    clip_val = getattr(config, "gradient_clip_val", None)

    if amp_enabled:
        scaler.scale(loss).backward()
        if (
            (batch_idx + 1) % config.gradient_accumulation_steps == 0
            or batch_idx + 1 == total_batches
        ):
            scaler.unscale_(optimizer)
            if clip_val:
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    else:
        loss.backward()
        if (
            (batch_idx + 1) % config.gradient_accumulation_steps == 0
            or batch_idx + 1 == total_batches
        ):
            if clip_val:
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)


# ====================================================================
# Epoch-level training & evaluation
# ====================================================================


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    config: Any,
    loss_computer: Any,
    optimizer: torch.optim.Optimizer,
    global_step_ref: dict[str, int],
    nas_helper: Any,
    scaler: Any,
    amp_context: Any,
    moe_log: Any,
    moe_meta_builder: Any,
    current_epoch: int = 0,
    forward_pass_fn: Any = forward_pass,
    backward_step_fn: Any = backward_step,
    device: torch.device = torch.device("cpu"),
    scheduler: Any = None,
) -> tuple[float, dict[str, float], int]:
    """Train for one epoch.  Returns ``(total_loss, avg_components, batch_idx)``.

    Optionally performs NAS alpha-optimization warmup before the main loop.
    """
    model.train()
    total_loss = 0.0
    all_components: dict[str, list[float]] = {}

    do_nas = (
        config.train_nas
        and nas_helper.has_nas
        and getattr(config, "nas_warmup_epochs", 0) is not None
        and (
            val_loader is not None or not getattr(config, "nas_use_val_for_alpha", True)
        )
    )

    if do_nas and getattr(config, "nas_use_val_for_alpha", True):
        alpha_loss = _alpha_step(
            val_loader,
            num_steps=getattr(config, "nas_alternate_steps", 1),
            config=config,
            loss_computer=loss_computer,
            optimizer=nas_helper.get_alpha_optimizer(),
            scaler=scaler,
            amp_context=amp_context,
            model=model,
            device=device,
            forward_pass_fn=forward_pass_fn,
        )
        all_components.setdefault("alpha_loss", []).append(alpha_loss)

    if do_nas:
        for p in nas_helper.get_alpha_parameters():
            p.requires_grad_(False)

    total_batches = loader_len(train_loader)
    if total_batches is None:
        total_batches = 0

    batch_idx = 0
    try:
        for batch_idx, batch in enumerate(train_loader):
            X, y, time_feat, graph_kwargs = unpack_batch(batch)
            X, y, time_feat, graph_kwargs = move_batch_to_device(
                X, y, time_feat, graph_kwargs, device
            )

            with amp_context():
                outputs, aux = forward_pass_fn(
                    model,
                    X,
                    y,
                    time_feat,
                    current_epoch,
                    batch_idx,
                    moe_log,
                    moe_meta_builder,
                    graph_kwargs,
                    amp_context,
                    device,
                )
                if y is not None and outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
                target = y if y is not None else outputs.detach() * 0
                loss = loss_computer.compute(outputs, target, aux)

            amp_enabled = getattr(config, "use_amp", False) and device.type == "cuda"
            backward_step_fn(
                loss,
                model,
                config,
                optimizer,
                batch_idx,
                total_batches,
                scaler,
                amp_enabled,
                scheduler=scheduler,
            )
            total_loss += float(loss.detach())
            global_step_ref["step"] = global_step_ref.get("step", 0) + 1

            for k, v in loss_computer.components.items():
                all_components.setdefault(k, []).append(v)
    finally:
        if do_nas:
            for p in nas_helper.get_alpha_parameters():
                p.requires_grad_(True)

    avg_components = {k: float(np.mean(v)) for k, v in all_components.items()}
    batches_seen = max(1, batch_idx + 1) if "batch_idx" in locals() else 1
    return total_loss / batches_seen, avg_components, batches_seen


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu"),
    amp_context: Any = contextlib.nullcontext,
    moe_log: Any = None,
    moe_meta_builder: Any = None,
    forward_pass_fn: Any = forward_pass,
) -> float:
    """Evaluate model on *dataloader* and return mean loss."""
    if dataloader is None:
        return float("nan")
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            X, y, time_feat, graph_kwargs = unpack_batch(batch)
            X, y, time_feat, graph_kwargs = move_batch_to_device(
                X, y, time_feat, graph_kwargs, device
            )

            with amp_context():
                outputs, _ = forward_pass_fn(
                    model,
                    X,
                    y,
                    time_feat,
                    0,
                    batch_idx,
                    moe_log,
                    moe_meta_builder,
                    graph_kwargs,
                    amp_context,
                    device,
                )

                if y is None:
                    continue
                if outputs.ndim == 4 and y.ndim == 3:
                    y = y.unsqueeze(-1)
                loss = nn.MSELoss()(outputs, y)
                bs = X.size(0)
                total_loss += float(loss.detach()) * bs
                n += bs

    return (total_loss / max(n, 1)) if n > 0 else float("nan")


# ====================================================================
# NAS alpha-step
# ====================================================================


def _alpha_step(
    dataloader: DataLoader,
    num_steps: int = 1,
    *,
    config: Any,
    loss_computer: Any,
    optimizer: torch.optim.Optimizer | None,
    scaler: Any,
    amp_context: Any,
    model: nn.Module,
    device: torch.device,
    forward_pass_fn: Any = forward_pass,
) -> float:
    """Optimize NAS architecture parameters (alphas) for one step.

    Freezes the weights (θ) and takes a gradient step on the architecture
    parameters (α).  Includes extensive debug-fail-fast checks to surface
    common NAS wiring issues early.
    """
    if optimizer is None:
        raise ValueError("[NAS] alpha_optimizer is None.")

    total_loss = 0.0
    alpha_params = (
        list(optimizer.param_groups[0]["params"]) if optimizer.param_groups else []
    )

    # Freeze θ, train α
    weight_params = [p for p in model.parameters() if p not in set(alpha_params)]
    for p in weight_params:
        p.requires_grad_(False)
    for p in alpha_params:
        p.requires_grad_(True)

    prev_mode = model.training
    model.eval()

    dataloader_iter = iter(dataloader)

    try:
        for step in range(num_steps):
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            X, y, time_feat, graph_kwargs = unpack_batch(batch)
            X, y, time_feat, graph_kwargs = move_batch_to_device(
                X, y, time_feat, graph_kwargs, device
            )

            with torch.set_grad_enabled(True):
                with amp_context():
                    outputs, aux = forward_pass_fn(
                        model,
                        X,
                        y,
                        time_feat,
                        0,
                        step,
                        None,
                        None,
                        graph_kwargs,
                        amp_context,
                        device,
                    )

                    if y is None:
                        target = outputs.detach().clone()
                    else:
                        target = y
                    target.requires_grad_(False)

                    loss = loss_computer.compute(outputs, target, aux)

                # ── Debug / Fail Fast ──────────────────────────────────
                if not alpha_params:
                    raise RuntimeError(
                        "[NAS] alpha_params is empty. NASHelper did not find "
                        "architecture parameters."
                    )
                if not any(p.requires_grad for p in alpha_params):
                    names = []
                    for n, p in model.named_parameters():
                        if p in set(alpha_params):
                            names.append((n, p.requires_grad))
                    raise RuntimeError(
                        f"[NAS] No alpha params require grad. Found: {names[:20]}"
                    )
                if not torch.is_tensor(loss):
                    raise RuntimeError(
                        f"[NAS] LossComputer.compute returned non-tensor: {type(loss)}"
                    )
                if not outputs.requires_grad:
                    raise RuntimeError(
                        "[NAS] outputs.requires_grad is False during alpha step. "
                        "The forward pass is not connected to alpha parameters "
                        "(likely using argmax/.item()/hard indexing on alphas)."
                    )
                if not loss.requires_grad:
                    raise RuntimeError(
                        "[NAS] loss.requires_grad is False during alpha step. "
                        "Either outputs are detached, LossComputer detaches "
                        "internally, or alphas are not used differentiably."
                    )
                # ────────────────────────────────────────────────────────

                optimizer.zero_grad(set_to_none=True)
                if getattr(config, "use_amp", False):
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

                for p in weight_params:
                    p.grad = None

                total_loss += float(loss.detach())

    finally:
        model.train(prev_mode)
        for p in weight_params:
            p.requires_grad_(True)

    return total_loss / max(num_steps, 1)
