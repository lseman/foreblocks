"""
DARTS bilevel training loop.

Extracted from ``DARTSTrainer`` so the logic lives in one focused module.
The public entry-point is :func:`train_darts_model`.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from ..config import DARTSEngineConfig
from ..evaluation.metrics import compute_final_metrics
from ..training.dynamic_scheduling import (
    _dynamic_arch_update_freq,
    _dynamic_inner_arch_iters,
)
from ..training.architecture_step import compose_architecture_loss
from ..training.darts_engine import configure_mixed_op_for_variant
from ..training.edge_regularization import (
    _extract_edge_probs,
)
from ..training.helpers import (
    ArchitectureRegularizer,
    BilevelOptimizer,
    RegularizationType,
    TemperatureScheduler,
)
from ..training.perturbation_hessian import (
    _apply_darts_pt_perturbation,
    _restore_model_params,
    compute_implicit_arch_gradient_correction,
    finite_difference_hessian_penalty,
)
from ..training.utils import (
    _log_arch_gradients,
    _maybe_prune,
    _safe_load_state,
    snapshot_state_dict,
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


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _use_fused_optimizers(device: str) -> bool:
    # Use fused Adam/AdamW on CUDA by default. Keep an escape hatch for unusual
    # PyTorch/driver combinations where the native fused kernels misbehave.
    return device.startswith("cuda") and not _env_flag(
        "FORE_DARTS_DISABLE_FUSED_OPTIMIZERS"
    )


def _debug_crash_enabled() -> bool:
    return _env_flag("FORE_DARTS_DEBUG_CRASH")


def _enable_fatal_signal_tracebacks() -> None:
    if _debug_crash_enabled() and not faulthandler.is_enabled():
        faulthandler.enable(file=sys.stderr, all_threads=True)


@contextlib.contextmanager
def _crash_trace(stage: str, device: str = "cpu"):
    if not _debug_crash_enabled():
        yield
        return

    print(f"[DARTS-CRASH-TRACE] >> {stage}", flush=True)
    t0 = time.perf_counter()
    try:
        yield
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    finally:
        dt = time.perf_counter() - t0
        print(f"[DARTS-CRASH-TRACE] << {stage} ({dt:.3f}s)", flush=True)


def _optimizer_steps_per_epoch(
    loader_length: int,
    max_batches: int | None,
    gradient_accumulation_steps: int,
) -> int:
    """Return the number of optimizer steps, including a partial final window."""
    batch_budget = max(int(loader_length), 0)
    if max_batches is not None:
        batch_budget = min(batch_budget, max(int(max_batches), 0))
    accumulation = max(int(gradient_accumulation_steps), 1)
    return max(1, (batch_budget + accumulation - 1) // accumulation)


def _prepare_edge_routing(model: nn.Module) -> None:
    """Refresh discrete cell topology once, outside repeated forward calls."""
    for module in model.modules():
        prepare = getattr(module, "prepare_edge_routing", None)
        if callable(prepare):
            prepare()


def train_darts_model(
    trainer,
    model: nn.Module,
    train_loader,
    val_loader,
    *,
    epochs: int = 50,
    arch_learning_rate: float = 1e-2,
    model_learning_rate: float = 1e-3,
    arch_weight_decay: float = 1e-3,
    model_weight_decay: float = 1e-4,
    patience: int = 10,
    loss_type: str = "huber",
    use_swa: bool = False,
    warmup_epochs: int = 2,
    architecture_update_freq: int = 3,
    diversity_check_freq: int = 1,
    progressive_shrinking: bool = True,
    hybrid_pruning_start_epoch: int = 20,
    hybrid_pruning_interval: int = 10,
    hybrid_pruning_base_threshold: float = 0.15,
    hybrid_pruning_strategy: str = "performance",
    hybrid_pruning_freeze_logit: float = -20.0,
    use_bilevel_optimization: bool = True,
    use_amp: bool = True,
    gradient_accumulation_steps: int = 1,
    verbose: bool = True,
    regularization_types: list[str] | None = None,
    regularization_weights: list[float] | None = None,
    temperature_schedule: str = "cosine",
    edge_sharpening_max_weight: float = 0.03,
    edge_sharpening_start_frac: float = 0.35,
    hessian_penalty_weight: float = 0.0,
    hessian_fd_eps: float = 1e-2,
    hessian_update_freq: int = 1,
    bilevel_split_seed: int = 42,
    state_mix_ortho_reg_weight: float = 1e-3,
    edge_diversity_weight: float = 0.03,
    edge_usage_balance_weight: float = 0.04,
    edge_identity_cap: float = 0.45,
    edge_identity_cap_weight: float = 0.02,
    arch_grad_ema_beta: float = 0.0,
    beta_darts_weight: float = 0.0,
    moe_balance_weight: float = 5e-3,
    transformer_exploration_weight: float = 1e-2,
    use_darts_pt: bool = False,
    darts_pt_xi: float = 0.01,
    initial_drnas_concentration: float = 10.0,
    final_drnas_concentration: float = 2.0,
    op_gdas: bool | None = None,
    variant_gdas: bool | None = None,
    compute_metrics: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    engine: DARTSEngineConfig | None = None,
) -> dict[str, Any]:
    """
    Run one DARTS bilevel training cycle.

    Args:
        trainer: DARTSTrainer instance providing ``device``, ``alpha_tracker``,
                 and helper methods.
        model:   The DARTS mixed-operation model.
        train_loader: DataLoader for training data.
        val_loader:   DataLoader for validation data.
        epochs / arch_learning_rate / ... : See :class:`~darts.config.DARTSTrainConfig`.

    Returns:
        Dict with keys: ``model``, ``train_losses``, ``val_losses``,
        ``alpha_values``, ``diversity_scores``, ``best_val_loss``,
        ``training_time``, ``final_metrics``.
    """
    device = trainer.device
    _enable_fatal_signal_tracebacks()
    loss_fn = trainer._get_loss_function(loss_type)
    with _crash_trace("model.to(device)", device):
        model = model.to(device)
    start_time = time.time()

    # ── GPU fast-path settings (Ampere+) ──────────────────────────────────
    # TF32 matmul/conv is a large free speedup on Ampere and newer with no
    # change to which architecture the search converges to.
    if device.startswith("cuda"):
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # cudnn.benchmark autotunes conv kernels on the first batch of each new
        # shape. That one-off cost only pays off over many steps, so for short
        # runs (e.g. phase-3 ASHA rungs of 2-4 epochs) it is net-negative —
        # gate it on a minimum epoch budget.
        torch.backends.cudnn.benchmark = epochs >= 10

    # Warm-up must leave room for at least one architecture-update epoch.
    # The arch loop runs only when ``epoch >= warmup_epochs``; with the default
    # warmup_epochs=2 a 2-epoch run (epochs 0,1) would never update the
    # architecture at all — training weights but doing zero search. Cap warmup
    # so short rungs still search.
    if warmup_epochs >= epochs:
        warmup_epochs = max(0, epochs - 1)

    resolved_engine_variant = None
    if engine is not None:
        resolved_engine_variant = engine.resolve_variant().value

    # ── Parameter groups ──────────────────────────────────────────────────
    (
        arch_params,
        model_params,
        edge_arch_params,
        component_arch_params,
    ) = split_arch_and_model_params(model, alpha_tracker=trainer.alpha_tracker)

    if verbose:
        print(
            f"Architecture params: {len(arch_params)}, Model params: {len(model_params)}"
        )

    arch_param_groups = build_arch_param_groups(
        edge_arch_params=edge_arch_params,
        component_arch_params=component_arch_params,
        arch_params=arch_params,
        arch_learning_rate=arch_learning_rate,
    )

    # ── Optimizers ────────────────────────────────────────────────────────
    with _crash_trace("optimizer.init", device):
        optimizer_kwargs = {"fused": True} if _use_fused_optimizers(device) else {}
        try:
            # Architecture params: Adam (betas=(0.5, 0.999) per DARTS paper).
            # Model params: AdamW — decoupled weight decay is strictly better for
            # over-parameterised models.
            arch_optimizer = torch.optim.Adam(
                arch_param_groups,
                betas=(0.5, 0.999),
                weight_decay=arch_weight_decay,
                **optimizer_kwargs,
            )
            model_optimizer = torch.optim.AdamW(
                model_params,
                lr=model_learning_rate,
                weight_decay=model_weight_decay,
                **optimizer_kwargs,
            )
        except (TypeError, RuntimeError):
            arch_optimizer = torch.optim.Adam(
                arch_param_groups,
                betas=(0.5, 0.999),
                weight_decay=arch_weight_decay,
            )
            model_optimizer = torch.optim.AdamW(
                model_params, lr=model_learning_rate, weight_decay=model_weight_decay
            )

    # ── Bilevel split ─────────────────────────────────────────────────────
    # Resolve this before sizing OneCycleLR: the model-side split can contain
    # fewer batches than the original loader.
    train_arch_loader = None
    if use_bilevel_optimization:
        train_arch_loader, train_model_loader = trainer._create_bilevel_loaders(
            train_loader, seed=bilevel_split_seed
        )
    else:
        train_model_loader = train_loader

    # ── Schedulers ────────────────────────────────────────────────────────
    arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        arch_optimizer, T_max=epochs, eta_min=arch_learning_rate * 0.01
    )
    model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        model_optimizer,
        max_lr=model_learning_rate,
        epochs=epochs,
        steps_per_epoch=_optimizer_steps_per_epoch(
            len(train_model_loader),
            max_train_batches,
            gradient_accumulation_steps,
        ),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    bilevel_optimizer = BilevelOptimizer(
        arch_optimizer=arch_optimizer,
        arch_scheduler=arch_scheduler,
        arch_params=arch_params,
        edge_arch_params=edge_arch_params,
        component_arch_params=component_arch_params,
        use_bilevel_optimization=use_bilevel_optimization,
        train_arch_loader=train_arch_loader,
        val_loader=val_loader,
        train_model_loader=train_model_loader,
        arch_grad_ema_beta=arch_grad_ema_beta,
    )

    # ── Regularization & temperature ──────────────────────────────────────
    regularization_types = regularization_types or ["kl_divergence", "efficiency"]
    regularization_weights = regularization_weights or [0.05, 0.01]
    reg_types = [RegularizationType(rt) for rt in regularization_types]
    regularizer = ArchitectureRegularizer(reg_types, regularization_weights)

    temp_scheduler = TemperatureScheduler(
        initial_temp=2.0,
        final_temp=0.1,
        schedule_type=temperature_schedule,
        warmup_epochs=warmup_epochs,
        initial_drnas_concentration=initial_drnas_concentration,
        final_drnas_concentration=final_drnas_concentration,
    )

    # ── SWA setup ─────────────────────────────────────────────────────────
    swa_model, swa_start = None, None
    if use_swa:
        swa_start = max(epochs // 2, warmup_epochs + 5)
        swa_model = torch.optim.swa_utils.AveragedModel(model).to(device)

    scaler = GradScaler(enabled=use_amp and device.startswith("cuda"))

    # ── State ─────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None
    best_progressive_state = None
    train_losses, val_losses, alpha_values, diversity_scores = [], [], [], []
    prev_component_probs: dict = {}
    prev_edge_probs: dict = {}
    last_edge_entropy = float("nan")
    last_edge_sharpen_weight = 0.0

    if verbose:
        print(f"Training DARTS for {epochs} epochs")
        print(f"  Arch LR: {arch_learning_rate}, Model LR: {model_learning_rate}")
        print(f"  Bilevel: {use_bilevel_optimization}, SWA: {use_swa}, AMP: {use_amp}")
        if hasattr(model, "max_active_edges_per_node"):
            print(
                f"  Cell edge budget: max_active_edges_per_node="
                f"{int(getattr(model, 'max_active_edges_per_node', 0))}"
            )
        print(
            f"  Edge regs: cosine={edge_diversity_weight:.3f}, "
            f"usage={edge_usage_balance_weight:.3f}, "
            f"id_cap={edge_identity_cap:.2f}@{edge_identity_cap_weight:.3f}"
        )
        if moe_balance_weight > 0.0:
            print(f"  MoE balance regularizer: {moe_balance_weight:.4f}")
        if transformer_exploration_weight > 0.0:
            print(
                "  Transformer exploration regularizer: "
                f"{transformer_exploration_weight:.4f}"
            )
        print("-" * 60)

    epoch_pbar = (
        create_progress_bar(range(epochs), "DARTS", leave=True, unit="epoch")
        if verbose
        else range(epochs)
    )

    epoch = 0
    for epoch in epoch_pbar:
        model.train()

        # -- Temperature scheduling ----------------------------------------
        if hasattr(model, "schedule_temperature"):
            current_temperature = model.schedule_temperature(
                epoch=epoch,
                total_epochs=epochs,
                schedule_type=temperature_schedule,
                final_temp=temp_scheduler.final_temp,
                warmup_epochs=warmup_epochs,
            )
        else:
            current_temperature = temp_scheduler.get_temperature(epoch, epochs)

        if hasattr(model, "set_temperature") and not hasattr(
            model, "schedule_temperature"
        ):
            model.set_temperature(current_temperature)

        # DrNAS concentration annealing: high → exploration (samples near mean),
        # low → exploitation (samples near simplex vertices).
        current_drnas_concentration = temp_scheduler.get_drnas_concentration(
            epoch, epochs
        )
        if hasattr(model, "set_drnas_concentration"):
            model.set_drnas_concentration(current_drnas_concentration)

        if progressive_shrinking and hasattr(model, "schedule_progressive_stage"):
            model.schedule_progressive_stage(epoch=epoch, total_epochs=epochs)

        # Apply the variant after generic scheduling so fixed variant policy
        # (for example GD-DARTS commitment temperature) wins for this epoch.
        if resolved_engine_variant is not None:
            configure_mixed_op_for_variant(
                model=model,
                variant=resolved_engine_variant,
                engine_cfg=engine,
                epoch=epoch,
                total_epochs=epochs,
            )

        # Engine policy supplies the default, while an explicit run-level flag
        # is the final override. Keep transformer and cell sampling aligned
        # unless variant_gdas was intentionally decoupled.
        if op_gdas is not None:
            for module in model.modules():
                if hasattr(module, "op_gdas"):
                    module.op_gdas = bool(op_gdas)
        effective_op_gdas = next(
            (
                bool(module.op_gdas)
                for module in model.modules()
                if hasattr(module, "op_gdas")
            ),
            True,
        )
        effective_variant_gdas = bool(
            effective_op_gdas if variant_gdas is None else variant_gdas
        )
        for module in model.modules():
            if hasattr(module, "variant_gdas"):
                module.variant_gdas = effective_variant_gdas

        _prepare_edge_routing(model)

        if epoch % 5 == 0:
            alpha_values.append(trainer.alpha_tracker.extract_alpha_values(model))

        # -- Architecture parameter updates ----------------------------------
        _arch_freq = _dynamic_arch_update_freq(
            epoch, epochs, warmup_epochs, architecture_update_freq
        )
        _arch_iters = _dynamic_inner_arch_iters(epoch, epochs, warmup_epochs)
        # Count the update cadence from the warm-up boundary so the first
        # post-warmup epoch always updates the architecture. The previous
        # ``epoch % _arch_freq`` keyed off the absolute epoch index, which on
        # short runs (e.g. a 2-epoch phase-3 rung) could skip the only eligible
        # epoch entirely — training weights but never searching.
        if epoch >= warmup_epochs and (epoch - warmup_epochs) % _arch_freq == 0:
            for _ in range(_arch_iters):
                arch_batch = bilevel_optimizer.next_arch_batch()
                arch_x, arch_y, arch_model_kwargs = unpack_forecasting_batch(
                    arch_batch,
                    device,
                    include_decoder_targets=True,
                )
                bilevel_optimizer.zero_arch_grads()
                # Weight gradients from one architecture inner-step must not
                # leak into the next R-DARTS norm reference.
                for param in model_params:
                    param.grad = None

                # -- Implicit arch gradient correction (second-order DARTS) ----
                # Replaces the finite-difference curvature *penalty* with a
                # direct correction applied to arch param gradients after
                # backward.  Correction = -xi/(2ε) * (∇_α L_val(w+) - ∇_α L_val(w-))
                # where w± are model weights perturbed along ∇_w L_train.
                implicit_corrections: list[torch.Tensor | None] | None = None
                hessian_penalty = torch.tensor(0.0, device=device)
                if (
                    hessian_penalty_weight > 0.0
                    and hessian_fd_eps > 0.0
                    and hessian_update_freq > 0
                    and (epoch - warmup_epochs) % hessian_update_freq == 0
                    and model_params
                    and arch_params
                ):
                    h_batch = bilevel_optimizer.next_hessian_batch()
                    h_train_x, h_train_y, h_train_model_kwargs = (
                        unpack_forecasting_batch(
                            h_batch, device, include_decoder_targets=True
                        )
                    )
                    implicit_corrections = compute_implicit_arch_gradient_correction(
                        model=model,
                        loss_fn=loss_fn,
                        arch_x=arch_x,
                        arch_y=arch_y,
                        arch_model_kwargs=arch_model_kwargs,
                        train_x=h_train_x,
                        train_y=h_train_y,
                        train_model_kwargs=h_train_model_kwargs,
                        model_params=model_params,
                        arch_params=arch_params,
                        xi=hessian_penalty_weight,
                        eps=hessian_fd_eps,
                        device=device,
                        use_amp=use_amp,
                    )

                # -- DARTS-PT: perturb model weights toward training minimum ----
                # w' = w - xi * unit(∇_w L_train).  The arch gradient is then
                # computed at w' instead of w, reducing bias from shared weights
                # being positioned sub-optimally relative to the arch optimum.
                darts_pt_originals: list[torch.Tensor] | None = None
                if use_darts_pt and model_params:
                    pt_batch = bilevel_optimizer.next_hessian_batch()
                    pt_train_x, pt_train_y, pt_train_model_kwargs = (
                        unpack_forecasting_batch(
                            pt_batch, device, include_decoder_targets=True
                        )
                    )
                    darts_pt_originals = _apply_darts_pt_perturbation(
                        model=model,
                        model_params=model_params,
                        train_x=pt_train_x,
                        train_y=pt_train_y,
                        train_model_kwargs=pt_train_model_kwargs,
                        loss_fn=loss_fn,
                        xi=darts_pt_xi,
                        device=device,
                        use_amp=use_amp,
                    )

                with autocast_ctx(device, enabled=use_amp):
                    arch_result = compose_architecture_loss(
                        model=model,
                        x=arch_x,
                        y=arch_y,
                        model_kwargs=arch_model_kwargs,
                        loss_fn=loss_fn,
                        regularizer=regularizer,
                        alpha_tracker=trainer.alpha_tracker,
                        arch_params=arch_params,
                        epoch=epoch,
                        epochs=epochs,
                        warmup_epochs=warmup_epochs,
                        device=device,
                        engine_variant=resolved_engine_variant,
                        engine_cfg=engine,
                        state_mix_ortho_reg_weight=state_mix_ortho_reg_weight,
                        beta_darts_weight=beta_darts_weight,
                        edge_diversity_weight=edge_diversity_weight,
                        edge_usage_balance_weight=edge_usage_balance_weight,
                        edge_identity_cap=edge_identity_cap,
                        edge_identity_cap_weight=edge_identity_cap_weight,
                        moe_balance_weight=moe_balance_weight,
                        transformer_exploration_weight=transformer_exploration_weight,
                        edge_sharpening_max_weight=edge_sharpening_max_weight,
                        edge_sharpening_start_frac=edge_sharpening_start_frac,
                    )
                    total_arch_loss = arch_result.loss
                    last_edge_entropy = float(
                        arch_result.edge_entropy.detach().item()
                    )
                    last_edge_sharpen_weight = arch_result.edge_sharpen_weight

                # Restore DARTS-PT perturbation before optimizer step.
                # Gradients are already computed; restoring weights now ensures
                # the model_optimizer step later uses the original weights.
                if darts_pt_originals is not None:
                    _restore_model_params(model_params, darts_pt_originals)

                bilevel_optimizer.step_architecture(
                    total_arch_loss,
                    scaler,
                    already_backward=False,
                    implicit_corrections=implicit_corrections,
                    gradient_balance_params=(
                        model_params
                        if resolved_engine_variant == "r_darts"
                        and engine.r_darts.balance_gradient_norms
                        else None
                    ),
                    gradient_balance_warmup=(
                        engine.r_darts.norm_balance_warmup
                        if resolved_engine_variant == "r_darts"
                        else 0
                    ),
                    gradient_balance_epoch=epoch,
                    arch_grad_scale=(
                        engine.r_darts.arch_grad_scale
                        if resolved_engine_variant == "r_darts"
                        else 1.0
                    ),
                )
                scaler.update()

                if verbose and hasattr(model, "forecast_encoder"):
                    _log_arch_gradients(model)

            bilevel_optimizer.step_scheduler()

            if verbose:
                with torch.no_grad():
                    trainer.alpha_tracker.log_architecture_update_block(
                        model=model,
                        prev_component_probs=prev_component_probs,
                        prev_edge_probs=prev_edge_probs,
                        last_edge_sharpen_weight=last_edge_sharpen_weight,
                        last_edge_entropy=last_edge_entropy,
                        hessian_penalty_weight=hessian_penalty_weight,
                        hessian_penalty=hessian_penalty,
                    )

        # Architecture updates may change edge importance. Resolve the discrete
        # topology once here instead of synchronizing in every model forward.
        _prepare_edge_routing(model)

        # -- Model parameter updates ----------------------------------------
        avg_train_loss = _run_model_training_epoch(
            model=model,
            train_model_loader=train_model_loader,
            model_params=model_params,
            model_optimizer=model_optimizer,
            model_scheduler=model_scheduler,
            scaler=scaler,
            loss_fn=loss_fn,
            gradient_accumulation_steps=gradient_accumulation_steps,
            device=device,
            use_amp=use_amp,
            verbose=verbose,
            epoch=epoch,
            max_batches=max_train_batches,
            engine_variant=resolved_engine_variant,
            engine_cfg=engine,
        )

        avg_val_loss = _run_validation_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            verbose=verbose,
            max_batches=max_val_batches,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        # -- Hybrid pruning --------------------------------------------------
        _maybe_prune(
            model=model,
            epoch=epoch,
            epochs=epochs,
            progressive_shrinking=progressive_shrinking,
            hybrid_pruning_start_epoch=hybrid_pruning_start_epoch,
            hybrid_pruning_interval=hybrid_pruning_interval,
            hybrid_pruning_base_threshold=hybrid_pruning_base_threshold,
            hybrid_pruning_strategy=hybrid_pruning_strategy,
            hybrid_pruning_freeze_logit=hybrid_pruning_freeze_logit,
            verbose=verbose,
        )

        # -- Early stopping --------------------------------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = snapshot_state_dict(model)
            best_progressive_state = capture_progressive_state(model)
            if use_swa and swa_model and epoch >= swa_start:
                swa_model.update_parameters(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                break

        if verbose and hasattr(epoch_pbar, "set_postfix"):
            epoch_pbar.set_postfix(
                {
                    "train": f"{avg_train_loss:.4f}",
                    "val": f"{avg_val_loss:.4f}",
                    "best": f"{best_val_loss:.4f}",
                    "patience": f"{patience_counter}/{patience}",
                    "conc": f"{current_drnas_concentration:.1f}",
                }
            )

    if verbose and hasattr(epoch_pbar, "close"):
        epoch_pbar.close()

    training_time = time.time() - start_time

    # ── SWA finalization ──────────────────────────────────────────────────
    if use_swa and swa_model and epoch >= swa_start:
        if verbose:
            print("\nFinalizing SWA...")
        try:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
            swa_val_loss = trainer._evaluate_model(swa_model, val_loader, loss_type)
            if swa_val_loss < best_val_loss:
                if verbose:
                    print("SWA model is better")
                best_state = snapshot_state_dict(swa_model.module)
                best_progressive_state = capture_progressive_state(model)
                best_val_loss = swa_val_loss
        except Exception as exc:
            if verbose:
                print(f"Warning: SWA failed ({exc})")

    # ── Load best weights ─────────────────────────────────────────────────
    restore_progressive_state(model, best_progressive_state)
    _safe_load_state(model, best_state, verbose=verbose)

    if hasattr(model, "ensure_float32_dtype"):
        model.ensure_float32_dtype()
    else:
        model = model.float()

    # Skip the extra full-val metrics pass when the caller doesn't need it
    # (e.g. phase-3 ASHA rungs, which only read best_val_loss). Saves one
    # validation pass per call across many candidate×rung invocations.
    with _crash_trace("final_metrics", device):
        final_metrics = (
            compute_final_metrics(model, val_loader, device)
            if compute_metrics
            else {"mse": float("nan"), "mae": float("nan")}
        )

    if verbose:
        print(f"\nTraining completed in {training_time:.1f}s")
        print(f"Best Val Loss: {best_val_loss:.6f}")
        print(f"MSE: {final_metrics['mse']:.6f} | MAE: {final_metrics['mae']:.6f}")

    results = {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "alpha_values": alpha_values,
        "diversity_scores": diversity_scores,
        "final_architecture": model,
        "best_val_loss": best_val_loss,
        "training_time": training_time,
        "final_metrics": final_metrics,
    }
    trainer.training_history.append(results)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _run_model_training_epoch(
    *,
    model: nn.Module,
    train_model_loader,
    model_params: list[torch.Tensor],
    model_optimizer,
    model_scheduler,
    scaler: GradScaler,
    loss_fn,
    gradient_accumulation_steps: int,
    device: str,
    use_amp: bool,
    verbose: bool,
    epoch: int,
    max_batches: int | None = None,
    engine_variant: str | None = None,
    engine_cfg: DARTSEngineConfig | None = None,
) -> float:
    """Run one model-parameter epoch and return mean training loss."""
    model.train()
    accumulation_steps = max(int(gradient_accumulation_steps), 1)
    epoch_train_loss = 0.0
    batch_pbar = (
        create_progress_bar(
            enumerate(train_model_loader),
            f"Epoch {epoch + 1:3d}",
            leave=False,
            total=len(train_model_loader),
        )
        if verbose
        else enumerate(train_model_loader)
    )

    model_optimizer.zero_grad()
    batches_seen = 0
    batches_since_step = 0
    for batch_idx, (batch_x, batch_y, *_) in batch_pbar:
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        batches_seen += 1
        batches_since_step += 1
        with _crash_trace(f"epoch {epoch + 1}: train batch {batch_idx} transfer", device):
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

        with _crash_trace(f"epoch {epoch + 1}: train batch {batch_idx} forward", device):
            with autocast_ctx(device, enabled=use_amp):
                if engine_variant == "bi_darts" and engine_cfg is not None:
                    raw_loss = compute_backward_loss(
                        model=model,
                        x=batch_x,
                        y=batch_y,
                        loss_fn=loss_fn,
                        backward_loss_weight=engine_cfg.bi_darts.backward_loss_weight,
                        backward_passes=engine_cfg.bi_darts.backward_passes,
                    )
                else:
                    raw_loss = loss_fn(model(batch_x), batch_y)
                loss = raw_loss / accumulation_steps

        with _crash_trace(f"epoch {epoch + 1}: train batch {batch_idx} backward", device):
            scaler.scale(loss).backward()

        if batches_since_step >= accumulation_steps:
            with _crash_trace(
                f"epoch {epoch + 1}: train batch {batch_idx} optimizer_step",
                device,
            ):
                scaler.unscale_(model_optimizer)
                torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)
                scaler.step(model_optimizer)
                scaler.update()
                model_scheduler.step()
                model_optimizer.zero_grad()
                batches_since_step = 0

        batch_loss = float(loss.detach()) * accumulation_steps
        epoch_train_loss += batch_loss

        if verbose and hasattr(batch_pbar, "set_postfix"):
            batch_pbar.set_postfix(
                {
                    "loss": f"{batch_loss:.4f}",
                    "avg": f"{epoch_train_loss / (batch_idx + 1):.4f}",
                }
            )

    if verbose and hasattr(batch_pbar, "close"):
        batch_pbar.close()

    # Do not discard gradients when the epoch/batch cap ends between complete
    # accumulation windows.
    if batches_since_step > 0:
        with _crash_trace(f"epoch {epoch + 1}: final optimizer_step", device):
            scaler.unscale_(model_optimizer)
            # Losses were divided by the configured window size. Correct the
            # smaller final window so it remains an average of its own batches.
            partial_scale = accumulation_steps / batches_since_step
            for param in model_params:
                if param.grad is not None:
                    param.grad.mul_(partial_scale)
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)
            scaler.step(model_optimizer)
            scaler.update()
            model_scheduler.step()
            model_optimizer.zero_grad()

    return epoch_train_loss / max(batches_seen, 1)


def _run_validation_epoch(
    *,
    model: nn.Module,
    val_loader,
    loss_fn,
    device: str,
    use_amp: bool,
    verbose: bool,
    max_batches: int | None = None,
) -> float:
    """Run one validation epoch and return mean validation loss."""
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        val_pbar = (
            create_progress_bar(val_loader, "Val", leave=False)
            if verbose
            else val_loader
        )
        batches_seen = 0
        for batch_idx, batch_data in enumerate(val_pbar):
            if max_batches is not None and batch_idx >= int(max_batches):
                break
            batches_seen += 1
            with _crash_trace(f"val batch {batch_idx} transfer", device):
                x = batch_data[0].to(device, non_blocking=True)
                y = batch_data[1].to(device, non_blocking=True)
            with _crash_trace(f"val batch {batch_idx} forward", device):
                with autocast_ctx(device, enabled=use_amp):
                    val_loss += loss_fn(model(x), y).item()
            if verbose and hasattr(val_pbar, "set_postfix"):
                val_pbar.set_postfix(
                    {"val_loss": f"{val_loss / max(batches_seen, 1):.4f}"}
                )
        if verbose and hasattr(val_pbar, "close"):
            val_pbar.close()

    return val_loss / max(batches_seen, 1)
