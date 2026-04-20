"""
DARTS bilevel training loop.

Extracted from ``DARTSTrainer`` so the logic lives in one focused module.
The public entry-point is :func:`train_darts_model`.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

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
    regularization_types: Optional[List[str]] = None,
    regularization_weights: Optional[List[float]] = None,
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
    use_gdas: bool = False,
) -> Dict[str, Any]:
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
    loss_fn = trainer._get_loss_function(loss_type)
    model = model.to(device)
    start_time = time.time()

    # Propagate use_gdas to all MixedOp edges so callers can override the
    # model-level default without reconstructing the model.
    if use_gdas:
        for m in model.modules():
            if hasattr(m, "use_gdas"):
                m.use_gdas = True

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
    try:
        # Architecture params: Adam (betas=(0.5, 0.999) per DARTS paper).
        # Model params: AdamW — decoupled weight decay is strictly better for
        # over-parameterised models and consistent with the fallback path.
        arch_optimizer = torch.optim.Adam(
            arch_param_groups,
            betas=(0.5, 0.999),
            weight_decay=arch_weight_decay,
            fused=True,
        )
        model_optimizer = torch.optim.AdamW(
            model_params,
            lr=model_learning_rate,
            weight_decay=model_weight_decay,
            fused=True,
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

    # ── Schedulers ────────────────────────────────────────────────────────
    arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        arch_optimizer, T_max=epochs, eta_min=arch_learning_rate * 0.01
    )
    model_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        model_optimizer,
        max_lr=model_learning_rate,
        epochs=epochs,
        steps_per_epoch=max(
            1, len(train_loader) // max(1, gradient_accumulation_steps)
        ),
        pct_start=0.3,
        anneal_strategy="cos",
    )

    # ── Bilevel split ─────────────────────────────────────────────────────
    train_arch_loader = None
    if use_bilevel_optimization:
        train_arch_loader, train_model_loader = trainer._create_bilevel_loaders(
            train_loader, seed=bilevel_split_seed
        )
    else:
        train_model_loader = train_loader

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
    prev_component_probs: Dict = {}
    prev_edge_probs: Dict = {}
    last_edge_entropy = float("nan")
    last_edge_sharpen_weight = 0.0
    last_moe_balance_loss = float("nan")
    last_transformer_exploration_bonus = float("nan")

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

        if epoch % 5 == 0:
            alpha_values.append(trainer.alpha_tracker.extract_alpha_values(model))

        # -- Architecture parameter updates ----------------------------------
        _arch_freq = _dynamic_arch_update_freq(
            epoch, epochs, warmup_epochs, architecture_update_freq
        )
        _arch_iters = _dynamic_inner_arch_iters(epoch, epochs, warmup_epochs)
        if epoch >= warmup_epochs and epoch % _arch_freq == 0:
            for _ in range(_arch_iters):
                arch_batch = bilevel_optimizer.next_arch_batch()
                arch_x, arch_y, arch_model_kwargs = unpack_forecasting_batch(
                    arch_batch,
                    device,
                    include_decoder_targets=True,
                )
                bilevel_optimizer.zero_arch_grads()

                # -- Implicit arch gradient correction (second-order DARTS) ----
                # Replaces the finite-difference curvature *penalty* with a
                # direct correction applied to arch param gradients after
                # backward.  Correction = -xi/(2ε) * (∇_α L_val(w+) - ∇_α L_val(w-))
                # where w± are model weights perturbed along ∇_w L_train.
                implicit_corrections: Optional[List[Optional[torch.Tensor]]] = None
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
                darts_pt_originals: Optional[List[torch.Tensor]] = None
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
                    arch_preds = model(arch_x, **arch_model_kwargs)
                    arch_loss = loss_fn(arch_preds, arch_y)

                    reg_losses = regularizer.compute_regularization(
                        model, arch_params, epoch, epochs
                    )
                    total_arch_loss = arch_loss + reg_losses["total"]

                    # Orthogonal state-mixing regularization
                    ortho_reg = torch.tensor(0.0, device=device)
                    if state_mix_ortho_reg_weight > 0.0 and hasattr(
                        model, "get_orthogonal_regularization"
                    ):
                        ortho_reg = model.get_orthogonal_regularization()
                        total_arch_loss = (
                            total_arch_loss
                            + float(state_mix_ortho_reg_weight) * ortho_reg
                        )

                    # β-DARTS: L2 regularization on raw arch logits.
                    # Penalising large logit magnitudes prevents any single
                    # operation from dominating early in search (a.k.a.
                    # skip-connection collapse).  Inspired by Ye et al.
                    # "β-DARTS: β-Decay Regularization for Differentiable
                    # Architecture Search", NeurIPS 2022.
                    if beta_darts_weight > 0.0 and arch_params:
                        beta_reg = beta_darts_weight * sum(
                            p.pow(2).mean() for p in arch_params
                        )
                        total_arch_loss = total_arch_loss + beta_reg

                    # Edge diversity regularization
                    total_arch_loss, edge_diversity_pairs = _add_edge_diversity_reg(
                        model=model,
                        total_arch_loss=total_arch_loss,
                        edge_diversity_weight=edge_diversity_weight,
                        edge_usage_balance_weight=edge_usage_balance_weight,
                        edge_identity_cap=edge_identity_cap,
                        edge_identity_cap_weight=edge_identity_cap_weight,
                        device=device,
                    )

                    moe_balance_loss = torch.tensor(0.0, device=device)
                    if moe_balance_weight > 0.0 and hasattr(
                        model, "get_moe_balance_loss"
                    ):
                        moe_balance_loss = model.get_moe_balance_loss()
                        total_arch_loss = (
                            total_arch_loss
                            + float(moe_balance_weight) * moe_balance_loss
                        )
                    last_moe_balance_loss = float(moe_balance_loss.detach().item())

                    transformer_exploration_bonus = torch.tensor(0.0, device=device)
                    if transformer_exploration_weight > 0.0:
                        entropy_terms: List[torch.Tensor] = []
                        for source in trainer.alpha_tracker.component_alpha_sources(model):
                            name = str(source.get("name", ""))
                            if not (
                                name.startswith("encoder_")
                                or name.startswith("decoder_")
                                or name in {"decoder_style", "decoder_memory_queries"}
                            ):
                                continue
                            alpha = source.get("alpha")
                            if not isinstance(alpha, torch.Tensor) or alpha.numel() <= 1:
                                continue
                            probs = F.softmax(alpha, dim=0)
                            entropy_terms.append(
                                -(probs * torch.log(probs.clamp_min(1e-8))).sum()
                            )
                        if entropy_terms:
                            transformer_exploration_bonus = torch.stack(
                                entropy_terms
                            ).mean()
                            if epochs > warmup_epochs:
                                progress = float(
                                    max(epoch - warmup_epochs, 0)
                                ) / float(max(epochs - warmup_epochs, 1))
                            else:
                                progress = float(epoch) / float(max(epochs, 1))
                            early_phase_scale = max(0.0, 1.0 - progress)
                            total_arch_loss = (
                                total_arch_loss
                                - float(transformer_exploration_weight)
                                * early_phase_scale
                                * transformer_exploration_bonus
                            )
                    last_transformer_exploration_bonus = float(
                        transformer_exploration_bonus.detach().item()
                    )

                    # Late-phase edge-entropy sharpening
                    total_arch_loss, edge_entropy, edge_sharpen_weight = (
                        _add_edge_sharpening(
                            model=model,
                            total_arch_loss=total_arch_loss,
                            epoch=epoch,
                            epochs=epochs,
                            warmup_epochs=warmup_epochs,
                            edge_sharpening_max_weight=edge_sharpening_max_weight,
                            edge_sharpening_start_frac=edge_sharpening_start_frac,
                            device=device,
                        )
                    )

                    last_edge_entropy = float(edge_entropy.detach().item())
                    last_edge_sharpen_weight = float(edge_sharpen_weight)

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
        )

        avg_val_loss = _run_validation_epoch(
            model=model,
            val_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
            use_amp=use_amp,
            verbose=verbose,
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
            best_state = {
                k: v.detach().clone().float() for k, v in model.state_dict().items()
            }
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
                best_state = {
                    k: v.detach().clone() for k, v in swa_model.state_dict().items()
                }
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

    final_metrics = compute_final_metrics(model, val_loader, device)

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


def _run_model_training_epoch(
    *,
    model: nn.Module,
    train_model_loader,
    model_params: List[torch.Tensor],
    model_optimizer,
    model_scheduler,
    scaler: GradScaler,
    loss_fn,
    gradient_accumulation_steps: int,
    device: str,
    use_amp: bool,
    verbose: bool,
    epoch: int,
) -> float:
    """Run one model-parameter epoch and return mean training loss."""
    model.train()
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
    for batch_idx, (batch_x, batch_y, *_) in batch_pbar:
        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        with autocast_ctx(device, enabled=use_amp):
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y) / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(model_optimizer)
            torch.nn.utils.clip_grad_norm_(model_params, max_norm=5.0)
            scaler.step(model_optimizer)
            scaler.update()
            model_scheduler.step()
            model_optimizer.zero_grad()

        epoch_train_loss += loss.item() * gradient_accumulation_steps

        if verbose and hasattr(batch_pbar, "set_postfix"):
            batch_pbar.set_postfix(
                {
                    "loss": f"{loss.item() * gradient_accumulation_steps:.4f}",
                    "avg": f"{epoch_train_loss / (batch_idx + 1):.4f}",
                }
            )

    if verbose and hasattr(batch_pbar, "close"):
        batch_pbar.close()

    return epoch_train_loss / max(len(train_model_loader), 1)


def _run_validation_epoch(
    *,
    model: nn.Module,
    val_loader,
    loss_fn,
    device: str,
    use_amp: bool,
    verbose: bool,
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
        for batch_data in val_pbar:
            x = batch_data[0].to(device, non_blocking=True)
            y = batch_data[1].to(device, non_blocking=True)
            with autocast_ctx(device, enabled=use_amp):
                val_loss += loss_fn(model(x), y).item()
            if verbose and hasattr(val_pbar, "set_postfix"):
                val_pbar.set_postfix(
                    {"val_loss": f"{val_loss / max(len(val_loader), 1):.4f}"}
                )
        if verbose and hasattr(val_pbar, "close"):
            val_pbar.close()

    return val_loss / max(len(val_loader), 1)


def _apply_darts_pt_perturbation(
    *,
    model: nn.Module,
    model_params: List[torch.Tensor],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: Optional[Dict[str, Any]],
    loss_fn,
    xi: float,
    device: str,
    use_amp: bool,
) -> List[torch.Tensor]:
    """Perturb model weights by ``-xi * unit(∇_w L_train)`` (DARTS-PT step).

    Returns the list of original parameter tensors so the caller can restore
    them via :func:`_restore_model_params` after the architecture backward pass.

    Moving weights toward the training loss minimum before evaluating the
    validation loss reduces the coupling bias that arises when shared weights
    are sub-optimally positioned for the current architecture distribution.
    """
    train_model_kwargs = dict(train_model_kwargs or {})

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    norm_sq = torch.tensor(0.0, device=device)
    for g in grads:
        if g is not None:
            norm_sq = norm_sq + g.pow(2).sum()
    norm = norm_sq.sqrt().clamp_min(1e-12)

    with torch.no_grad():
        for p, g in zip(model_params, grads):
            if g is not None:
                p.add_(g.detach() / norm, alpha=-float(xi))

    return originals


def _restore_model_params(
    model_params: List[torch.Tensor],
    originals: List[torch.Tensor],
) -> None:
    """Restore model parameters to their pre-perturbation values."""
    with torch.no_grad():
        for p, orig in zip(model_params, originals):
            p.copy_(orig)


def compute_implicit_arch_gradient_correction(
    *,
    model: nn.Module,
    loss_fn,
    arch_x: torch.Tensor,
    arch_y: torch.Tensor,
    arch_model_kwargs: Optional[Dict[str, Any]],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: Optional[Dict[str, Any]],
    model_params: List[torch.Tensor],
    arch_params: List[torch.Tensor],
    xi: float,
    eps: float,
    device: str,
    use_amp: bool,
) -> List[Optional[torch.Tensor]]:
    """Compute second-order implicit arch gradient correction.

    Implements the DARTS second-order approximation via finite differences in
    model-weight space.  Instead of adding a scalar curvature *penalty* to the
    arch loss, this function returns per-parameter gradient *corrections* that
    are applied directly to ``p.grad`` after the main backward pass
    (see ``BilevelOptimizer.step_architecture``).

    Correction:
        Δ∇_α = -xi / (2ε) * (∇_α L_val(w⁺) − ∇_α L_val(w⁻))

    where ``w± = w ± ε · unit(∇_w L_train)``.

    This removes the need for a tuned penalty weight (beyond ``xi``, which is
    already the model learning-rate scale) and avoids the instability of
    relu-clipped curvature scalars in noisy training regimes.
    """
    if not model_params or not arch_params or eps <= 0:
        return [None] * len(arch_params)

    arch_model_kwargs = dict(arch_model_kwargs or {})
    train_model_kwargs = dict(train_model_kwargs or {})

    # --- Training gradient direction ---
    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads_w = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    direction = [
        torch.zeros_like(p) if g is None else g.detach()
        for p, g in zip(model_params, grads_w)
    ]
    norm = torch.sqrt(sum(d.pow(2).sum() for d in direction)).clamp_min(1e-12)
    # step size in the unit-gradient direction
    scale = float(eps) / float(norm)

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    corrections: List[Optional[torch.Tensor]] = [None] * len(arch_params)
    try:
        # ∇_α L_val(w⁺)
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(d, alpha=scale)
        with autocast_ctx(device, enabled=use_amp):
            loss_plus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
        grads_plus = torch.autograd.grad(
            loss_plus,
            arch_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # ∇_α L_val(w⁻)
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(d, alpha=-2.0 * scale)
        with autocast_ctx(device, enabled=use_amp):
            loss_minus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
        grads_minus = torch.autograd.grad(
            loss_minus,
            arch_params,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )

        # Δ∇_α = -xi / (2ε) * (∇_α(w⁺) − ∇_α(w⁻))
        # The 2ε denominator corresponds to the actual step size (eps in unit-grad space).
        factor = -float(xi) / (2.0 * float(eps))
        for i, (gp, gm) in enumerate(zip(grads_plus, grads_minus)):
            if gp is not None and gm is not None:
                corrections[i] = (factor * (gp - gm)).detach()
            elif gp is not None:
                corrections[i] = (factor * gp).detach()
    finally:
        with torch.no_grad():
            for p, orig in zip(model_params, originals):
                p.copy_(orig)

    return corrections


def finite_difference_hessian_penalty(
    *,
    model: nn.Module,
    loss_fn,
    arch_loss: torch.Tensor,
    arch_x: torch.Tensor,
    arch_y: torch.Tensor,
    arch_model_kwargs: Optional[Dict[str, Any]],
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    train_model_kwargs: Optional[Dict[str, Any]],
    model_params: List[torch.Tensor],
    device: str,
    eps: float = 1e-2,
    use_amp: bool = True,
) -> torch.Tensor:
    """
    Finite-difference curvature proxy used to penalise sharp architecture
    landscapes (from DARTS+ / SDARTS literature).
    """
    if eps <= 0 or not model_params:
        return torch.tensor(0.0, device=arch_x.device)

    arch_model_kwargs = dict(arch_model_kwargs or {})
    train_model_kwargs = dict(train_model_kwargs or {})

    with autocast_ctx(device, enabled=use_amp):
        train_loss = loss_fn(model(train_x, **train_model_kwargs), train_y)

    grads = torch.autograd.grad(
        train_loss,
        model_params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    direction, norm_sq = [], torch.tensor(0.0, device=arch_x.device)
    for p, g in zip(model_params, grads):
        d = torch.zeros_like(p) if g is None else g.detach()
        direction.append(d)
        norm_sq = norm_sq + d.pow(2).sum()

    norm = torch.sqrt(norm_sq).clamp_min(1e-12)
    scale = float(eps) / norm

    with torch.no_grad():
        originals = [p.detach().clone() for p in model_params]

    try:
        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(scale * d)
        with autocast_ctx(device, enabled=use_amp):
            loss_plus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)

        with torch.no_grad():
            for p, d in zip(model_params, direction):
                p.add_(-2.0 * scale * d)
        with autocast_ctx(device, enabled=use_amp):
            loss_minus = loss_fn(model(arch_x, **arch_model_kwargs), arch_y)
    finally:
        with torch.no_grad():
            for p, orig in zip(model_params, originals):
                p.copy_(orig)

    curvature = (loss_plus + loss_minus - 2.0 * arch_loss.detach()) / (eps**2)
    return F.relu(curvature)


def _add_edge_diversity_reg(
    *,
    model: nn.Module,
    total_arch_loss: torch.Tensor,
    edge_diversity_weight: float,
    edge_usage_balance_weight: float,
    edge_identity_cap: float,
    edge_identity_cap_weight: float,
    device: str,
) -> tuple:
    """Compute and add edge diversity + identity-cap regularisation terms."""
    edge_diversity_loss = torch.tensor(0.0, device=device)
    edge_diversity_pairs = 0
    edge_usage_balance_loss = torch.tensor(0.0, device=device)
    edge_usage_cells = 0
    edge_identity_cap_loss = torch.tensor(0.0, device=device)
    edge_identity_cells = 0

    for cell in getattr(model, "cells", []):
        if not hasattr(cell, "edges"):
            continue

        edge_probs_by_name = []
        union_op_names: List[str] = []

        for edge in cell.edges:
            probs = _extract_edge_probs(edge)
            if (
                probs is None
                or probs.numel() <= 1
                or not hasattr(edge, "available_ops")
                or len(edge.available_ops) != probs.numel()
            ):
                continue

            probs = probs.clamp_min(1e-8)
            probs = probs / probs.sum().clamp_min(1e-8)
            prob_map = {}
            for op_idx, op_name in enumerate(edge.available_ops):
                prob_map[op_name] = probs[op_idx]
                if op_name not in union_op_names:
                    union_op_names.append(op_name)
            edge_probs_by_name.append(prob_map)

        if len(edge_probs_by_name) < 2 or len(union_op_names) <= 1:
            continue

        base_zero = edge_probs_by_name[0][union_op_names[0]].new_tensor(0.0)
        aligned: List[torch.Tensor] = []
        for prob_map in edge_probs_by_name:
            vec = torch.stack(
                [prob_map.get(n, base_zero) for n in union_op_names], dim=0
            )
            vec = vec / vec.sum().clamp_min(1e-8)
            aligned.append(vec)

        for i in range(len(aligned)):
            vi = aligned[i] / aligned[i].norm(p=2).clamp_min(1e-8)
            for j in range(i + 1, len(aligned)):
                vj = aligned[j] / aligned[j].norm(p=2).clamp_min(1e-8)
                edge_diversity_loss = edge_diversity_loss + torch.dot(vi, vj)
                edge_diversity_pairs += 1

        mean_probs = torch.stack(aligned, dim=0).mean(dim=0)
        mean_probs = mean_probs / mean_probs.sum().clamp_min(1e-8)
        entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum()
        norm_entropy = entropy / np.log(max(mean_probs.numel(), 2))
        edge_usage_balance_loss = edge_usage_balance_loss + (1.0 - norm_entropy)
        edge_usage_cells += 1

        if "Identity" in union_op_names and edge_identity_cap < 1.0:
            id_idx = union_op_names.index("Identity")
            id_prob = mean_probs[id_idx]
            edge_identity_cap_loss = edge_identity_cap_loss + F.relu(
                id_prob - float(edge_identity_cap)
            )
            edge_identity_cells += 1

    if edge_diversity_pairs > 0:
        total_arch_loss = total_arch_loss + edge_diversity_weight * (
            edge_diversity_loss / edge_diversity_pairs
        )
    if edge_usage_cells > 0:
        total_arch_loss = total_arch_loss + edge_usage_balance_weight * (
            edge_usage_balance_loss / edge_usage_cells
        )
    if edge_identity_cells > 0 and edge_identity_cap_weight > 0:
        total_arch_loss = total_arch_loss + edge_identity_cap_weight * (
            edge_identity_cap_loss / edge_identity_cells
        )

    return total_arch_loss, edge_diversity_pairs


def _add_edge_sharpening(
    *,
    model: nn.Module,
    total_arch_loss: torch.Tensor,
    epoch: int,
    epochs: int,
    warmup_epochs: int,
    edge_sharpening_max_weight: float,
    edge_sharpening_start_frac: float,
    device: str,
):
    """Add late-phase entropy sharpening to encourage decisive operation choice."""
    edge_entropy = torch.tensor(0.0, device=device)
    edge_sharpen_weight = 0.0

    if edge_sharpening_max_weight <= 0 or epoch < warmup_epochs:
        return total_arch_loss, edge_entropy, edge_sharpen_weight

    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    if progress < edge_sharpening_start_frac:
        return total_arch_loss, edge_entropy, edge_sharpen_weight

    ramp = (progress - edge_sharpening_start_frac) / max(
        1e-8, 1.0 - edge_sharpening_start_frac
    )
    edge_sharpen_weight = edge_sharpening_max_weight * min(1.0, max(0.0, ramp))

    entropy_terms = []
    for cell in getattr(model, "cells", []):
        if not hasattr(cell, "edges"):
            continue
        for edge in cell.edges:
            if (
                hasattr(edge, "use_hierarchical")
                and edge.use_hierarchical
                and hasattr(edge, "group_alphas")
            ):
                gt = max(
                    float(
                        getattr(
                            edge, "group_temperature", getattr(edge, "temperature", 1.0)
                        )
                    ),
                    1e-6,
                )
                g_probs = F.softmax(edge.group_alphas / gt, dim=0)
                g_ent = -(g_probs * torch.log(g_probs + 1e-8)).sum() / np.log(
                    max(g_probs.numel(), 2)
                )
                entropy_terms.append(g_ent)
                if hasattr(edge, "op_alphas"):
                    ot = max(
                        float(
                            getattr(
                                edge,
                                "op_temperature",
                                getattr(edge, "temperature", 1.0),
                            )
                        ),
                        1e-6,
                    )
                    for alpha in edge.op_alphas.values():
                        o_probs = F.softmax(alpha / ot, dim=0)
                        o_ent = -(o_probs * torch.log(o_probs + 1e-8)).sum() / np.log(
                            max(o_probs.numel(), 2)
                        )
                        entropy_terms.append(o_ent)
            elif hasattr(edge, "_alphas"):
                ot = max(
                    float(
                        getattr(
                            edge, "op_temperature", getattr(edge, "temperature", 1.0)
                        )
                    ),
                    1e-6,
                )
                probs = F.softmax(edge._alphas / ot, dim=0)
                ent = -(probs * torch.log(probs + 1e-8)).sum() / np.log(
                    max(probs.numel(), 2)
                )
                entropy_terms.append(ent)

    if entropy_terms:
        edge_entropy = torch.stack(entropy_terms).mean()
        total_arch_loss = total_arch_loss + edge_sharpen_weight * edge_entropy

    return total_arch_loss, edge_entropy, edge_sharpen_weight


def _extract_edge_probs(edge) -> Optional[torch.Tensor]:
    """Return per-operation probability vector for a single DARTS edge."""
    if (
        hasattr(edge, "use_hierarchical")
        and edge.use_hierarchical
        and hasattr(edge, "_get_weights")
        and hasattr(edge, "ops")
    ):
        try:
            routed = edge._get_weights(top_k=None)
            if routed:
                probs = torch.zeros(
                    len(edge.ops),
                    device=routed[0][1].device,
                    dtype=routed[0][1].dtype,
                )
                for op_idx, weight in routed:
                    probs[op_idx] = probs[op_idx] + weight
                return probs
        except Exception:
            pass
    if hasattr(edge, "_alphas"):
        temp = max(
            float(getattr(edge, "op_temperature", getattr(edge, "temperature", 1.0))),
            1e-6,
        )
        return F.softmax(edge._alphas / temp, dim=0)
    return None


def _maybe_prune(
    *,
    model,
    epoch,
    epochs,
    progressive_shrinking,
    hybrid_pruning_start_epoch,
    hybrid_pruning_interval,
    hybrid_pruning_base_threshold,
    hybrid_pruning_strategy,
    hybrid_pruning_freeze_logit,
    verbose,
) -> None:
    """Apply hybrid pruning schedule when criteria are met."""
    should_prune = (
        progressive_shrinking
        and epoch > int(hybrid_pruning_start_epoch)
        and int(hybrid_pruning_interval) > 0
        and epoch % int(hybrid_pruning_interval) == 0
    )
    if not should_prune or not hasattr(model, "prune_weak_operations"):
        return

    threshold = float(hybrid_pruning_base_threshold) * (
        float(epoch) / float(max(epochs, 1))
    )
    threshold = min(max(threshold, 0.0), 0.95)
    pruning_stats = model.prune_weak_operations(
        threshold=threshold, strategy=hybrid_pruning_strategy
    )
    frozen = 0
    if hasattr(model, "freeze_pruned_operations"):
        frozen = int(
            model.freeze_pruned_operations(
                pruning_stats=pruning_stats,
                logit_value=hybrid_pruning_freeze_logit,
            )
        )
    if verbose:
        print(
            f"  [Hybrid Prune] epoch={epoch + 1}/{epochs} "
            f"threshold={threshold:.3f} "
            f"pruned={int(pruning_stats.get('operations_pruned', 0))} "
            f"frozen={frozen}"
        )


def _safe_load_state(model: nn.Module, best_state: dict, *, verbose: bool) -> None:
    """Load best_state with graceful fallback for shape mismatches."""
    try:
        model.load_state_dict(best_state)
    except RuntimeError:
        current = model.state_dict()
        filtered = {
            k: v
            for k, v in best_state.items()
            if not k.startswith(("_forecast_buffer", "_context_buffer"))
            and k in current
            and current[k].shape == v.shape
        }
        if filtered:
            if verbose:
                dropped = len(best_state) - len(filtered)
                print(f"Warning: partial checkpoint load ({dropped} tensors skipped)")
            model.load_state_dict(filtered, strict=False)


def _log_arch_gradients(model: nn.Module) -> None:
    """Log encoder/decoder architecture gradient stats when both exist."""
    enc = getattr(model.forecast_encoder, "alphas", None)
    dec = getattr(model.forecast_decoder, "alphas", None)
    if enc is None or dec is None:
        return
    enc_gn = enc.grad.norm().item() if enc.grad is not None else float("nan")
    dec_gn = dec.grad.norm().item() if dec.grad is not None else float("nan")
    ev, dv = enc.detach().view(-1), dec.detach().view(-1)
    if ev.shape == dv.shape:
        cos = F.cosine_similarity(ev, dv, dim=0).item()
        cos_str = f"{cos:.4f}"
    else:
        cos_str = f"n/a(enc={ev.numel()},dec={dv.numel()})"
    print(
        f"  [Arch Grad] enc={enc_gn:.6e}, dec={dec_gn:.6e}, cos={cos_str}, "
        f"shared={enc is dec}"
    )
