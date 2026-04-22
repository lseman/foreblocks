"""
Fixed-architecture final model training.

After DARTS search and architecture derivation, this module trains the
discrete model to convergence.  The public entry-point is
:func:`train_final_model`.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler

from ..evaluation.metrics import compute_metrics
from ..utils.io import print_final_results
from ..utils.training import autocast_ctx, create_progress_bar, unpack_forecasting_batch

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def train_final_model(
    trainer,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    *,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 50,
    loss_type: str = "huber",
    use_onecycle: bool = True,
    swa_start_ratio: float = 0.33,
    grad_clip_norm: float = 1.0,
    use_amp: bool = True,
) -> dict[str, Any]:
    """
    Train the *fixed-operation* model resulting from architecture derivation.

    Args:
        trainer:         DARTSTrainer instance (for ``device`` and history).
        model:           Fixed model (operations already discrete).
        train_loader:    Training DataLoader.
        val_loader:      Validation DataLoader.
        test_loader:     Test DataLoader used for final evaluation.
        epochs / ...     See :class:`~darts.config.FinalTrainConfig`.

    Returns:
        Dict with keys: ``model``, ``train_losses``, ``val_losses``,
        ``test_loss``, ``training_time``, ``final_metrics``,
        ``training_info``.
    """
    device = trainer.device
    loss_fn = trainer._get_loss_function(loss_type)
    model = model.to(device)
    start_time = time.time()

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # ── Scheduler ─────────────────────────────────────────────────────────
    if use_onecycle:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1000,
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── SWA ───────────────────────────────────────────────────────────────
    # Lazily instantiated at swa_start to avoid holding a second full copy
    # of the model in VRAM for the entire run (especially costly when early
    # stopping triggers well before swa_start).
    swa_model = None
    _amp_enabled = use_amp and device.startswith("cuda")
    scaler = GradScaler(enabled=_amp_enabled)
    swa_start = int(epochs * swa_start_ratio)

    # ── State ─────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state: dict = {}
    train_losses, val_losses = [], []

    print(f"Training final model for {epochs} epochs")
    print(f"  LR: {learning_rate}, weight_decay: {weight_decay}")
    print(
        f"  Scheduler: {'OneCycle' if use_onecycle else 'CosineAnnealing'}, "
        f"SWA starts: epoch {swa_start}, patience: {patience}"
    )
    print("-" * 70)

    epoch_pbar = create_progress_bar(range(epochs), "Final Training", unit="epoch")

    epoch = 0
    for epoch in epoch_pbar:
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        train_pbar = create_progress_bar(
            train_loader,
            f"Epoch {epoch + 1:3d} Train",
            leave=False,
            total=len(train_loader),
        )

        for batch_idx, batch in enumerate(train_pbar):
            batch_x, batch_y, model_kwargs = unpack_forecasting_batch(
                batch,
                device,
                include_decoder_targets=True,
            )
            optimizer.zero_grad()

            with autocast_ctx(device, enabled=_amp_enabled):
                preds = model(batch_x, **model_kwargs)
                loss = loss_fn(preds, batch_y)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            if use_onecycle:
                scheduler.step()

            epoch_train_loss += loss.item()
            train_pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "avg": f"{epoch_train_loss / (batch_idx + 1):.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                }
            )

        train_pbar.close()
        if not use_onecycle:
            scheduler.step()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation phase
        avg_val_loss = trainer._evaluate_model(model, val_loader, loss_type)
        val_losses.append(avg_val_loss)

        # SWA update — instantiate lazily to avoid holding a full GPU copy
        # for the entire run when early stopping fires before swa_start.
        swa_updated = epoch >= swa_start
        if swa_updated:
            if swa_model is None:
                swa_model = torch.optim.swa_utils.AveragedModel(model).to(device)
            swa_model.update_parameters(model)

        postfix = {
            "train": f"{avg_train_loss:.4f}",
            "val": f"{avg_val_loss:.4f}",
            "best_val": f"{best_val_loss:.4f}",
            "patience": f"{patience_counter}/{patience}",
        }
        if swa_updated:
            postfix["swa"] = "ok"
        epoch_pbar.set_postfix(postfix)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Move directly to CPU: saves a full-param VRAM copy competing
            # with the training batch for the rest of the run.
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                epoch_pbar.set_description(f"Early stop @ epoch {epoch + 1}")
                break

    epoch_pbar.close()

    # ── SWA finalisation ──────────────────────────────────────────────────
    swa_used = _finalize_swa(
        trainer=trainer,
        model=model,
        swa_model=swa_model,
        val_loader=val_loader,
        train_loader=train_loader,
        epoch=epoch,
        swa_start=swa_start,
        loss_type=loss_type,
        best_val_loss=best_val_loss,
        best_state=best_state,
    )
    if swa_used and any(k.startswith("module.") for k in best_state):
        best_state = {k.replace("module.", "", 1): v for k, v in best_state.items()}

    # ── Final evaluation ──────────────────────────────────────────────────
    model.load_state_dict(best_state, strict=False)
    test_results = _evaluate_test_set(
        trainer, model, test_loader, loss_type, amp_enabled=_amp_enabled
    )
    training_time = time.time() - start_time

    results = {
        "model": model,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "test_loss": test_results["test_loss"],
        "training_time": training_time,
        "final_metrics": test_results["metrics"],
        "training_info": {
            "epochs_completed": epoch + 1,
            "swa_used": swa_used,
            "final_lr": optimizer.param_groups[0]["lr"],
            "best_val_loss": best_val_loss,
        },
    }

    print_final_results(results)
    trainer.training_history.append(results)
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _finalize_swa(
    *,
    trainer,
    model: nn.Module,
    swa_model,
    val_loader,
    train_loader,
    epoch: int,
    swa_start: int,
    loss_type: str,
    best_val_loss: float,
    best_state: dict,
) -> bool:
    """Finalise SWA and update *best_state* if SWA is better. Returns bool."""
    if epoch < swa_start:
        return False

    print("\nFinalizing SWA model...")
    device = trainer.device

    try:
        bn_pbar = create_progress_bar(train_loader, "Updating BN", leave=False)
        torch.optim.swa_utils.update_bn(bn_pbar, swa_model, device=device)
        bn_pbar.close()
    except Exception as exc:
        print(f"BN update failed ({exc}), using fallback...")
        swa_model.train()
        with torch.no_grad():
            for batch in create_progress_bar(train_loader, "Fallback BN", leave=False):
                batch_x, _, model_kwargs = unpack_forecasting_batch(
                    batch,
                    device,
                    include_decoder_targets=False,
                    teacher_forcing_ratio=0.0,
                )
                swa_model(batch_x, **model_kwargs)

    swa_val_loss = trainer._evaluate_model(swa_model, val_loader, loss_type)
    print(f"SWA val loss: {swa_val_loss:.6f} vs best: {best_val_loss:.6f}")

    if swa_val_loss < best_val_loss:
        print("Using SWA model (better).")
        best_state.update(
            {k: v.cpu().clone() for k, v in swa_model.state_dict().items()}
        )
        return True

    print("Keeping original best model.")
    return False


def _evaluate_test_set(
    trainer, model: nn.Module, test_loader, loss_type: str, *, amp_enabled: bool = False
) -> dict:
    """Evaluate model on the test set and return loss + metrics dict."""
    print("\nEvaluating on test set...")
    device = trainer.device
    loss_fn = trainer._get_loss_function(loss_type)
    model.eval()

    test_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        test_pbar = create_progress_bar(test_loader, "Test Evaluation")
        for batch in test_pbar:
            batch_x, batch_y, model_kwargs = unpack_forecasting_batch(
                batch,
                device,
                include_decoder_targets=False,
                teacher_forcing_ratio=0.0,
            )
            with autocast_ctx(device, enabled=amp_enabled):
                preds = model(batch_x, **model_kwargs)
                batch_loss = loss_fn(preds, batch_y).item()
            test_loss += batch_loss
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_y.cpu().numpy())
            test_pbar.set_postfix({"test_loss": f"{batch_loss:.4f}"})
        test_pbar.close()

    test_loss /= len(test_loader)
    preds_flat = np.concatenate(all_preds).reshape(-1)
    targets_flat = np.concatenate(all_targets).reshape(-1)
    return {
        "test_loss": test_loss,
        "metrics": compute_metrics(preds_flat, targets_flat),
    }
