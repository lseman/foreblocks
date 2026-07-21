"""State, pruning, and diagnostics helpers for DARTS training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a non-aliasing CPU snapshot suitable for long-lived checkpoints."""
    return {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in model.state_dict().items()
    }


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
