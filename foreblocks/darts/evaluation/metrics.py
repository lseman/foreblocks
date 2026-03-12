"""
Evaluation metrics for DARTS time-series forecasting models.

Provides pure functions so they can be used both standalone and from within
DARTSTrainer methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np
import torch
import torch.nn as nn

from ..utils.training import unpack_forecasting_batch

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------


def compute_metrics(preds: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute standard regression / forecasting metrics.

    Args:
        preds:   Flat prediction array (any shape, will be flattened).
        targets: Flat target array     (same shape as preds).

    Returns:
        Dict with keys: mse, rmse, mae, mape, r2_score.
    """
    preds = preds.reshape(-1).astype(np.float64)
    targets = targets.reshape(-1).astype(np.float64)

    mse = float(np.mean((preds - targets) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - targets)))
    mape = float(np.mean(np.abs((preds - targets) / (np.abs(targets) + 1e-8))) * 100)
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = float(1.0 - (ss_res / (ss_tot + 1e-8)))

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "r2_score": r2_score,
    }


# ---------------------------------------------------------------------------
# Model-level evaluation helpers
# ---------------------------------------------------------------------------


def evaluate_on_loader(
    model: nn.Module,
    dataloader,
    loss_fn,
    device: str,
    *,
    autocast_ctx=None,
) -> float:
    """
    Run one full evaluation pass and return mean loss.

    Args:
        model:        PyTorch model in eval mode.
        dataloader:   DataLoader yielding (x, y, ...) tuples.
        loss_fn:      Callable(predictions, targets) -> scalar Tensor.
        device:       Target device string.
        autocast_ctx: Optional autocast context manager factory (``lambda: autocast(...)``).

    Returns:
        Mean loss over all batches.
    """
    model.eval()
    total_loss = 0.0
    _ctx = autocast_ctx or _null_ctx

    with torch.no_grad():
        for batch in dataloader:
            x, y, model_kwargs = unpack_forecasting_batch(
                batch,
                device,
                include_decoder_targets=False,
                teacher_forcing_ratio=0.0,
            )
            with _ctx():
                preds = model(x, **model_kwargs)
                total_loss += loss_fn(preds, y).item()

    return total_loss / max(len(dataloader), 1)


def compute_final_metrics(
    model: nn.Module,
    val_loader,
    device: str,
    *,
    autocast_ctx=None,
    progress_fn=None,
) -> Dict[str, float]:
    """
    Collect all predictions from a dataloader and compute regression metrics.

    Args:
        model:        PyTorch model.
        val_loader:   Validation DataLoader yielding (x, y, ...) tuples.
        device:       Target device string.
        autocast_ctx: Optional autocast context factory.
        progress_fn:  Optional callable that wraps the dataloader with a progress bar.

    Returns:
        Dict from :func:`compute_metrics`.
    """
    model.eval()
    all_preds, all_targets = [], []
    _ctx = autocast_ctx or _null_ctx
    iterable = progress_fn(val_loader) if progress_fn else val_loader

    with torch.no_grad():
        for batch in iterable:
            x, y, model_kwargs = unpack_forecasting_batch(
                batch,
                device,
                include_decoder_targets=False,
                teacher_forcing_ratio=0.0,
            )
            x = x.float()
            with _ctx():
                all_preds.append(model(x, **model_kwargs).cpu().numpy())
            all_targets.append(y.cpu().numpy())

    preds_flat = np.concatenate(all_preds).reshape(-1)
    targets_flat = np.concatenate(all_targets).reshape(-1)
    return compute_metrics(preds_flat, targets_flat)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _null_ctx:  # noqa: N801
    """No-op context manager used when AMP is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
