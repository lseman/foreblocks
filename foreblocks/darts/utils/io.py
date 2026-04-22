"""
Model persistence and summary utilities.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger("NASLogger")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_model(
    model: nn.Module,
    filepath: str,
    *,
    final_metrics: dict[str, float] | None = None,
    training_info: dict[str, Any] | None = None,
    search_config: dict[str, Any] | None = None,
) -> None:
    """
    Save a model checkpoint to *filepath*.

    Args:
        model:          The model whose ``state_dict`` is saved.
        filepath:       Destination ``.pth`` path.
        final_metrics:  Optional dict of evaluation metrics to store.
        training_info:  Optional dict with training metadata.
        search_config:  Optional dict with search configuration.
    """
    payload: dict[str, Any] = {"model_state_dict": model.state_dict()}
    if final_metrics is not None:
        payload["final_metrics"] = final_metrics
    if training_info is not None:
        payload["training_info"] = training_info
    if search_config is not None:
        payload["search_config"] = search_config

    torch.save(payload, filepath)
    print(f"Model saved to {filepath}")
    if final_metrics:
        print(f"  RMSE: {final_metrics.get('rmse', 'N/A'):.6f}")
        print(f"  R²:   {final_metrics.get('r2_score', 'N/A'):.4f}")


def load_model_checkpoint(
    filepath: str,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Load a checkpoint saved with :func:`save_model`.

    Args:
        filepath: Source ``.pth`` path.
        device:   Device to map tensors to.

    Returns:
        The raw checkpoint dict (keys: ``"model_state_dict"``, and
        whichever optional keys were saved).
    """
    ckpt: dict[str, Any] = torch.load(filepath, map_location=device)
    print(f"Loaded checkpoint from {filepath}")
    if "final_metrics" in ckpt:
        m = ckpt["final_metrics"]
        print(f"  RMSE: {m.get('rmse', 'N/A'):.6f}")
        print(f"  R²:   {m.get('r2_score', 'N/A'):.4f}")
    return ckpt


# ---------------------------------------------------------------------------
# Search / training summary text
# ---------------------------------------------------------------------------


def format_search_summary(search_history: list) -> str:
    """
    Return a formatted string summarising all multi-fidelity search runs.

    Args:
        search_history: The ``DARTSTrainer.search_history`` list.

    Returns:
        A human-readable summary string.
    """
    if not search_history:
        return "No searches performed yet."

    lines = ["DARTS SEARCH SUMMARY", "=" * 50]
    for i, search in enumerate(search_history):
        metrics = search["final_results"]["final_metrics"]
        config = search["search_config"]
        lines += [
            f"\nSearch {i + 1}:",
            f"  Candidates evaluated: {config['num_candidates']}",
            f"  Final test RMSE:      {metrics['rmse']:.6f}",
            f"  Final R² score:       {metrics['r2_score']:.4f}",
            f"  Training time:        {search['final_results']['training_time']:.1f}s",
        ]
    lines.append("\n" + "=" * 50)
    return "\n".join(lines)


def format_training_summary(training_history: list) -> str:
    """
    Return a formatted string summarising all training sessions.

    Args:
        training_history: The ``DARTSTrainer.training_history`` list.

    Returns:
        A human-readable summary string.
    """
    if not training_history:
        return "No training sessions completed yet."

    lines = ["TRAINING SUMMARY", "=" * 40]
    for i, run in enumerate(training_history):
        if "final_metrics" not in run:
            continue
        m = run["final_metrics"]
        lines += [
            f"\nSession {i + 1}:",
            f"  Best val loss: {run.get('best_val_loss', 'N/A')}",
            f"  RMSE:          {m.get('rmse', 'N/A')}",
            f"  R²:            {m.get('r2_score', 'N/A')}",
            f"  Training time: {run.get('training_time', 'N/A')}s",
        ]
    lines.append("\n" + "=" * 40)
    return "\n".join(lines)


def print_final_results(results: dict[str, Any]) -> None:
    """
    Pretty-print the final training results to the NASLogger.

    Args:
        results: Dict returned by the final training phase, containing
                 ``"final_metrics"``, ``"training_info"``,
                 ``"test_loss"``, and ``"training_time"``.
    """
    metrics = results["final_metrics"]
    info = results["training_info"]

    logger.info("\n" + "=" * 70)
    logger.info("FINAL MODEL TRAINING COMPLETED")
    logger.info("=" * 70)
    logger.info(
        f"{'Training duration:':<30} {results['training_time']:.1f}s  "
        f"({results['training_time'] / 60:.1f} min)"
    )
    logger.info(f"{'Total epochs:':<30} {info['epochs_completed']}")
    logger.info(
        f"{'Checkpoint used:':<30} {'SWA' if info.get('swa_used') else 'Best model'}"
    )
    logger.info(f"{'Final learning rate:':<30} {info['final_lr']:.2e}")
    logger.info("-" * 70)
    logger.info("PERFORMANCE METRICS:")
    logger.info(f"{'Test Loss:':<30} {results['test_loss']:.6f}")
    logger.info(f"{'MSE:':<30} {metrics['mse']:.6f}")
    logger.info(f"{'RMSE:':<30} {metrics['rmse']:.6f}")
    logger.info(f"{'MAE:':<30} {metrics['mae']:.6f}")
    logger.info(f"{'MAPE:':<30} {metrics['mape']:.2f}%")
    logger.info(f"{'R² Score:':<30} {metrics['r2_score']:.4f}")
    logger.info("=" * 70 + "\n")
