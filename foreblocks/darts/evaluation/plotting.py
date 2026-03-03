"""
Visualization utilities for DARTS training and architecture search.

All functions are pure (stateless) - they receive models / data / results
as arguments rather than accessing a trainer instance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast

# ---------------------------------------------------------------------------
# Training-curve plot
# ---------------------------------------------------------------------------


def plot_training_curve(
    train_losses: List[float],
    val_losses: List[float],
    *,
    title: str = "Training Progress",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Plot train / validation loss curves side by side.

    Args:
        train_losses: Per-epoch training losses.
        val_losses:   Per-epoch validation losses.
        title:        Figure title.
        save_path:    If provided, save the figure to this path.

    Returns:
        The matplotlib Figure object (caller may ``plt.show()`` it).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, "b-", linewidth=2, label="Train Loss", alpha=0.8)
    ax.plot(epochs, val_losses, "r-", linewidth=2, label="Validation Loss", alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.7)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Training curve saved to {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Architecture-parameter (alpha) evolution
# ---------------------------------------------------------------------------


def plot_alpha_evolution(
    alpha_values: List,
    *,
    save_path: str = "alpha_evolution.png",
) -> Optional[plt.Figure]:
    """
    Plot how architecture weights (alphas) evolve across training epochs.

    Args:
        alpha_values: List of snapshots – each snapshot is a list of
                      ``(cell_idx, edge_idx, alphas_tensor)`` tuples (the
                      format produced by :class:`AlphaTracker`).
        save_path:    File path to save the plot.

    Returns:
        The matplotlib Figure, or ``None`` if there is nothing to plot.
    """
    if not alpha_values:
        print("No alpha values to plot.")
        return None

    num_edges_to_plot = min(4, len(alpha_values[0]))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for edge_idx in range(num_edges_to_plot):
        ax = axes[edge_idx]
        edge_alphas: List[np.ndarray] = []

        for epoch_alphas in alpha_values:
            if edge_idx < len(epoch_alphas):
                _cell_idx, _edge_in_cell, alphas = epoch_alphas[edge_idx]
                edge_alphas.append(alphas)

        if not edge_alphas:
            continue

        edge_arr = np.array(edge_alphas)  # [epochs, n_ops]
        for op_idx in range(edge_arr.shape[1]):
            ax.plot(
                range(len(edge_arr)),
                edge_arr[:, op_idx],
                label=f"Op {op_idx}",
                linewidth=2,
                alpha=0.8,
            )

        ax.set_title(f"Edge {edge_idx} Alpha Evolution", fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alpha Weight")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Alpha evolution plot saved to {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Batched forecasting helper
# ---------------------------------------------------------------------------


def batched_forecast(
    model: nn.Module,
    X_val: torch.Tensor,
    *,
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Run model inference in mini-batches and stitch the outputs into a single
    aligned forecast tensor.

    Args:
        model:      Trained model (will be set to eval mode).
        X_val:      Input tensor of shape ``[N, seq_len, input_size]``.
        batch_size: Mini-batch size for inference (avoids OOM for large N).

    Returns:
        Forecast tensor of shape ``[N + output_len - 1, output_size]``.
    """
    model.eval()
    device = next(model.parameters()).device
    device_str = str(device)
    X_val = X_val.to(device)
    N = X_val.shape[0]

    # --- probe output shape ---
    with torch.no_grad():
        dummy = model(X_val[0:1])
        if isinstance(dummy, tuple):
            dummy = dummy[0]
        if dummy.dim() == 3:
            output_len, output_size = dummy.shape[1], dummy.shape[2]
        elif dummy.dim() == 2:
            output_len, output_size = 1, dummy.shape[1]
        else:
            output_len, output_size = 1, dummy.shape[0]

    forecast = torch.zeros(N + output_len - 1, output_size, device=device)
    count = torch.zeros_like(forecast)
    amp_enabled = device_str.startswith("cuda")

    with torch.no_grad():
        for i in range(0, N, batch_size):
            batch = X_val[i : i + batch_size]
            with autocast(
                device_type="cuda" if amp_enabled else "cpu",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                outputs = model(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                outputs = outputs.float()

            for j in range(outputs.shape[0]):
                pred = outputs[j]
                start = i + j
                if pred.dim() == 3:
                    pred = pred.squeeze(0)
                if pred.dim() == 2:
                    if pred.shape[0] == 1:
                        forecast[start] += pred.squeeze(0)
                        count[start] += 1
                    else:
                        forecast[start : start + pred.shape[0]] += pred
                        count[start : start + pred.shape[0]] += 1
                elif pred.dim() == 1:
                    forecast[start] += pred
                    count[start] += 1
                else:
                    raise ValueError(f"Unexpected prediction shape: {pred.shape}")

    return forecast / count.clamp(min=1.0)


# ---------------------------------------------------------------------------
# Prediction vs. ground-truth plot
# ---------------------------------------------------------------------------


def plot_prediction(
    model: nn.Module,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    full_series: Optional[torch.Tensor] = None,
    offset: int = 0,
    figsize: Tuple[int, int] = (12, 4),
    show: bool = False,
    device: Optional[torch.device] = None,
    names: Optional[List[str]] = None,
    batch_size: int = 256,
) -> plt.Figure:
    """
    Visualise model predictions over the validation period.

    Args:
        model:       Trained model.
        X_val:       Validation inputs ``[N, seq_len, input_dim]``.
        y_val:       Validation targets ``[N, target_len, output_dim]``.
        full_series: Optional full time series for context ``[T, features]``.
        offset:      Index in ``full_series`` where the validation window starts.
        figsize:     ``(width, height)`` in inches per feature subplot.
        show:        Call ``plt.show()`` before returning.
        device:      Device override (defaults to model's device).
        names:       Optional feature names for subplot titles.
        batch_size:  Mini-batch size passed to :func:`batched_forecast`.

    Returns:
        The matplotlib Figure object.
    """
    if device is not None:
        X_val = X_val.to(device)

    forecast = batched_forecast(model, X_val, batch_size=batch_size).cpu().numpy()

    if full_series is not None:
        full_series_np = full_series.cpu().numpy()
        n_features = full_series_np.shape[-1] if full_series_np.ndim > 1 else 1
        fig, axes = plt.subplots(
            n_features,
            1,
            figsize=(figsize[0], figsize[1] * n_features),
            sharex=True,
        )
        if n_features == 1:
            axes = [axes]

        forecast_start = offset + X_val.shape[1]

        for i in range(n_features):
            ax = axes[i]
            feat_label = names[i] if names else f"Feature {i}"

            if full_series_np.ndim == 3:
                feature_series = full_series_np[:, 0, i]
            elif full_series_np.ndim == 2:
                feature_series = full_series_np[:, i]
            else:
                feature_series = full_series_np

            ax.plot(
                np.arange(len(feature_series)),
                feature_series,
                label=f"Original – {feat_label}",
                alpha=0.5,
            )

            feature_forecast = forecast[:, i] if forecast.ndim > 1 else forecast
            end_idx = min(forecast_start + len(feature_forecast), len(feature_series))
            fslice = slice(forecast_start, end_idx)
            forecast_plot = feature_forecast[: end_idx - forecast_start]

            ax.plot(
                np.arange(fslice.start, fslice.stop),
                forecast_plot,
                label=f"Forecast – {feat_label}",
                color="orange",
            )

            if len(feature_series) >= end_idx:
                ax.fill_between(
                    np.arange(fslice.start, fslice.stop),
                    forecast_plot,
                    feature_series[fslice],
                    color="red",
                    alpha=0.2,
                    label="Forecast Error",
                )

            ax.axvline(
                x=forecast_start, color="gray", linestyle="--", label="Forecast Start"
            )
            ax.set_title(f"{feat_label} Forecast")
            ax.legend(loc="upper left")
            ax.grid(True)

        plt.xlabel("Time Step")
        axes[n_features // 2].set_ylabel("Value")
        fig.tight_layout()

    else:
        fig, ax = plt.subplots(figsize=figsize)
        if forecast.ndim > 1:
            for i in range(forecast.shape[1]):
                label = names[i] if names else f"Feature {i}"
                ax.plot(forecast[:, i], label=f"Forecast {label}")
        else:
            ax.plot(forecast, label="Forecast", color="orange")
        ax.set_title("Validation Prediction")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.legend(loc="upper left")
        ax.grid(True)

    if show:
        plt.show()

    return fig
