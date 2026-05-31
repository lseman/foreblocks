"""Visualization helpers for Trainer.

Extracted from the monolithic ``trainer.py``.  Provides prediction plots,
conformal interval plots, and violation heatmaps.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch

if TYPE_CHECKING:
    from .trainer import Trainer


def _require_matplotlib() -> None:
    """Ensure matplotlib is importable; raise a helpful error otherwise."""
    try:
        import matplotlib  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization methods. "
            "Install it with: pip install matplotlib"
        )


def _flatten_forecast_array(values: torch.Tensor | np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
    """Flatten forecast arrays to ``(N, H, D)``."""
    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    if arr.ndim > 3:
        return arr.reshape(arr.shape[0], arr.shape[1], -1)
    raise ValueError(f"Expected forecast array with at least 2 dims, got {arr.shape}.")


def _forecast_channel_names(
    values: torch.Tensor | np.ndarray,
    names: str | list | None,
) -> list[str]:
    """Generate channel names for forecast arrays."""
    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    if arr.ndim <= 3:
        channels = 1 if arr.ndim == 2 else arr.shape[-1]
        if isinstance(names, str):
            return [names]
        if names is not None and len(names) == channels:
            return list(names)
        return [f"Feature {idx}" for idx in range(channels)]

    nodes = arr.shape[2]
    features = int(np.prod(arr.shape[3:]))
    channels = nodes * features
    if isinstance(names, str):
        names = [names]
    if names is not None and len(names) == nodes:
        return [f"{names[node]} feature {feat}" for node in range(nodes) for feat in range(features)]
    if names is not None and len(names) == channels:
        return list(names)
    return [f"node {node} feature {feat}" for node in range(nodes) for feat in range(features)]


def _flatten_series_array(values: torch.Tensor | np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
    """Flatten series arrays to ``(T, S_dim)``."""
    arr = values.detach().cpu().numpy() if hasattr(values, "detach") else np.asarray(values)
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def plot_prediction(
    trainer: "Trainer",
    X_val: torch.Tensor,  # type: ignore[name-defined]
    y_val: torch.Tensor,  # type: ignore[name-defined]
    graph_kwargs: dict[str, Any] | None = None,
    full_series: torch.Tensor | None = None,
    offset: int = 0,
    stride: int = 1,
    figsize: tuple[int, int] = (12, 4),
    show: bool = True,
    names: str | list | None = None,
    pred_color: str = "orange",
    series_color: str = "blue",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot model predictions against actual values."""
    _require_matplotlib()

    from foreblocks.evaluation.model_evaluator import ModelEvaluator

    evaluator = ModelEvaluator(trainer)
    predictions = evaluator.predict(X_val, graph_kwargs=graph_kwargs)
    pred_np = _flatten_forecast_array(predictions)
    y_np = _flatten_forecast_array(y_val)
    N, H = pred_np.shape[0], pred_np.shape[1]
    D = pred_np.shape[2] if pred_np.ndim >= 3 else 1
    channel_names = _forecast_channel_names(predictions, names)

    if full_series is not None:
        series = _flatten_series_array(full_series)
        T, S_dim = series.shape
        D_plot = S_dim
        if len(channel_names) != D_plot:
            channel_names = [f"Feature {i}" for i in range(D_plot)]
        seq_len = X_val.shape[1]
        starts = offset + seq_len + np.arange(N) * stride
        coverage_end = min(T, int(starts[-1] + H)) if N > 0 else 0

        fig, axes = plt.subplots(D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True)
        axes = np.atleast_1d(axes)
        for j in range(D_plot):
            ax = axes[j]
            acc = np.zeros(T)
            cnt = np.zeros(T)
            for k in range(N):
                s = int(starts[k])
                if s >= T:
                    continue
                e = min(s + H, T)
                if e > s:
                    pred_col = j if D > j else 0
                    acc[s:e] += pred_np[k, : e - s, pred_col]
                    cnt[s:e] += 1
            have = cnt > 0
            mean_pred = np.zeros(T)
            mean_pred[have] = acc[have] / cnt[have]
            x = np.arange(coverage_end)
            ax.plot(series[:coverage_end, j], label=f"Actual {channel_names[j]}", alpha=0.8)
            if have[:coverage_end].any():
                ax.plot(
                    x[have[:coverage_end]],
                    mean_pred[:coverage_end][have[:coverage_end]],
                    label=f"Predicted {channel_names[j]}",
                    linestyle="--", color=pred_color,
                )
            ax.axvline(offset + seq_len, color="gray", linestyle="--", alpha=0.5, label="First forecast")
            ax.set_title(f"{channel_names[j]}: Prediction vs Actual")
            ax.legend(loc="upper left")
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=120, bbox_inches="tight")
        if show:
            plt.show()
        return fig

    # ── No full_series: average-over-samples plot ──────────────────────
    pred_mean = pred_np.mean(axis=0)
    y_mean = y_np.mean(axis=0)
    if pred_mean.ndim == 1:
        pred_mean = pred_mean[:, None]
        y_mean = y_mean[:, None]
    D_plot = pred_mean.shape[1]
    if len(channel_names) != D_plot:
        channel_names = [f"Feature {i}" for i in range(D_plot)]

    fig, axes = plt.subplots(D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True)
    axes = np.atleast_1d(axes)
    for j in range(D_plot):
        ax = axes[j]
        horizon = np.arange(len(pred_mean))
        ax.plot(horizon, y_mean[:, j], label=f"Actual {channel_names[j]}", marker="o", alpha=0.7)
        ax.plot(
            horizon, pred_mean[:, j], label=f"Predicted {channel_names[j]}",
            marker="s", linestyle="--", alpha=0.7,
        )
        ax.set_title(f"{channel_names[j]}: Average Forecast")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Forecast Horizon")
    plt.tight_layout()
    if show:
        plt.show()
    return fig
