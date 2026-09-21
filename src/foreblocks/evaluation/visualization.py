"""Forecast and interval visualization.

Visualization helpers for training predictions and conformal intervals.

Extracted from the Trainer module. Provides prediction-vs-actual plots,
conformal interval plots, a streaming violation heatmap, forecast channel
name generation, and array flattening utilities for forecast and series
data in various tensor shapes.

Core API:
- plot_prediction: plot model predictions against actual values
- plot_intervals: plot predictions with conformal intervals
- plot_violation_heatmap_streaming: streaming conformal-coverage-miss heatmap
- _flatten_forecast_array: flatten forecast arrays to (N, H, D)
- _forecast_channel_names: generate channel names for forecast arrays
- _flatten_series_array: flatten series arrays to (T, S_dim)

"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from torch.utils.data import DataLoader, TensorDataset

if TYPE_CHECKING:
    from foreblocks.training.trainer import Trainer


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "matplotlib is required for visualization methods. "
            "Install it with: pip install matplotlib"
        ) from None


def _flatten_forecast_array(values: torch.Tensor | np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
    arr = (
        values.detach().cpu().numpy()
        if hasattr(values, "detach")
        else np.asarray(values)
    )
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
    arr = (
        values.detach().cpu().numpy()
        if hasattr(values, "detach")
        else np.asarray(values)
    )
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
        return [
            f"{names[node]} feature {feat}"
            for node in range(nodes)
            for feat in range(features)
        ]
    if names is not None and len(names) == channels:
        return list(names)
    return [
        f"node {node} feature {feat}"
        for node in range(nodes)
        for feat in range(features)
    ]


def _flatten_series_array(values: torch.Tensor | np.ndarray) -> np.ndarray:  # type: ignore[name-defined]
    arr = (
        values.detach().cpu().numpy()
        if hasattr(values, "detach")
        else np.asarray(values)
    )
    if arr.ndim == 1:
        return arr[:, None]
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def plot_prediction(
    trainer: Trainer,
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

        fig, axes = plt.subplots(
            D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True
        )
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
                    pred_col = j if j < D else 0
                    acc[s:e] += pred_np[k, : e - s, pred_col]
                    cnt[s:e] += 1
            have = cnt > 0
            mean_pred = np.zeros(T)
            mean_pred[have] = acc[have] / cnt[have]
            x = np.arange(coverage_end)
            ax.plot(
                series[:coverage_end, j], label=f"Actual {channel_names[j]}", alpha=0.8
            )
            if have[:coverage_end].any():
                ax.plot(
                    x[have[:coverage_end]],
                    mean_pred[:coverage_end][have[:coverage_end]],
                    label=f"Predicted {channel_names[j]}",
                    linestyle="--",
                    color=pred_color,
                )
            ax.axvline(
                offset + seq_len,
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="First forecast",
            )
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

    fig, axes = plt.subplots(
        D_plot, 1, figsize=(figsize[0], figsize[1] * D_plot), sharex=True
    )
    axes = np.atleast_1d(axes)
    for j in range(D_plot):
        ax = axes[j]
        horizon = np.arange(len(pred_mean))
        ax.plot(
            horizon,
            y_mean[:, j],
            label=f"Actual {channel_names[j]}",
            marker="o",
            alpha=0.7,
        )
        ax.plot(
            horizon,
            pred_mean[:, j],
            label=f"Predicted {channel_names[j]}",
            marker="s",
            linestyle="--",
            alpha=0.7,
        )
        ax.set_title(f"{channel_names[j]}: Average Forecast")
        ax.legend()
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Forecast Horizon")
    plt.tight_layout()
    if show:
        plt.show()
    return fig


def plot_intervals(
    trainer: Trainer,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    full_series: torch.Tensor | None = None,
    time_index: Sequence[Any] | None = None,
    offset: int = 0,
    stride: int = 1,
    figsize: tuple[int, int] = (14, 5),
    show: bool = True,
    names: str | list | None = None,
    interval_alpha: float = 0.25,
    pred_color: str = "blue",
    interval_color: str = "blue",
    aggregation: str = "envelope",
    show_width_plot: bool = True,
    min_count: int = 1,
    do_update: bool = False,
) -> plt.Figure:  # type: ignore[name-defined]
    _require_matplotlib()

    if (
        trainer.conformal_engine is None
        or getattr(trainer.conformal_engine, "radii", None) is None
    ):
        raise RuntimeError(
            "Conformal engine not calibrated. Call calibrate_conformal() first."
        )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val), batch_size=256, shuffle=False
    )
    preds, lower, upper, y_stream = trainer.predict_with_intervals_streaming(
        val_loader,
        do_update=do_update,
        return_numpy=True,
    )

    N, H, D = preds.shape
    seq_len = X_val.shape[1]

    if full_series is None:
        raise ValueError("full_series must be provided for time-aligned plotting.")

    series = (
        full_series.detach().cpu().numpy()
        if isinstance(full_series, torch.Tensor)
        else full_series
    )
    if series.ndim == 1:
        series = series[:, None]

    T, S_dim = series.shape
    D_plot = min(D, S_dim)
    names = names or [f"Feature {i}" for i in range(D_plot)]

    starts = offset + seq_len + np.arange(N) * stride
    coverage_end = min(int(starts[-1] + H), T)
    if time_index is None:
        xs = np.arange(coverage_end)
        first_forecast_x = offset + seq_len
        xlabel = "Time Step"
    else:
        xs_full = np.asarray(time_index)
        if xs_full.ndim != 1:
            raise ValueError("time_index must be 1-dimensional.")
        if len(xs_full) < coverage_end:
            raise ValueError(
                f"time_index must have at least {coverage_end} elements, got {len(xs_full)}."
            )
        if offset + seq_len >= len(xs_full):
            raise ValueError(
                f"time_index must include the first forecast boundary at {offset + seq_len}."
            )
        xs = xs_full[:coverage_end]
        first_forecast_x = xs_full[offset + seq_len]
        xlabel = "Time"

    count = np.zeros((T,))

    # Initialize based on aggregation method
    if aggregation == "envelope":
        agg_pred = np.zeros((T, D_plot))
        agg_low = np.full((T, D_plot), np.inf)
        agg_up = np.full((T, D_plot), -np.inf)
        for k in range(N):
            start = int(starts[k])
            if start >= T:
                continue
            end = min(start + H, T)
            h = end - start
            if h <= 0:
                continue
            for j in range(D_plot):
                pred_col = j if j < D else 0
                agg_pred[start:end, j] += preds[k, :h, pred_col]
                agg_low[start:end, j] = np.minimum(
                    agg_low[start:end, j], lower[k, :h, pred_col]
                )
                agg_up[start:end, j] = np.maximum(
                    agg_up[start:end, j], upper[k, :h, pred_col]
                )
            count[start:end] += 1
        have = count >= min_count
        mean_pred = np.zeros_like(agg_pred)
        mean_pred[have] = agg_pred[have] / count[have, None]
        mean_low = np.where(agg_low == np.inf, 0, agg_low)
        mean_up = np.where(agg_up == -np.inf, 0, agg_up)

    elif aggregation == "last":
        mean_pred = np.full((T, D_plot), np.nan)
        mean_low = np.full((T, D_plot), np.nan)
        mean_up = np.full((T, D_plot), np.nan)
        for k in range(N):
            start = int(starts[k])
            if start >= T:
                continue
            end = min(start + H, T)
            h = end - start
            if h <= 0:
                continue
            for j in range(D_plot):
                pred_col = j if j < D else 0
                mean_pred[start:end, j] = preds[k, :h, pred_col]
                mean_low[start:end, j] = lower[k, :h, pred_col]
                mean_up[start:end, j] = upper[k, :h, pred_col]
            count[start:end] += 1
        have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)

    elif aggregation == "min_width":
        mean_pred = np.full((T, D_plot), np.nan)
        mean_low = np.full((T, D_plot), np.nan)
        mean_up = np.full((T, D_plot), np.nan)
        min_width = np.full((T, D_plot), np.inf)
        for k in range(N):
            start = int(starts[k])
            if start >= T:
                continue
            end = min(start + H, T)
            h = end - start
            if h <= 0:
                continue
            for j in range(D_plot):
                pred_col = j if j < D else 0
                width_k = upper[k, :h, pred_col] - lower[k, :h, pred_col]
                for t_idx, t in enumerate(range(start, end)):
                    if width_k[t_idx] < min_width[t, j]:
                        min_width[t, j] = width_k[t_idx]
                        mean_pred[t, j] = preds[k, t_idx, pred_col]
                        mean_low[t, j] = lower[k, t_idx, pred_col]
                        mean_up[t, j] = upper[k, t_idx, pred_col]
            count[start:end] += 1
        have = (~np.isnan(mean_pred[:, 0])) & (count >= min_count)

    else:  # "mean"
        acc_pred = np.zeros((T, D_plot))
        acc_low = np.zeros((T, D_plot))
        acc_up = np.zeros((T, D_plot))
        for k in range(N):
            start = int(starts[k])
            if start >= T:
                continue
            end = min(start + H, T)
            h = end - start
            if h <= 0:
                continue
            for j in range(D_plot):
                pred_col = j if j < D else 0
                acc_pred[start:end, j] += preds[k, :h, pred_col]
                acc_low[start:end, j] += lower[k, :h, pred_col]
                acc_up[start:end, j] += upper[k, :h, pred_col]
            count[start:end] += 1
        have = count >= min_count
        mean_pred = np.zeros_like(acc_pred)
        mean_low = np.zeros_like(acc_low)
        mean_up = np.zeros_like(acc_up)
        for j in range(D_plot):
            mean_pred[have, j] = acc_pred[have, j] / count[have]
            mean_low[have, j] = acc_low[have, j] / count[have]
            mean_up[have, j] = acc_up[have, j] / count[have]

    interval_widths = mean_up - mean_low
    n_rows = D_plot + (1 if show_width_plot else 0)

    fig, axes = plt.subplots(
        n_rows, 1, figsize=(figsize[0], figsize[1] * n_rows), sharex=True
    )
    axes = np.atleast_1d(axes)

    for j in range(D_plot):
        ax = axes[j]
        ax.plot(
            xs,
            series[:coverage_end, j],
            label=f"Actual {names[j]}",
            alpha=0.8,
            linewidth=1,
        )
        mask = have[:coverage_end]
        if mask.any():
            yp = mean_pred[:coverage_end, j]
            yl = mean_low[:coverage_end, j]
            yu = mean_up[:coverage_end, j]
            ax.plot(
                xs[mask],
                yp[mask],
                label=f"Predicted {names[j]}",
                linestyle="--",
                color=pred_color,
                linewidth=1,
            )
            ax.fill_between(
                xs[mask],
                yl[mask],
                yu[mask],
                color=interval_color,
                alpha=interval_alpha,
                label=f"Interval ({aggregation})",
            )
        ax.axvline(
            first_forecast_x,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="First forecast",
        )
        ax.set_title(f"{names[j]} — Forecast with Conformal Intervals")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)

    if show_width_plot:
        ax_width = axes[-1]
        for j in range(D_plot):
            mask = have[:coverage_end]
            widths_j = interval_widths[:coverage_end, j]
            ax_width.plot(
                xs[mask],
                widths_j[mask],
                label=f"Width {names[j]}",
                alpha=0.8,
                linewidth=1,
            )
        ax_width.axvline(first_forecast_x, color="gray", linestyle="--", alpha=0.5)
        ax_width.set_ylabel("Interval Width")
        ax_width.set_title("Adaptive Interval Widths Over Time")
        ax_width.legend(loc="upper left", fontsize=8)
        ax_width.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xlabel)
    plt.tight_layout()

    if show:
        plt.show()
    return fig


def plot_violation_heatmap_streaming(
    trainer: Trainer,
    dataloader: DataLoader,
    feature: int = 0,
    do_update: bool = True,
    figsize: tuple[int, int] = (10, 4),
    show: bool = True,
    sequential: bool | None = None,
) -> plt.Figure:  # type: ignore[name-defined]
    _require_matplotlib()

    if (
        trainer.conformal_engine is None
        or getattr(trainer.conformal_engine, "radii", None) is None
    ):
        raise RuntimeError(
            "Conformal engine not calibrated. Call calibrate_conformal() first."
        )

    preds, L, U, y_true = trainer.predict_with_intervals_streaming(
        dataloader,
        do_update=do_update,
        return_numpy=True,
        sequential=sequential,
    )

    N, H, D = L.shape
    j = int(feature)
    if j < 0 or j >= D:
        raise ValueError(f"feature index out of range: {j} (D={D})")

    covered = (y_true >= L) & (y_true <= U)
    miss = ~covered[:, :, j]

    fig, ax = plt.subplots(figsize=figsize)
    binary_cmap = ListedColormap(["white", "black"])
    im = ax.imshow(
        miss.astype(float),
        aspect="auto",
        interpolation="nearest",
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
    )

    ax.set_xlabel("Horizon")
    ax.set_ylabel("Window index (stream order)")
    ax.set_title(f"Conformal Misses — feature={j}")
    ax.set_xticks(np.arange(H))
    ax.set_xticklabels([str(h + 1) for h in range(H)])

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(["Covered", "Miss"])

    plt.tight_layout()
    if show:
        plt.show()
    return fig
