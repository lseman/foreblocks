"""foreblocks.ts_handler.plotting.

Plotting utilities for time-series preprocessing.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (18, 9),
            "figure.facecolor": "white",
            "figure.dpi": 100,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#333333",
            "axes.labelsize": 14,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "grid.color": "#dddddd",
            "grid.linestyle": "--",
            "grid.linewidth": 0.5,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cccccc",
            "legend.fontsize": 12,
            "legend.loc": "upper right",
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "font.family": "DejaVu Sans",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "savefig.dpi": 150,
        }
    )


def _plot_comparison(
    original: np.ndarray,
    cleaned: np.ndarray,
    title: str = "Preprocessing Comparison",
    time_stamps: np.ndarray | None = None,
    max_features: int | None = None,
) -> None:
    max_features = (
        max_features if max_features is not None else 8
    )

    original = np.atleast_2d(original)
    cleaned = np.atleast_2d(cleaned)

    if original.shape[0] == 1:
        original = original.T
    if cleaned.shape[0] == 1:
        cleaned = cleaned.T
    if original.shape != cleaned.shape:
        raise ValueError(
            f"Shape mismatch after processing: original {original.shape}, cleaned {cleaned.shape}"
        )

    x = time_stamps if time_stamps is not None else np.arange(original.shape[0])
    if len(x) != original.shape[0]:
        raise ValueError(
            f"Length of x ({len(x)}) != n_samples ({original.shape[0]})"
        )

    d = original.shape[1]
    idx = list(range(min(d, max_features)))

    fig, axs = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for i in idx:
        axs[0].plot(x, original[:, i], label=f"Feature {i}")
        axs[1].plot(x, cleaned[:, i], label=f"Feature {i}")

    axs[0].set_title("Original")
    axs[1].set_title("Cleaned")
    axs[0].legend(ncol=min(len(idx), 4))
    axs[1].legend(ncol=min(len(idx), 4))
    axs[0].grid(True)
    axs[1].grid(True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()
