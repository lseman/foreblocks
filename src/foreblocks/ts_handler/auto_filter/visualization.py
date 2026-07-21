"""foreblocks.ts_handler.auto_filter.visualization.

Plotting functions for auto-filter results.

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(
    noisy: pd.Series,
    score_table: pd.DataFrame,
    candidates: dict[str, pd.Series],
    best_name: str,
    clean: pd.Series | None = None,
    top_k: int = 6,
) -> plt.Figure:
    ranked = list(score_table.index)
    top_names = ranked[: min(top_k, len(ranked))]

    fig, axs = plt.subplots(
        3,
        1,
        figsize=(14, 12),
        gridspec_kw={"height_ratios": [2.2, 2.0, 1.3]},
    )

    ax = axs[0]
    ax.plot(noisy.index, noisy.values, label="Noisy", alpha=0.35, color="tab:gray")
    ax.plot(
        noisy.index,
        candidates[best_name].values,
        label=f"Best: {best_name}",
        linewidth=2.4,
        color="tab:blue",
    )
    if clean is not None:
        ax.plot(
            noisy.index,
            clean.values,
            label="Clean (reference)",
            linestyle="--",
            alpha=0.8,
            color="tab:green",
        )
    ax.set_title("Best Filter vs Noisy Input")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.2)

    ax = axs[1]
    ax.plot(noisy.index, noisy.values, label="Noisy", alpha=0.20, color="tab:gray")
    for name in top_names:
        ax.plot(noisy.index, candidates[name].values, label=name, linewidth=1.4)
    ax.set_title(f"Top {len(top_names)} Filters by Score")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(alpha=0.2)

    ax = axs[2]
    score_series = score_table["score"]
    bar_colors = [
        "tab:blue" if n == best_name else "tab:orange" for n in score_series.index
    ]
    ax.bar(score_series.index, score_series.values, color=bar_colors)
    ax.set_ylabel("Score (lower is better)")
    ax.set_title("Filter Ranking")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.2, axis="y")

    offset = max(1e-4, float(score_series.max()) * 0.03)
    for i, (name, val) in enumerate(score_series.items()):
        ax.text(
            i,
            val + offset,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )

    fig.tight_layout()
    return fig
