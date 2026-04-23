from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from foreblocks.config import TrainingConfig
from foreblocks.core.heads.head_helper import HeadComposer
from foreblocks.training.history import TrainingHistory


try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


class NASHelper:
    """Detects HeadComposer modules and manages alpha optimization."""

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.composers: list[tuple[str, nn.Module]] = []

        if HeadComposer is not None:
            for name, module in model.named_modules():
                if isinstance(module, HeadComposer):
                    self.composers.append((name, module))

        self.has_nas = len(self.composers) > 0

        if config.train_nas and not self.has_nas:
            logging.warning("train_nas=True but no HeadComposer found in model")

    def get_alpha_parameters(self) -> list[torch.nn.Parameter]:
        params = []
        for _, composer in self.composers:
            params.extend(composer.arch_parameters())
        return params

    def get_weight_parameters(self) -> list[torch.nn.Parameter]:
        if not self.has_nas:
            return list(self.model.parameters())

        alpha_ids = {
            id(p) for _, composer in self.composers for p in composer.arch_parameters()
        }
        return [p for p in self.model.parameters() if id(p) not in alpha_ids]

    def collect_alpha_report(self) -> dict[str, Any]:
        return {name: composer.alpha_report() for name, composer in self.composers}

    def discretize_all(self, threshold: float | None = None):
        thresh = threshold or self.config.nas_discretize_threshold
        for name, composer in self.composers:
            composer.discretize_(threshold=thresh)
            logging.info(f"Discretized alphas in {name} with threshold={thresh}")


def plot_alpha_evolution(history: TrainingHistory, figsize: tuple[int, int] = (15, 6)):
    if plt is None:
        raise RuntimeError(
            "Matplotlib is required for NAS plotting utilities. "
            "Install with: pip install foreblocks[plotting]"
        )
    if not history.alpha_values:
        print("No alpha values recorded in history")
        return None

    alpha_series = {}
    for epoch_alphas in history.alpha_values:
        for composer_name, heads in epoch_alphas.items():
            for head_name, values in heads.items():
                for metric, value in values.items():
                    if isinstance(value, (int, float)):
                        key = (composer_name, head_name, metric)
                        alpha_series.setdefault(key, []).append(value)

    if not alpha_series:
        print("No numeric alpha values found")
        return None

    composers = {}
    for (comp, head, metric), values in alpha_series.items():
        composers.setdefault(comp, {}).setdefault(head, {})[metric] = values

    fig, axes = plt.subplots(len(composers), 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    epochs = list(range(len(history.alpha_values)))

    for idx, (composer_name, heads) in enumerate(composers.items()):
        ax = axes[idx]
        for head_name, metrics in heads.items():
            for metric, values in metrics.items():
                if metric in ["w_head", "p_on"]:
                    label = f"{head_name}.{metric}"
                    ax.plot(epochs, values, label=label, marker="o", markersize=3)

        ax.set_title(f"Alpha Evolution: {composer_name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Alpha Value")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-0.1, 1.1])

    plt.tight_layout()
    return fig
