"""
Trainer utility helpers shared across the training stack.
"""

import logging
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from foreblocks.aux.config import TrainingConfig
from foreblocks.core.heads.head_helper import HeadComposer

# Optional: import your MoE classes and HeadComposer
try:
    from foreblocks.tf.experts.moe_logging import (
        MoELogger,
        ReportInputs,
        build_moe_report,
    )
except Exception:
    MoELogger = None
    ReportInputs = None

    def build_moe_report(*args, **kwargs):
        raise RuntimeError("MoE logging not available")

# Try to import HeadComposer for NAS detection


# ============================================================================
# Dataset (unchanged)
# ============================================================================


class TimeSeriesDataset(Dataset):
    """Clean dataset for time series data"""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        time_feat: np.ndarray | None = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.t = (
            torch.tensor(time_feat, dtype=torch.long) if time_feat is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None and self.t is None:
            return self.X[idx]
        if self.t is None:
            return self.X[idx], self.y[idx]
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[
            idx
        ], self.t[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    time_feat_train: np.ndarray | None = None,
    time_feat_val: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader | None]:

    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, time_feat_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val, time_feat_val),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader


# ============================================================================
# Loss Computation (unchanged)
# ============================================================================


class LossComputer:
    """Handles all loss computation logic"""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        criterion: Callable | None = None,
    ):
        self.model = model
        self.config = config
        self.criterion = criterion or nn.MSELoss()
        self.components: dict[str, float] = {}

    def compute(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        aux_data: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        self.components = {}
        aux_data = aux_data or {}

        task_loss = self.criterion(outputs, targets)
        total_loss = task_loss
        self.components["task_loss"] = task_loss.item()

        # Distillation loss
        if (
            hasattr(self.model, "compute_distillation_loss")
            and "teacher_outputs" in aux_data
        ):
            distill_loss, distill_components = self.model.compute_distillation_loss(
                outputs, aux_data["teacher_outputs"], targets, self.criterion
            )
            total_loss = distill_loss
            self.components.update(
                {
                    f"distill_{k}": v.item() if isinstance(v, torch.Tensor) else v
                    for k, v in distill_components.items()
                }
            )

        # L1
        if self.config.l1_regularization > 0:
            l1_loss = sum(
                p.abs().sum() for p in self.model.parameters() if p.requires_grad
            )
            total_loss += self.config.l1_regularization * l1_loss
            self.components["l1_loss"] = (
                self.config.l1_regularization * l1_loss
            ).item()

        # KL
        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                total_loss += self.config.kl_weight * kl_div
                self.components["kl_loss"] = (self.config.kl_weight * kl_div).item()

        # MoE aux
        if hasattr(self.model, "get_aux_loss"):
            aux_loss = self.model.get_aux_loss()
            if aux_loss is not None:
                total_loss += aux_loss.detach()
                self.components["aux_loss"] = aux_loss.item()

        return total_loss


# ============================================================================
# Training History (enhanced with alpha tracking)
# ============================================================================


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    task_losses: list[float] = field(default_factory=list)
    distillation_losses: list[float] = field(default_factory=list)
    model_info: list[dict[str, Any]] = field(default_factory=list)

    # ── NEW: NAS tracking ──
    alpha_values: list[dict[str, Any]] = field(
        default_factory=list
    )  # Per-epoch alpha snapshots

    def record_epoch(
        self,
        train_loss: float,
        val_loss: float | None,
        lr: float,
        loss_components: dict[str, float],
        model_info: dict[str, Any] | None = None,
        alpha_info: dict[str, Any] | None = None,
    ):
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        self.learning_rates.append(lr)

        if "task_loss" in loss_components:
            self.task_losses.append(loss_components["task_loss"])

        distill_loss = sum(
            v for k, v in loss_components.items() if k.startswith("distill_")
        )
        if distill_loss > 0:
            self.distillation_losses.append(distill_loss)

        if model_info:
            self.model_info.append(model_info)

        if alpha_info:
            self.alpha_values.append(alpha_info)


# ============================================================================
# NAS Helper - detects and manages HeadComposer alphas
# ============================================================================


class NASHelper:
    """
    Utility to detect HeadComposer modules and manage alpha optimization.
    """

    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.composers: list[tuple[str, nn.Module]] = []

        # Find all HeadComposer instances
        if HeadComposer is not None:
            for name, module in model.named_modules():
                if isinstance(module, HeadComposer):
                    self.composers.append((name, module))

        self.has_nas = len(self.composers) > 0

        if config.train_nas and not self.has_nas:
            logging.warning("train_nas=True but no HeadComposer found in model")

    def get_alpha_parameters(self) -> list[torch.nn.Parameter]:
        """Collect all alpha parameters from all composers."""
        params = []
        for name, composer in self.composers:
            params.extend(composer.arch_parameters())
        return params

    def get_weight_parameters(self) -> list[torch.nn.Parameter]:
        """Collect all non-alpha parameters."""
        if not self.has_nas:
            return list(self.model.parameters())

        # Collect alpha param IDs to exclude
        alpha_ids = {
            id(p) for _, composer in self.composers for p in composer.arch_parameters()
        }

        # Return all other params
        weight_params = []
        for p in self.model.parameters():
            if id(p) not in alpha_ids:
                weight_params.append(p)
        return weight_params

    def collect_alpha_report(self) -> dict[str, Any]:
        """Collect alpha values from all composers."""
        report = {}
        for name, composer in self.composers:
            report[name] = composer.alpha_report()
        return report

    def discretize_all(self, threshold: float | None = None):
        """Discretize alphas in all composers."""
        thresh = threshold or self.config.nas_discretize_threshold
        for name, composer in self.composers:
            composer.discretize_(threshold=thresh)
            logging.info(f"Discretized alphas in {name} with threshold={thresh}")


# ============================================================================
# Utility: Plot alpha evolution
# ============================================================================


def plot_alpha_evolution(history: TrainingHistory, figsize: tuple[int, int] = (15, 6)):
    """
    Plot the evolution of alpha values across training epochs.

    Args:
        history: TrainingHistory with alpha_values recorded
        figsize: Figure size
    """
    if not history.alpha_values:
        print("No alpha values recorded in history")
        return None

    # Organize data by composer and head
    alpha_series = {}  # {(composer, head, metric): [values]}

    for epoch_alphas in history.alpha_values:
        for composer_name, heads in epoch_alphas.items():
            for head_name, values in heads.items():
                for metric, value in values.items():
                    if isinstance(value, (int, float)):
                        key = (composer_name, head_name, metric)
                        if key not in alpha_series:
                            alpha_series[key] = []
                        alpha_series[key].append(value)

    if not alpha_series:
        print("No numeric alpha values found")
        return None

    # Group by composer
    composers = {}
    for (comp, head, metric), values in alpha_series.items():
        if comp not in composers:
            composers[comp] = {}
        if head not in composers[comp]:
            composers[comp][head] = {}
        composers[comp][head][metric] = values

    # Create subplots
    n_composers = len(composers)
    fig, axes = plt.subplots(n_composers, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    epochs = list(range(len(history.alpha_values)))

    for idx, (composer_name, heads) in enumerate(composers.items()):
        ax = axes[idx]

        for head_name, metrics in heads.items():
            for metric, values in metrics.items():
                if metric in ["w_head", "p_on"]:  # Plot relevant metrics
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
