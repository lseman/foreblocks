from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from foreblocks.config import TrainingConfig


class LossComputer:
    """Handles all loss computation logic."""

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

        if self.config.l1_regularization > 0:
            l1_loss = sum(
                p.abs().sum() for p in self.model.parameters() if p.requires_grad
            )
            total_loss += self.config.l1_regularization * l1_loss
            self.components["l1_loss"] = (
                self.config.l1_regularization * l1_loss
            ).item()

        if hasattr(self.model, "get_kl"):
            kl_div = self.model.get_kl()
            if kl_div is not None:
                total_loss += self.config.kl_weight * kl_div
                self.components["kl_loss"] = (self.config.kl_weight * kl_div).item()

        if hasattr(self.model, "get_aux_loss"):
            aux_loss = self.model.get_aux_loss()
            if aux_loss is not None:
                total_loss += aux_loss.detach()
                self.components["aux_loss"] = aux_loss.item()

        return total_loss
