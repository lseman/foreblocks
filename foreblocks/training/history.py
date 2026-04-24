from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingHistory:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    task_losses: list[float] = field(default_factory=list)
    distillation_losses: list[float] = field(default_factory=list)
    model_info: list[dict[str, Any]] = field(default_factory=list)
    alpha_values: list[dict[str, Any]] = field(default_factory=list)

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
