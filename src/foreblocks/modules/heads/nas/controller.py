"""NAS scheduling and deterministic export for head graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from foreblocks.modules.heads.graph import HeadGraph
from foreblocks.modules.heads.nas.schedules import CosineTemperatureSchedule


@dataclass(slots=True)
class HeadNASController:
    graph: HeadGraph
    schedule: CosineTemperatureSchedule | None = None

    def step(self, step: int) -> float:
        temperature = (
            self.schedule(step)
            if self.schedule is not None
            else self.graph.config.nas.temperature
        )
        for stage in self.graph.stages:
            manager = getattr(stage, "state_manager", None)
            if manager is not None:
                manager.alpha_temperature = temperature
                manager.gumbel_temperature = temperature
        return temperature

    def loss(self) -> torch.Tensor:
        return self.graph.architecture_regularization()

    def export(self) -> dict[str, tuple[str, ...]]:
        return self.graph.export_architecture()


__all__ = ["HeadNASController"]
