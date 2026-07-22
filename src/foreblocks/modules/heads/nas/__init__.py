"""Differentiable architecture search for head graphs."""

from foreblocks.modules.heads.nas.controller import HeadNASController
from foreblocks.modules.heads.nas.schedules import CosineTemperatureSchedule

__all__ = ["CosineTemperatureSchedule", "HeadNASController"]
