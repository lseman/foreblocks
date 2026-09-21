"""Differentiable architecture search for head graphs."""

from foreblocks.nn.heads.nas.controller import HeadNASController
from foreblocks.nn.heads.nas.schedules import CosineTemperatureSchedule

__all__ = ["CosineTemperatureSchedule", "HeadNASController"]
