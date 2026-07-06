"""Training-time helper utilities for DARTS search."""

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler


class TemperatureScheduler:
    """Advanced temperature scheduling for architecture search."""

    def __init__(
        self,
        initial_temp: float = 2.0,
        final_temp: float = 0.1,
        schedule_type: str = "cosine",
        warmup_epochs: int = 5,
        initial_drnas_concentration: float = 10.0,
        final_drnas_concentration: float = 2.0,
    ):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.schedule_type = schedule_type
        self.warmup_epochs = warmup_epochs
        self.initial_drnas_concentration = max(float(initial_drnas_concentration), 1e-3)
        self.final_drnas_concentration = max(float(final_drnas_concentration), 1e-3)

    def get_temperature(self, epoch: int, total_epochs: int) -> float:
        """Get temperature for current epoch."""
        if epoch < self.warmup_epochs:
            return self.initial_temp

        anneal_span = max(total_epochs - self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / anneal_span
        progress = min(max(progress, 0.0), 1.0)

        schedule_fns = {
            "cosine": lambda: (
                self.final_temp
                + (self.initial_temp - self.final_temp)
                * (1 + np.cos(np.pi * progress))
                / 2
            ),
            "exponential": lambda: (
                self.initial_temp
                * np.exp(
                    np.log(self.final_temp / self.initial_temp)
                    / anneal_span
                    * (epoch - self.warmup_epochs)
                )
            ),
            "linear": lambda: (
                self.initial_temp - (self.initial_temp - self.final_temp) * progress
            ),
            "step": lambda: (
                self.initial_temp
                if progress < 0.3
                else (self.initial_temp * 0.5 if progress < 0.7 else self.final_temp)
            ),
        }
        temp = schedule_fns.get(self.schedule_type, lambda: self.initial_temp)()
        return max(float(temp), float(self.final_temp))

    def get_drnas_concentration(self, epoch: int, total_epochs: int) -> float:
        """Cosine-anneal DrNAS Dirichlet concentration from high (exploration) to low (exploitation).

        High concentration → samples cluster near the mean (softmax) → exploration.
        Low  concentration → samples spread toward simplex vertices       → exploitation.
        """
        if epoch < self.warmup_epochs:
            return self.initial_drnas_concentration
        anneal_span = max(total_epochs - self.warmup_epochs, 1)
        progress = (epoch - self.warmup_epochs) / anneal_span
        progress = min(max(progress, 0.0), 1.0)
        return (
            self.final_drnas_concentration
            + (self.initial_drnas_concentration - self.final_drnas_concentration)
            * (1.0 + np.cos(np.pi * progress))
            / 2.0
        )


