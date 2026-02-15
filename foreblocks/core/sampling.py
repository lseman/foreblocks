import torch
import torch.nn as nn
from typing import Optional, Callable
import math

from foreblocks.ui.node_spec import node

@node(
    type_id="scheduled_sampling",
    name="Scheduled Sampling",
    category="Training",
    outputs=["sampling_fn"],
    color="bg-gradient-to-br from-amber-600 to-amber-700",
)
class ScheduledSampling:
    """
    Node that provides a scheduled sampling function for teacher forcing.
    """
    def __init__(
        self,
        strategy: str = "linear",
        start_ratio: float = 1.0,
        end_ratio: float = 0.0,
        decay_steps: int = 100,
    ):
        self.strategy = strategy
        self.start_ratio = start_ratio
        self.end_ratio = end_ratio
        self.decay_steps = decay_steps

    def forward(self) -> Callable[[Optional[int]], float]:
        def sampling_fn(epoch: Optional[int]) -> float:
            if epoch is None:
                return self.start_ratio
            
            if self.strategy == "constant":
                return self.start_ratio
            
            if self.strategy == "linear":
                ratio = self.start_ratio - (self.start_ratio - self.end_ratio) * min(1.0, epoch / self.decay_steps)
                return max(self.end_ratio, ratio)
            
            if self.strategy == "exponential":
                ratio = self.start_ratio * (self.end_ratio / self.start_ratio) ** (epoch / self.decay_steps)
                return max(self.end_ratio, ratio)
            
            return self.start_ratio

        return sampling_fn
