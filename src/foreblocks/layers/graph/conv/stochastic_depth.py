"""foreblocks.layers.graph.conv.stochastic_depth.

Stochastic depth layer implementation.

"""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x + residual
        keep_prob = 1 - self.drop_prob
        mask = torch.bernoulli(
            torch.full((x.size(0), 1, 1, 1), keep_prob, device=x.device)
        )
        return (x / keep_prob) * mask + residual
