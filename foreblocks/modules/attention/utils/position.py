"""foreblocks.modules.attention.utils.position.

This module implements the position pieces for its package.
It belongs to the attention modules, variants, caches, and utilities area of Foreblocks.
It exposes classes such as PositionEncodingApplier.
"""

import torch


class PositionEncodingApplier:
    """Composable positional transform pipeline for Q/K tensors."""

    def __init__(self):
        self._transforms = []

    def add_transform(self, name: str, fn) -> None:
        self._transforms.append((str(name), fn))

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        **context,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for _, transform in self._transforms:
            q, k = transform(q, k, context)
        return q, k
