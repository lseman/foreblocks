from typing import Tuple

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
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        for _, transform in self._transforms:
            q, k = transform(q, k, context)
        return q, k
