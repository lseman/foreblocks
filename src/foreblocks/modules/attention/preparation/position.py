"""Position-encoding preparation for attention inputs.

Composable positional transform pipeline for Q/K tensors.

Registers named transform functions (RoPE, SyPE, ALiBi adapters) that are
applied sequentially to query and key tensors before attention computation.
Use to attach positional encoding transforms to a parent attention module.

Core API:
- PositionEncodingApplier: named-transform pipeline for Q/K positional encodings

"""

import torch


class PositionEncodingApplier:
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
