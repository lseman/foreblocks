"""Stable tensor, output, and shape contracts for head graphs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import torch


@dataclass(frozen=True, slots=True)
class HeadShape:
    features: int
    sequence_length: int | None = None

    def __post_init__(self) -> None:
        if self.features <= 0:
            raise ValueError("features must be positive")


@dataclass(slots=True)
class HeadOutput:
    value: torch.Tensor
    residual: torch.Tensor | None = None
    inverse_state: object | None = None
    auxiliary_losses: dict[str, torch.Tensor] = field(default_factory=dict)


@runtime_checkable
class ShapeAwareHead(Protocol):
    def infer_output_shape(self, shape: HeadShape) -> HeadShape: ...


def normalize_head_output(output: Any) -> HeadOutput:
    """Normalize modern and legacy head return values."""
    if isinstance(output, HeadOutput):
        return output
    if isinstance(output, torch.Tensor):
        return HeadOutput(value=output)
    if isinstance(output, (tuple, list)) and len(output) == 2:
        value, second = output
        if not isinstance(value, torch.Tensor):
            raise TypeError("the first structured head output must be a tensor")
        return HeadOutput(
            value=value,
            residual=second if isinstance(second, torch.Tensor) else None,
            inverse_state=None if isinstance(second, torch.Tensor) else second,
        )
    raise TypeError(f"unsupported head output type: {type(output).__name__}")


def infer_head_shape(head: object, input_shape: HeadShape) -> HeadShape:
    if isinstance(head, ShapeAwareHead):
        return head.infer_output_shape(input_shape)
    output_dim = getattr(head, "output_dim", None)
    if isinstance(output_dim, int) and output_dim > 0:
        return HeadShape(output_dim, input_shape.sequence_length)
    return input_shape


__all__ = [
    "HeadOutput",
    "HeadShape",
    "ShapeAwareHead",
    "infer_head_shape",
    "normalize_head_output",
]
