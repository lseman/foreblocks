"""Structural contracts shared by transformer runtime workflows."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol

import torch
import torch.nn as nn

from foreblocks.models.transformer.runtime.state import DecoderState


class DecoderProtocol(Protocol):
    output_size: int
    num_layers: int

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...
    def forward_one_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: DecoderState | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, DecoderState]: ...
    def forward_multi_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: DecoderState,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, DecoderState]: ...
    def prefill(
        self, tgt: torch.Tensor, memory: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, DecoderState]: ...
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: DecoderState,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, DecoderState]: ...


__all__ = ["DecoderProtocol"]
