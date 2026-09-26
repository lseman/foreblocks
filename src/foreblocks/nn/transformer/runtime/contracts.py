"""Structural contracts shared by transformer runtime workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.transformer.config import TransformerConfig

from collections.abc import Iterator
from typing import Any, Protocol

import torch
import torch.nn as nn

from foreblocks.nn.transformer.runtime.state import DecoderState


class DecoderOwner(Protocol):
    config: TransformerConfig

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


__all__ = ["DecoderOwner"]
