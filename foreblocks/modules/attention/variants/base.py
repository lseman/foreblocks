"""foreblocks.modules.attention.variants.base.

This module implements the base pieces for its package.
It belongs to the attention pattern variants area of Foreblocks.
It exposes classes such as AttentionImpl.
"""

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AttentionImpl(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        **extra,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]: ...
