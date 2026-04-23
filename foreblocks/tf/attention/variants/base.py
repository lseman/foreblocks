from typing import Protocol
from typing import runtime_checkable

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
