from typing import Optional, Protocol, Tuple, runtime_checkable

import torch


@runtime_checkable
class AttentionImpl(Protocol):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        **extra,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]: ...
