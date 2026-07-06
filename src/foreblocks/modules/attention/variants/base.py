"""foreblocks.modules.attention.variants.base.

Attention implementation protocol for variant backends.

Defines the AttentionImpl Protocol that all attention variants must satisfy.
Each variant (ProbSparse, NSA, MoBA, sliding window, etc.) implements this
protocol to plug into the parent MultiAttention module.

Core API:
- AttentionImpl: Protocol defining the forward signature for attention variants

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
