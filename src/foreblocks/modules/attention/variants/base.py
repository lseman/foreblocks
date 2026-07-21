"""foreblocks.modules.attention.variants.base.

Attention implementation protocol for variant backends.

Defines the AttentionImpl Protocol that all attention variants must satisfy.
Each variant (ProbSparse, NSA, MoBA, sliding window, etc.) implements this
protocol to plug into the parent MultiAttention module.

Core API:
- AttentionImpl: Protocol defining the forward signature for attention variants

"""

from typing import Any, Protocol, runtime_checkable

import torch


class AttentionContext(Protocol):
    """Typed services and state exposed to attention variants.

    Variants depend on this interface instead of the full ``MultiAttention``
    module.  The concrete adapter intentionally keeps the compatibility layer
    in one place while this surface is narrowed over time.
    """

    attention_type: str
    backends: dict[str, bool]
    chunk_size: int
    cross_attention: bool
    d_model: int
    dilated_window_size: int
    dropout: torch.nn.Module
    dropout_p: float
    dwt_attention: Any
    freq_attention: Any
    head_dim: int
    k_up_proj: Any
    moba_block_size: int
    moba_topk: int
    n_heads: int
    n_rep: int
    nsa_block_size: int
    nsa_gate_proj: Any
    nsa_topk_ratio: float
    out_proj: torch.nn.Module
    scale: float
    training: bool
    use_attention_matching_compaction: bool
    use_flash_sliding: bool
    use_paged_cache: bool
    v_up_proj: Any
    window_size: int
    attention_dilation: int
    _fallback_standard: "AttentionImpl"
    _paged_stream_decode: Any
    _triton_paged_decode: Any

    def _apply_gated_attention(self, value: torch.Tensor) -> torch.Tensor: ...

    def _apply_masks(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def _can_apply_attention_matching_compaction(
        self, *args: Any, **kwargs: Any
    ) -> bool: ...

    def _compute_attention(self, *args: Any, **kwargs: Any) -> tuple[Any, Any]: ...

    def _create_sliding_window_mask(self, *args: Any, **kwargs: Any) -> Any: ...

    def _dropout_weights(self, value: torch.Tensor) -> torch.Tensor: ...

    def _finalize_projected_output(self, *args: Any, **kwargs: Any) -> Any: ...

    def _maybe_compact_paged_cache(self, *args: Any, **kwargs: Any) -> Any: ...

    def _normalize_attn_mask(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def _prepare_qkv_attention(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

    def _prepare_qkv_with_provider(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]: ...

    def _repeat_kv(self, value: torch.Tensor) -> torch.Tensor: ...

    def _slice_attn_mask(self, *args: Any, **kwargs: Any) -> Any: ...


class MultiAttentionContext:
    """Compatibility adapter implementing :class:`AttentionContext`."""

    __slots__ = ("_attention",)

    def __init__(self, attention: Any) -> None:
        self._attention = attention

    def __getattr__(self, name: str) -> Any:
        return getattr(self._attention, name)


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


__all__ = ["AttentionContext", "AttentionImpl", "MultiAttentionContext"]
