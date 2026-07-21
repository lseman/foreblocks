"""KV-provider selection independent from attention kernel execution."""

from __future__ import annotations

import torch

from .kv import (
    DenseKVProvider,
    KVProvider,
    PagedKVProvider,
    StaticKVCache,
    StaticKVProvider,
)


class AttentionCacheSelector:
    def __init__(self, attention) -> None:
        self.attention = attention

    def select(
        self,
        layer_state,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        force_paged: bool | None = None,
    ) -> KVProvider:
        attn = self.attention
        static_cache = (
            layer_state.get("static_cache") if layer_state is not None else None
        )
        if isinstance(static_cache, StaticKVCache):
            return StaticKVProvider(
                static_cache,
                cache_position=layer_state.get("cache_position"),
                update_mask=layer_state.get("cache_update_mask"),
            )

        use_paged = (
            attn.use_paged_cache
            and layer_state is not None
            and not attn.cross_attention
        )
        if force_paged is not None:
            use_paged = bool(force_paged)
        if use_paged:
            if layer_state is None:
                raise ValueError("layer_state is required for paged KV caching")
            cache = attn._ensure_paged_cache(
                layer_state,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
            return PagedKVProvider(
                cache,
                use_mla=attn.use_mla,
                k_up_proj=attn.k_up_proj,
                v_up_proj=attn.v_up_proj,
                update_mask=layer_state.get("cache_update_mask"),
            )

        return DenseKVProvider(
            layer_state=layer_state,
            cross_attention=attn.cross_attention,
            use_mla=attn.use_mla,
            k_up_proj=attn.k_up_proj,
            v_up_proj=attn.v_up_proj,
        )


__all__ = ["AttentionCacheSelector"]
