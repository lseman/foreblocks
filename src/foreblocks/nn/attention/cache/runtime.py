"""Paged-cache lifecycle and compaction coordination for MultiAttention.

The mixin uses the owner's configuration and compactor. It registers no modules
or tensors; providers and storage remain independent of the attention model.
"""

from __future__ import annotations

import torch

from foreblocks.nn.attention.cache.paged import PagedKVCache


class _AttentionCache:
    def reset_paged_cache(self, layer_state: dict[str, torch.Tensor] | None):
        if layer_state and "paged_cache" in layer_state:
            cache: PagedKVCache = layer_state["paged_cache"]
            for b in range(cache.B):
                cache.reset_seq(b)

    def _ensure_paged_cache(
        self,
        layer_state: dict,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PagedKVCache:
        return PagedKVCache.ensure(
            layer_state,
            batch_size=batch_size,
            n_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            latent_dim=(self.kv_latent_dim if self.use_mla else None),
            block_size=self.cache_block_size,
            device=device,
            dtype=dtype,
            max_blocks=self.max_cache_blocks,
        )

    def _can_apply_attention_matching_compaction(
        self,
        *,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        need_weights: bool,
        cache: PagedKVCache,
        t_new: int,
    ) -> bool:
        if not self.use_attention_matching_compaction:
            return False
        if self.attention_matching_compactor is None:
            return False
        if self.cross_attention or cache.use_latent_cache:
            return False
        if attn_mask is not None or key_padding_mask is not None or need_weights:
            return False
        if self.training or self.dropout_p != 0.0:
            return False
        for b in range(cache.B):
            if self.attention_matching_compactor.should_compact(cache, b, t_new=t_new):
                return True
        return False

    def _maybe_compact_paged_cache(
        self,
        *,
        cache: PagedKVCache,
        q: torch.Tensor,
        q_start_pos: torch.Tensor | None,
        t_new: int,
    ) -> None:
        if self.attention_matching_compactor is None:
            return
        if q_start_pos is None:
            raise ValueError(
                "q_start_pos is required for attention-matching compaction"
            )
        self.attention_matching_compactor.compact_batch(
            cache=cache,
            q_bhtd=q,
            q_start_pos=q_start_pos,
            kv_repeat=self.n_rep,
            scale=self.scale,
            t_new=t_new,
        )
