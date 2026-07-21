"""Projection, normalization, position, and cache preparation pipeline."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class QKVPipeline:
    def __init__(self, attention) -> None:
        self.attention = attention

    def prepare(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_state,
        *,
        force_paged: bool | None = None,
    ):
        attn = self.attention
        batch_size = query.size(0)
        q, k, v, kv_latent = attn._project_qkv_heads(query, key, value)
        provider = attn.cache_selector.select(
            layer_state,
            batch_size=batch_size,
            device=q.device,
            dtype=q.dtype,
            force_paged=force_paged,
        )
        start_positions = None
        if not attn.cross_attention:
            start_positions = provider.get_start_positions(batch_size, q.device)

        if attn.qk_norm:
            if attn.q_norm is not None:
                q, k = attn.q_norm(q), attn.k_norm(k)
            else:
                q = F.normalize(q, p=2.0, dim=-1)
                k = F.normalize(k, p=2.0, dim=-1)

        q, k = attn.position_encoding_applier.apply(
            q,
            k,
            query=query,
            key=key,
            layer_state=layer_state,
            seqlen_offset=start_positions if start_positions is not None else 0,
        )
        return q, k, v, kv_latent, provider, start_positions


__all__ = ["QKVPipeline"]
