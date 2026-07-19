"""foreblocks.modules.attention.variants.standard.

Dense scaled dot-product attention with production KV-cache support.

Computes softmax(QKᵀ/√d)V over the full sequence. Includes grouped-query
attention (KV-head broadcasting), paged KV-cache decode for incremental
generation with attention-matching compaction. This is the default attention
path — use when no special constraints (sparsity, frequency domain, latency)
apply.

Core API:
- StandardAttentionImpl: full attention with GQA and paged KV-cache decode

"""

import torch

from foreblocks.modules.attention.cache.kv import PagedKVProvider, StaticKVCache


class StandardAttentionImpl:
    """Standard dense scaled dot-product attention (see module docstring)."""

    def __init__(self, parent):
        self.parent = parent

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if self._is_decode(layer_state=layer_state, is_causal=is_causal):
            return self.decode(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state=layer_state,
                **kwargs,
            )
        return self.prefill(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            layer_state=layer_state,
            **kwargs,
        )

    def _is_decode(self, *, layer_state, is_causal: bool) -> bool:
        return bool(
            layer_state is not None
            and not self.parent.cross_attention
            and is_causal
        )

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        del kwargs
        B, T_q, _ = query.shape
        q, k, v, q_start_pos = self.parent._prepare_qkv_attention(
            query, key, value, layer_state
        )
        out, weights = self.parent._compute_attention(
            q, k, v, attn_mask, key_padding_mask, is_causal, need_weights,
            q_start_pos=q_start_pos,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        return self._decode_impl(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            layer_state=layer_state,
            **kwargs,
        )

    def _decode_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        has_static_cache = bool(
            isinstance(layer_state, dict) and "static_cache" in layer_state
        )
        use_paged_decode = (
            self.parent.use_paged_cache
            and (layer_state is not None)
            and (not self.parent.cross_attention)
            and is_causal
            and not has_static_cache
        )

        if use_paged_decode:
            q, k, v, kv_latent, provider, q_start_pos = (
                self.parent._prepare_qkv_with_provider(
                    query,
                    key,
                    value,
                    layer_state,
                    force_paged=True,
                )
            )
            if not isinstance(provider, PagedKVProvider):
                raise RuntimeError("Expected PagedKVProvider in paged decode mode.")
            cache = provider.cache
            cache_is_compacted = bool(
                torch.any(cache.seq_len != cache.logical_seq_len).item()
            )

            if need_weights:
                if self.parent.use_attention_matching_compaction or cache_is_compacted:
                    raise RuntimeError(
                        "attention-matching KV compaction does not support need_weights=True."
                    )
                k, v = provider.get_kv(k, v, kv_latent=kv_latent)
                k = self.parent._repeat_kv(k)
                v = self.parent._repeat_kv(v)
                T_k_full = k.size(2)
                cache_kpad = (
                    torch.arange(T_k_full, device=q.device)
                    .view(1, T_k_full)
                    .ge(cache.logical_seq_len.view(B, 1))
                )
                if key_padding_mask is not None:
                    key_padding_mask_full = key_padding_mask.bool()
                    if key_padding_mask_full.ndim != 2:
                        raise ValueError(
                            "key_padding_mask must be [B, T_k_total] for paged decode"
                        )
                    if key_padding_mask_full.shape[0] != B:
                        raise ValueError(
                            f"key_padding_mask batch dim {key_padding_mask_full.shape[0]} != B={B}"
                        )
                    if key_padding_mask_full.shape[1] != T_k_full:
                        raise ValueError(
                            f"key_padding_mask length {key_padding_mask_full.shape[1]} != total cached length {T_k_full}"
                        )
                    key_padding_mask_full = key_padding_mask_full | cache_kpad
                else:
                    key_padding_mask_full = cache_kpad

                out, weights = self.parent._compute_attention(
                    q,
                    k,
                    v,
                    attn_mask,
                    key_padding_mask_full,
                    is_causal,
                    need_weights,
                    q_start_pos=q_start_pos,
                )
                return (
                    self.parent._finalize_projected_output(out, B, T_q),
                    weights,
                    layer_state,
                )

            if T_k > 0:
                for b in range(B):
                    provider.append(
                        k[b],
                        v[b],
                        b,
                        kv_latent=(kv_latent[b] if kv_latent is not None else None),
                    )
                if self.parent._can_apply_attention_matching_compaction(
                    attn_mask=attn_mask,
                    key_padding_mask=key_padding_mask,
                    need_weights=need_weights,
                    cache=cache,
                    t_new=T_k,
                ):
                    self.parent._maybe_compact_paged_cache(
                        cache=cache,
                        q=q,
                        q_start_pos=q_start_pos,
                        t_new=T_k,
                    )

            total_k = int(cache.seq_len.max().item())
            if cache_is_compacted and (
                attn_mask is not None or key_padding_mask is not None
            ):
                raise RuntimeError(
                    "attention-matching KV compaction does not support external attention masks."
                )
            attn_mask_norm = None
            if attn_mask is not None:
                attn_mask_norm = self.parent._normalize_attn_mask(
                    attn_mask,
                    B=B,
                    H=self.parent.n_heads,
                    T_q=T_q,
                    T_k=total_k,
                )
            key_padding_mask_norm = None
            if key_padding_mask is not None:
                key_padding_mask_norm = key_padding_mask.bool()
                if key_padding_mask_norm.ndim != 2:
                    raise ValueError(
                        "key_padding_mask must be [B, T_k_total] for paged decode"
                    )
                if key_padding_mask_norm.shape[0] != B:
                    raise ValueError(
                        f"key_padding_mask batch dim {key_padding_mask_norm.shape[0]} != B={B}"
                    )
                if key_padding_mask_norm.shape[1] != total_k:
                    raise ValueError(
                        f"key_padding_mask length {key_padding_mask_norm.shape[1]} != total cached length {total_k}"
                    )

            out_bhqd = None
            if (
                attn_mask_norm is None
                and key_padding_mask_norm is None
                and (not self.parent.training)
                and self.parent.dropout_p == 0.0
                and (not cache_is_compacted)
                and (not self.parent.use_attention_matching_compaction)
            ):
                try:
                    out_bhqd = self.parent._triton_paged_decode(
                        q,
                        cache,
                        self.parent.n_rep,
                        self.parent.scale,
                        q_start_pos=q_start_pos,
                    )
                except Exception:
                    out_bhqd = None
            if out_bhqd is None:
                out_bhqd = self.parent._paged_stream_decode(
                    q_bhtd=q,
                    cache=cache,
                    kv_repeat=self.parent.n_rep,
                    scale=self.parent.scale,
                    dropout_p=self.parent.dropout_p,
                    training=self.parent.training,
                    is_causal=is_causal and not self.parent.cross_attention,
                    q_start_pos=q_start_pos,
                    attn_mask=attn_mask_norm,
                    key_padding_mask=key_padding_mask_norm,
                    mla_k_up_proj=self.parent.k_up_proj,
                    mla_v_up_proj=self.parent.v_up_proj,
                )
            out_bhqd = self.parent._apply_gated_attention(out_bhqd)
            return (
                self.parent._finalize_projected_output(out_bhqd, B, T_q),
                None,
                layer_state,
            )

        q, k, v, q_start_pos = self.parent._prepare_qkv_attention(
            query, key, value, layer_state
        )
        static_cache = (
            layer_state.get("static_cache")
            if isinstance(layer_state, dict)
            else None
        )
        if isinstance(static_cache, StaticKVCache):
            # The fixed cache has a wider key axis than the current query
            # chunk. Causality is rebuilt from q_start_pos in _compute_attention.
            if is_causal and attn_mask is not None and attn_mask.shape[-1] == T_q:
                attn_mask = None
            static_padding = torch.arange(
                k.size(2), device=k.device, dtype=torch.long
            ).view(1, -1) >= static_cache.lengths.view(B, 1)
            key_padding_mask = (
                static_padding
                if key_padding_mask is None
                else (key_padding_mask.bool() | static_padding)
            )
        out, weights = self.parent._compute_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            q_start_pos=q_start_pos,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state
