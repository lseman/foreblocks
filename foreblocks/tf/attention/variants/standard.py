from typing import Optional, Tuple

import torch

from ..kv import PagedKVProvider


class StandardAttentionImpl:
    def __init__(self, parent):
        self.parent = parent

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool,
        need_weights: bool,
        layer_state=None,
        **_,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        B, T_q, _ = query.shape
        T_k = key.shape[1]

        use_paged_decode = (
            self.parent.use_paged_cache
            and (layer_state is not None)
            and (not self.parent.cross_attention)
            and is_causal
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

            if need_weights:
                k, v = provider.get_kv(k, v, kv_latent=kv_latent)
                k = self.parent._repeat_kv(k)
                v = self.parent._repeat_kv(v)
                T_k_full = k.size(2)
                cache_kpad = (
                    torch.arange(T_k_full, device=q.device)
                    .view(1, T_k_full)
                    .ge(cache.seq_len.view(B, 1))
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

            total_k = int(cache.seq_len.max().item())
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

        q, k, v = self.parent._prepare_qkv_attention(query, key, value, layer_state)
        out, weights = self.parent._compute_attention(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state
