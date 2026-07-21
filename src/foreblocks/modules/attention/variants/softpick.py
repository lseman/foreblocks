"""foreblocks.modules.attention.variants.softpick.

Attention with rectified softmax — eliminates attention sinks via unnormalized weights.

Softpick replaces the standard softmax normalization with a rectified variant whose
weights need not sum to one, removing the forced probability-mass allocation that
produces attention sinks and massive activations. Use when attention sink behavior
is degrading model quality on long sequences.

Core API:
- SoftpickAttentionImpl: Triton-backed or fallback Softpick attention

"""

import warnings

import torch

from foreblocks.modules.attention.variants.base import AttentionContext


class SoftpickAttentionImpl:
    def __init__(self, context: AttentionContext):
        self.context = context

    def forward(
        self,
        query,
        key,
        value,
        attn_mask,
        key_padding_mask,
        is_causal,
        need_weights,
        layer_state=None,
        cu_seqlens=None,
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if not self.context.backends.get("softpick"):
            warnings.warn(
                "[MultiAttention] SoftPick unavailable, falling back to standard."
            )
            return self.context._fallback_standard.forward(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state=layer_state,
                cu_seqlens=cu_seqlens,
            )

        try:
            from foreblocks.models.transformer.third_party.flash_softpick_attn import (
                parallel_softpick_attn,
            )

            B, T_q, _ = query.shape
            q, k, v, _ = self.context._prepare_qkv_attention(
                query, key, value, layer_state
            )

            if cu_seqlens is None:
                out = parallel_softpick_attn(
                    q,
                    k,
                    v,
                    scale=self.context.scale,
                    cu_seqlens=None,
                    head_first=False,
                )
                out = out.contiguous().view(B, T_q, self.context.d_model)
            else:
                T_k = k.size(2)
                q_flat = q.reshape(B * T_q, self.context.n_heads, self.context.head_dim)
                k_flat = k.reshape(B * T_k, self.context.n_heads, self.context.head_dim)
                v_flat = v.reshape(B * T_k, self.context.n_heads, self.context.head_dim)
                out = parallel_softpick_attn(
                    q_flat,
                    k_flat,
                    v_flat,
                    scale=self.context.scale,
                    cu_seqlens=cu_seqlens,
                    head_first=True,
                )
                out = (
                    out.view(B, T_q, self.context.n_heads, self.context.head_dim)
                    .contiguous()
                    .view(B, T_q, self.context.d_model)
                )

            return self.context.out_proj(self.context.dropout(out)), None, layer_state

        except Exception as e:
            warnings.warn(
                f"[MultiAttention] SoftPick failed ({e}), falling back to standard."
            )
            return self.context._fallback_standard.forward(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
                layer_state=layer_state,
                cu_seqlens=cu_seqlens,
            )
