import warnings
from typing import Optional, Tuple

import torch


class SoftpickAttentionImpl:
    def __init__(self, parent):
        self.parent = parent

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        if not self.parent.backends.get("softpick"):
            warnings.warn(
                "[MultiAttention] SoftPick unavailable, falling back to standard."
            )
            return self.parent._fallback_standard.forward(
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
            from ...third_party.flash_softpick_attn import parallel_softpick_attn

            B, T_q, _ = query.shape
            q, k, v = self.parent._prepare_qkv_attention(query, key, value, layer_state)

            if cu_seqlens is None:
                out = parallel_softpick_attn(
                    q,
                    k,
                    v,
                    scale=self.parent.scale,
                    cu_seqlens=None,
                    head_first=False,
                )
                out = out.contiguous().view(B, T_q, self.parent.d_model)
            else:
                T_k = k.size(2)
                q_flat = q.reshape(B * T_q, self.parent.n_heads, self.parent.head_dim)
                k_flat = k.reshape(B * T_k, self.parent.n_heads, self.parent.head_dim)
                v_flat = v.reshape(B * T_k, self.parent.n_heads, self.parent.head_dim)
                out = parallel_softpick_attn(
                    q_flat,
                    k_flat,
                    v_flat,
                    scale=self.parent.scale,
                    cu_seqlens=cu_seqlens,
                    head_first=True,
                )
                out = (
                    out.view(B, T_q, self.parent.n_heads, self.parent.head_dim)
                    .contiguous()
                    .view(B, T_q, self.parent.d_model)
                )

            return self.parent.out_proj(self.parent.dropout(out)), None, layer_state

        except Exception as e:
            warnings.warn(
                f"[MultiAttention] SoftPick failed ({e}), falling back to standard."
            )
            return self.parent._fallback_standard.forward(
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
