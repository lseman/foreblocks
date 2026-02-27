from typing import Optional, Tuple

import torch


class SpectralAttentionImpl:
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
        **_,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        if self.parent.attention_type == "frequency":
            out, weights = self.parent.freq_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )
            return out, weights, None

        if self.parent.attention_type == "dwt":
            out, weights = self.parent.dwt_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )
            return out, weights, None

        out, weights = self.parent.freq_attention(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return out, weights, None
