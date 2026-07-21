"""foreblocks.modules.attention.variants.spectral.

Dispatch to frequency-domain or wavelet-domain attention mechanisms.

Routes attention computation into the Fourier or Haar wavelet coefficient domain
instead of the standard time-domain dot-product. Use when frequency periodicity
or multi-scale wavelet patterns are expected in the data, such as seasonal time
series or signals with scale-invariant structure.

Core API:
- SpectralAttentionImpl: dispatcher for frequency and DWT attention backends

"""

import torch

from foreblocks.modules.attention.variants.base import AttentionContext


class SpectralAttentionImpl:
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
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        if self.context.attention_type == "frequency":
            out, weights = self.context.freq_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )
            return out, weights, None

        if self.context.attention_type == "dwt":
            out, weights = self.context.dwt_attention(
                query,
                key,
                value,
                attn_mask,
                key_padding_mask,
                is_causal,
                need_weights,
            )
            return out, weights, None

        out, weights = self.context.freq_attention(
            query,
            key,
            value,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
        )
        return out, weights, None
