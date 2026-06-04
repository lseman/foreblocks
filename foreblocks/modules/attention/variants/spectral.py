"""Spectral attention dispatcher — frequency- and wavelet-domain attention.

This is a thin router that forwards to the parent's spectral attention module
according to ``parent.attention_type``:

    * ``"frequency"`` → :class:`FrequencyAttention`, a re-implementation of the
      ``FourierCrossAttention`` block from FEDformer (Zhou et al., "FEDformer:
      Frequency Enhanced Decomposed Transformer for Long-term Series
      Forecasting", ICML 2022, arXiv:2201.12740).
    * ``"dwt"`` → :class:`DWTAttention`, an analogous attention computed in the
      Haar wavelet-coefficient domain.

Any other type falls through to the frequency module. The actual algorithms,
math, and references live in the respective module docstrings
(:mod:`..modules.frequency_att`, :mod:`..modules.dwt_att`); this class only
handles dispatch.
"""

import torch


class SpectralAttentionImpl:
    """Dispatcher to the parent's frequency/wavelet attention (see module docstring)."""

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
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
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
