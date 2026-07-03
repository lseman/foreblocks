"""foreblocks.modules.attention.variants.dilated_sliding_window.

Dilated sliding-window attention for long-context sparse coverage.

Combines a local attention window with strided keys at a configurable dilation
rate, inspired by LongNet. Nearby tokens are dense, older tokens are sampled
sparsely at dilation steps, increasing the effective receptive field without
full quadratic attention.

Core API:
- DilatedSlidingWindowAttentionImpl: local window + dilated long-range keys

"""

import torch
import torch.nn.functional as F


class DilatedSlidingWindowAttentionImpl:
    """Local attention plus dilated long-range keys."""

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
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T_q, _ = query.shape
        q, k, v, _ = self.parent._prepare_qkv_attention(query, key, value, layer_state)
        mask = self._create_dilated_mask(T_q, k.size(2), q.device, is_causal)

        if self.parent.backends.get("sdp") and not need_weights:
            try:
                combined = mask.view(1, 1, T_q, k.size(2))
                if attn_mask is not None:
                    combined = combined | self.parent._normalize_attn_mask(
                        attn_mask,
                        B,
                        self.parent.n_heads,
                        T_q,
                        k.size(2),
                    )
                if key_padding_mask is not None:
                    combined = (
                        combined
                        | key_padding_mask.view(
                            B,
                            1,
                            1,
                            k.size(2),
                        ).bool()
                    )

                additive_mask = torch.zeros(
                    combined.shape,
                    device=q.device,
                    dtype=torch.float32,
                ).masked_fill(combined, float("-inf"))

                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=additive_mask,
                    dropout_p=self.parent.dropout_p if self.parent.training else 0.0,
                    is_causal=False,
                )
                out = self.parent._apply_gated_attention(out)
                return (
                    self.parent._finalize_projected_output(out, B, T_q),
                    None,
                    layer_state,
                )
            except Exception:
                pass

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.parent.scale
        scores = scores.masked_fill(mask.view(1, 1, T_q, k.size(2)), float("-inf"))
        scores = self.parent._apply_masks(scores, attn_mask, key_padding_mask)

        weights = F.softmax(scores, dim=-1)
        weights = torch.where(
            torch.isfinite(weights),
            weights,
            torch.zeros_like(weights),
        )
        weights = self.parent._dropout_weights(weights)
        out = torch.matmul(weights, v)
        out = self.parent._apply_gated_attention(out)
        return (
            self.parent._finalize_projected_output(out, B, T_q),
            weights if need_weights else None,
            layer_state,
        )

    def _create_dilated_mask(
        self,
        T_q: int,
        T_k: int,
        device: torch.device,
        is_causal: bool,
    ) -> torch.Tensor:
        q_pos = torch.arange(T_q, device=device).view(T_q, 1)
        k_pos = torch.arange(T_k, device=device).view(1, T_k)
        distance = q_pos - k_pos

        local = distance.abs() < self.parent.window_size
        if is_causal and not self.parent.cross_attention:
            valid_direction = distance >= 0
            in_span = distance < self.parent.dilated_window_size
            dilated = (distance % self.parent.attention_dilation) == 0
        else:
            valid_direction = torch.ones_like(distance, dtype=torch.bool)
            in_span = distance.abs() < self.parent.dilated_window_size
            dilated = (distance.abs() % self.parent.attention_dilation) == 0

        keep = valid_direction & (local | (in_span & dilated))
        return ~keep
