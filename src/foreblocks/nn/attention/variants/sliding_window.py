"""foreblocks.nn.attention.variants.sliding_window.

Linear-cost local attention via fixed-size query windows.

Restricts each query to attend only to keys within a ``window_size`` neighbourhood,
reducing attention from O(T²) to O(T·window_size) in sequence length. Use when
processing long sequences where full attention is prohibitively expensive, or when
dependencies are predominantly local (e.g., time series with short-range correlations).

Core API:
- SlidingWindowAttentionImpl: sliding-window attention with chunked and causal modes

"""

import torch
import torch.nn.functional as F

from foreblocks.nn.attention.variants.base import AttentionOwner


class SlidingWindowAttentionImpl:
    def __init__(self, context: AttentionOwner):
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
        **_,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict | None]:
        B, T_q, _ = query.shape
        q, k, v, q_start_pos = self.context._prepare_qkv_attention(
            query, key, value, layer_state
        )
        k = self.context._repeat_kv(k)
        v = self.context._repeat_kv(v)

        if (
            self.context.use_flash_sliding
            and self.context.backends.get("sdp")
            and not need_weights
        ):
            try:
                window_mask = self.context._create_sliding_window_mask(
                    q.size(2),
                    k.size(2),
                    q.device,
                    is_causal,
                    q_start_pos=q_start_pos,
                )
                combined = (
                    window_mask if window_mask.ndim == 4 else window_mask[None, None]
                )
                if attn_mask is not None:
                    combined = combined | self.context._normalize_attn_mask(
                        attn_mask, B, self.context.n_heads, q.size(2), k.size(2)
                    )

                if key_padding_mask is not None:
                    combined = (
                        combined | key_padding_mask.view(B, 1, 1, k.size(2)).bool()
                    )

                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    # Package masks mark blocked entries; SDPA marks allowed ones.
                    attn_mask=~combined,
                    dropout_p=self.context.dropout_p if self.context.training else 0.0,
                    is_causal=False,
                )
                out = self.context._apply_gated_attention(out)
                return (
                    self.context._finalize_projected_output(out, B, T_q),
                    None,
                    layer_state,
                )
            except Exception:
                pass

        out, weights = self.manual(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            apply_gate=True,
            q_start_pos=q_start_pos,
        )
        return (
            self.context._finalize_projected_output(out, B, T_q),
            weights,
            layer_state,
        )

    def manual(
        self,
        q,
        k,
        v,
        attn_mask,
        key_padding_mask,
        is_causal,
        need_weights,
        apply_gate: bool = True,
        q_start_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)
        # NSA also calls this method directly with compact KV heads.
        if k.size(1) != H:
            k = self.context._repeat_kv(k)
            v = self.context._repeat_kv(v)
        if q_start_pos is None:
            starts = torch.zeros(B, device=q.device, dtype=torch.long)
            first_start = last_start = 0
        else:
            starts = q_start_pos.to(device=q.device, dtype=torch.long).reshape(-1)
            first_start = int(starts.min().item())
            last_start = int(starts.max().item())

        output = torch.zeros_like(q)
        full_weights = q.new_zeros(B, H, T_q, T_k) if need_weights else None
        half_window = self.context.window_size // 2
        for start in range(0, T_q, self.context.chunk_size):
            stop = min(start + self.context.chunk_size, T_q)
            # Gather the union of all query windows in this chunk and batch.
            # Centering one window on the chunk would omit boundary neighbors.
            left = self.context.window_size - 1 if is_causal else half_window
            right = 0 if is_causal else half_window
            key_start = max(0, first_start + start - left)
            key_stop = min(T_k, last_start + stop + right)
            if key_start >= key_stop:
                continue

            scores = (
                torch.matmul(
                    q[:, :, start:stop], k[:, :, key_start:key_stop].transpose(-2, -1)
                )
                * self.context.scale
            )
            local_mask = self.context._create_sliding_window_mask(
                stop - start,
                key_stop - key_start,
                q.device,
                is_causal,
                q_start_pos=starts + start - key_start,
            )
            scores = scores.masked_fill(local_mask, float("-inf"))
            chunk_attn = self.context._slice_attn_mask(
                attn_mask=attn_mask,
                B=B,
                H=H,
                q_start=start,
                q_end=stop,
                k_start=key_start,
                k_end=key_stop,
                T_q_full=T_q,
                T_k_full=T_k,
            )
            chunk_padding = (
                key_padding_mask[:, key_start:key_stop]
                if key_padding_mask is not None
                else None
            )
            scores = self.context._apply_masks(scores, chunk_attn, chunk_padding)
            # Avoid NaN softmax intermediates as well as NaN outputs: masking
            # them after softmax would still poison gradients for empty rows.
            empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
            weights = F.softmax(scores.masked_fill(empty, 0.0), dim=-1).masked_fill(
                empty, 0.0
            )
            weights = self.context._dropout_weights(weights)
            output[:, :, start:stop] = torch.matmul(
                weights, v[:, :, key_start:key_stop]
            )
            if full_weights is not None:
                full_weights[:, :, start:stop, key_start:key_stop] = weights

        if apply_gate:
            output = self.context._apply_gated_attention(output)
        return output, full_weights
