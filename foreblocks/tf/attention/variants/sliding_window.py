from typing import Optional, Tuple

import torch
import torch.nn.functional as F


class SlidingWindowAttentionImpl:
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        B, T_q, _ = query.shape
        q, k, v = self.parent._prepare_qkv_attention(query, key, value, layer_state)

        if (
            self.parent.use_flash_sliding
            and self.parent.backends.get("sdp")
            and not need_weights
        ):
            try:
                window_mask = self.parent._create_sliding_window_mask(
                    q.size(2),
                    k.size(2),
                    q.device,
                    is_causal,
                )
                combined = window_mask.view(1, 1, q.size(2), k.size(2))
                if attn_mask is not None:
                    combined = combined | self.parent._normalize_attn_mask(
                        attn_mask, B, self.parent.n_heads, q.size(2), k.size(2)
                    )

                if key_padding_mask is not None:
                    combined = (
                        combined | key_padding_mask.view(B, 1, 1, k.size(2)).bool()
                    )

                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=combined,
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

        out, weights = self.manual(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            apply_gate=True,
        )
        return self.parent._finalize_projected_output(out, B, T_q), weights, layer_state

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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, H, T_q, D = q.shape
        T_k = k.size(2)

        if T_q <= self.parent.chunk_size:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.parent.scale
            window_mask = self.parent._create_sliding_window_mask(
                T_q,
                T_k,
                q.device,
                is_causal,
            )
            scores = scores.masked_fill(window_mask.view(1, 1, T_q, T_k), float("-inf"))
            scores = self.parent._apply_masks(scores, attn_mask, key_padding_mask)

            weights = F.softmax(scores, dim=-1)
            weights = self.parent._dropout_weights(weights)

            out = torch.matmul(weights, v)
            if apply_gate:
                out = self.parent._apply_gated_attention(out)
            return out, (weights if need_weights else None)

        output = torch.zeros_like(q)
        for i in range(0, T_q, self.parent.chunk_size):
            end_i = min(i + self.parent.chunk_size, T_q)

            if is_causal:
                start_k = max(0, i - self.parent.window_size + 1)
                end_k = end_i
            else:
                center = (i + end_i) // 2
                half = self.parent.window_size // 2
                start_k = max(0, center - half)
                end_k = min(T_k, center + half + 1)

            q_chunk = q[:, :, i:end_i]
            k_chunk = k[:, :, start_k:end_k]
            v_chunk = v[:, :, start_k:end_k]

            scores = (
                torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * self.parent.scale
            )

            q_pos = torch.arange(i, end_i, device=q.device).unsqueeze(1)
            k_pos = torch.arange(start_k, end_k, device=q.device).unsqueeze(0)
            if is_causal:
                local_mask = (k_pos > q_pos) | (
                    k_pos < (q_pos - self.parent.window_size + 1)
                )
            else:
                half = self.parent.window_size // 2
                local_mask = (k_pos < (q_pos - half)) | (k_pos > (q_pos + half))

            scores = scores.masked_fill(
                local_mask.view(1, 1, end_i - i, end_k - start_k),
                float("-inf"),
            )

            if key_padding_mask is not None:
                chunk_pad = key_padding_mask[:, start_k:end_k]
                scores = scores.masked_fill(chunk_pad.view(B, 1, 1, -1), float("-inf"))

            chunk_attn = self.parent._slice_attn_mask(
                attn_mask=attn_mask,
                B=B,
                H=H,
                q_start=i,
                q_end=end_i,
                k_start=start_k,
                k_end=end_k,
                T_q_full=T_q,
                T_k_full=T_k,
            )
            if chunk_attn is not None:
                scores = scores.masked_fill(chunk_attn, float("-inf"))

            weights = F.softmax(scores, dim=-1)
            weights = self.parent._dropout_weights(weights)
            output[:, :, i:end_i] = torch.matmul(weights, v_chunk)

        if apply_gate:
            output = self.parent._apply_gated_attention(output)
        return output, None
