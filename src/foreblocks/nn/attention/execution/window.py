"""Sparse window predicates for compiled FlexAttention, without dense masks."""

from collections import OrderedDict

import torch

from foreblocks.nn.attention.execution.backends import (
    ATTENTION_BACKENDS,
    _compiled_flex_attention,
)
from foreblocks.nn.attention.preparation.masking import normalize_blocked_mask


class SparseWindowAttention:
    """Cache static block masks; dynamic offsets/padding always rebuild them."""

    def __init__(self):
        self._masks = OrderedDict()

    def supports(self, context, q, v, need_weights):
        return (
            context.attn_implementation in {"auto", "flex_attention"}
            and ATTENTION_BACKENDS.get("flex_attention").available
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and min(q.size(-1), v.size(-1)) >= 16
            and not need_weights
            and (not context.training or context.dropout_p == 0)
        )

    def __call__(
        self,
        q,
        k,
        v,
        *,
        window,
        causal,
        scale,
        q_start_pos=None,
        attention_mask=None,
        padding_mask=None,
        dilation=1,
        span=None,
    ):
        from torch.nn.attention.flex_attention import create_block_mask

        batch, heads, tq, _ = q.shape
        tk = k.size(2)
        blocked = None
        if attention_mask is not None:
            blocked = normalize_blocked_mask(
                attention_mask,
                batch_size=batch,
                num_heads=heads,
                query_length=tq,
                key_length=tk,
            )
        if padding_mask is not None and padding_mask.shape != (batch, tk):
            raise ValueError("padding_mask must have shape [B, key_length]")
        if q_start_pos is not None:
            q_start_pos = q_start_pos.to(device=q.device).reshape(batch)

        def keep(b, h, qi, ki):
            safe_q = qi.clamp(max=tq - 1)
            safe_k = ki.clamp(max=tk - 1)
            position = qi if q_start_pos is None else qi + q_start_pos[b]
            distance = position - ki
            if span is None:
                allowed = (
                    (distance >= 0) & (distance < window)
                    if causal
                    else distance.abs() <= window // 2
                )
            else:
                allowed = (distance.abs() < window) | (
                    (distance.abs() < span) & (distance.abs() % dilation == 0)
                )
                if causal:
                    allowed = allowed & (distance >= 0)
            if blocked is not None:
                allowed = allowed & ~blocked[b, h, safe_q, safe_k]
            if padding_mask is not None:
                allowed = allowed & ~padding_mask[b, safe_k].bool()
            return allowed & (qi < tq) & (ki < tk)

        static = q_start_pos is None and attention_mask is None and padding_mask is None
        key = (q.device, tq, tk, window, causal, dilation, span)
        block_mask = self._masks.get(key) if static else None
        if block_mask is None:
            block_mask = create_block_mask(
                keep,
                None if static else batch,
                None if static else heads,
                tq,
                tk,
                device=q.device,
                _compile=True,
            )
            if static:
                self._masks[key] = block_mask
                if len(self._masks) > 8:
                    self._masks.popitem(last=False)
        return _compiled_flex_attention()(
            q,
            k,
            v,
            block_mask=block_mask,
            scale=scale,
            enable_gqa=q.size(1) != k.size(1),
        )
