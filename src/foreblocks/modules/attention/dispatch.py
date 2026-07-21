"""Attention kernel selection and execution policy."""

from __future__ import annotations

from typing import Any, Protocol

import torch
import torch.nn.functional as F

from foreblocks.modules.attention.backends import ATTENTION_BACKENDS
from foreblocks.modules.attention.masking import build_attention_mask, to_additive_mask


class KernelDispatchContext(Protocol):
    """State and score transforms required by kernel dispatch."""

    alibi_bias: Any
    attn_implementation: str
    cross_attention: bool
    dropout_p: float
    logit_softcap: float | None
    scale: float
    training: bool
    use_alibi: bool
    use_learned_temp: bool
    use_multiscale_mask: bool

    def _apply_gated_attention(self, value: torch.Tensor) -> torch.Tensor: ...

    def _apply_learned_temperature(self, value: torch.Tensor) -> torch.Tensor: ...

    def _apply_logit_softcap(self, value: torch.Tensor) -> torch.Tensor: ...

    def _apply_masks(self, *args: Any, **kwargs: Any) -> torch.Tensor: ...

    def _apply_subquery_norm(self, value: torch.Tensor) -> torch.Tensor: ...

    def _create_multiscale_mask(self, *args: Any, **kwargs: Any) -> Any: ...

    def _dropout_weights(self, value: torch.Tensor) -> torch.Tensor: ...

    def _repeat_kv(self, value: torch.Tensor) -> torch.Tensor: ...


class AttentionKernelDispatcher:
    """Choose registered, SDPA, or eager attention from declared capabilities."""

    def __init__(self, context: KernelDispatchContext) -> None:
        self.context = context

    def _can_use_sdpa(
        self,
        *,
        need_weights: bool,
        q_start_pos: torch.Tensor | None,
    ) -> bool:
        context = self.context
        if context.attn_implementation == "eager":
            return False
        if not hasattr(F, "scaled_dot_product_attention"):
            return False
        if need_weights or q_start_pos is not None:
            return False
        return not (
            context.logit_softcap is not None
            or context.use_learned_temp
            or context.use_multiscale_mask
            or context.use_alibi
        )

    def _sdpa_attention_mask(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor | None:
        blocked = build_attention_mask(
            query=q,
            key_length=k.size(2),
            attention_mask=attn_mask,
            padding_mask=key_padding_mask,
            is_causal=is_causal and not self.context.cross_attention,
        )
        return to_additive_mask(blocked, dtype=q.dtype)

    def compute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        q_start_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        context = self.context
        batch, _, query_length, _ = q.shape
        key_length = k.size(2)

        compiler = getattr(torch, "compiler", None)
        compiling = bool(
            compiler is not None
            and hasattr(compiler, "is_compiling")
            and compiler.is_compiling()
        )
        backend = ATTENTION_BACKENDS.validate(
            context.attn_implementation,
            device_type=q.device.type,
            require_attention_weights=need_weights,
            require_gqa=q.size(1) != k.size(1),
            compiling=compiling,
        )
        if backend.runner is not None:
            if backend.mask_builder is None:
                raise RuntimeError(f"backend {backend.name!r} has no mask builder")
            blocked = backend.mask_builder(
                query=q,
                key_length=key_length,
                attention_mask=attn_mask,
                padding_mask=key_padding_mask,
                is_causal=is_causal and not context.cross_attention,
                cache_position=(
                    q_start_pos[:, None]
                    + torch.arange(query_length, device=q.device, dtype=torch.long)[
                        None, :
                    ]
                    if q_start_pos is not None
                    else None
                ),
            )
            backend_mask = (
                blocked
                if backend.supports_boolean_mask
                else to_additive_mask(blocked, dtype=q.dtype)
            )
            result = backend.runner(
                q,
                k,
                v,
                attention_mask=backend_mask,
                dropout_p=context.dropout_p if context.training else 0.0,
                scale=context.scale,
                need_weights=need_weights,
            )
            out, weights = result if isinstance(result, tuple) else (result, None)
            return self._finalize(out), weights

        if self._can_use_sdpa(
            need_weights=need_weights,
            q_start_pos=q_start_pos,
        ):
            sdpa_mask = self._sdpa_attention_mask(
                q, k, attn_mask, key_padding_mask, is_causal
            )
            try:
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=sdpa_mask,
                    dropout_p=context.dropout_p if context.training else 0.0,
                    is_causal=False,
                    enable_gqa=q.size(1) != k.size(1),
                    scale=context.scale,
                )
            except (RuntimeError, TypeError) as exc:
                if context.attn_implementation == "sdpa":
                    raise RuntimeError(
                        "requested SDPA attention backend failed"
                    ) from exc
            else:
                return self._finalize(out), None

        if k.size(1) != q.size(1):
            k = context._repeat_kv(k)
            v = context._repeat_kv(v)

        scores = torch.matmul(q, k.transpose(-2, -1)) * context.scale
        scores = context._apply_learned_temperature(scores)
        scores = context._apply_logit_softcap(scores)

        if is_causal and not context.cross_attention:
            if q_start_pos is None:
                causal_mask = torch.triu(
                    torch.ones(
                        query_length,
                        key_length,
                        device=q.device,
                        dtype=torch.bool,
                    ),
                    diagonal=1,
                )
                scores = scores.masked_fill(
                    causal_mask.view(1, 1, query_length, key_length),
                    float("-inf"),
                )
            else:
                if q_start_pos.ndim != 1 or q_start_pos.shape[0] != batch:
                    raise ValueError(
                        f"q_start_pos must be [B], got {tuple(q_start_pos.shape)}"
                    )
                q_pos = q_start_pos.to(device=q.device, dtype=torch.long).view(
                    batch, 1, 1, 1
                ) + torch.arange(
                    query_length, device=q.device, dtype=torch.long
                ).view(1, 1, query_length, 1)
                k_pos = torch.arange(
                    key_length, device=q.device, dtype=torch.long
                ).view(1, 1, 1, key_length)
                scores = scores.masked_fill(k_pos > q_pos, float("-inf"))

        scores = context._apply_masks(scores, attn_mask, key_padding_mask)
        if context.use_alibi and context.alibi_bias is not None:
            scores = scores + context.alibi_bias(
                query_length, key_length, device=q.device
            )
        if context.use_multiscale_mask:
            multiscale_mask = context._create_multiscale_mask(
                query_length,
                key_length,
                device=q.device,
                is_causal=is_causal,
            )
            if multiscale_mask is not None:
                scores = scores.masked_fill(
                    (~multiscale_mask).view(
                        1, 1, query_length, key_length
                    ),
                    float("-inf"),
                )

        weights = context._dropout_weights(F.softmax(scores, dim=-1))
        out = torch.matmul(weights, v)
        return self._finalize(out), weights if need_weights else None

    def _finalize(self, out: torch.Tensor) -> torch.Tensor:
        out = self.context._apply_subquery_norm(out)
        return self.context._apply_gated_attention(out)


__all__ = ["AttentionKernelDispatcher", "KernelDispatchContext"]
