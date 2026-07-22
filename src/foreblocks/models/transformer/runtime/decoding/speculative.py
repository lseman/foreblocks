"""Speculative decoding with typed cache rollback."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from foreblocks.models.transformer.runtime.contracts import DecoderProtocol
from foreblocks.models.transformer.runtime.state import DecoderState


def speculative_decode(
    decoder: DecoderProtocol,
    draft_tokens: torch.Tensor,
    memory: torch.Tensor,
    state: DecoderState,
    *,
    verifier_fn: Callable[[torch.Tensor, torch.Tensor], int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, DecoderState, int]:
    caches = [layer.self_attention.cache for layer in state.layers]
    start_lengths = [
        cache.get_seq_length() if cache is not None else None for cache in caches
    ]
    output, state = decoder.forward_multi_step(draft_tokens, memory, state, **kwargs)
    accepted = (
        int(verifier_fn(output, draft_tokens))
        if verifier_fn is not None
        else draft_tokens.size(1)
    )
    accepted = max(0, min(accepted, draft_tokens.size(1)))
    if accepted != draft_tokens.size(1):
        for start, layer in zip(start_lengths, state.layers, strict=True):
            cache = layer.self_attention.cache
            if start is not None and cache is not None:
                cache.crop(start + accepted)
    return output[:, :accepted], state, accepted


__all__ = ["speculative_decode"]
