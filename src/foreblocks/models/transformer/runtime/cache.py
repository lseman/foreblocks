"""Persistence and batch reordering for typed decoder cache state."""

from __future__ import annotations

from os import PathLike
from typing import Any, cast

import torch

from foreblocks.models.transformer.runtime.contracts import DecoderProtocol
from foreblocks.models.transformer.runtime.state import (
    AttentionCacheState,
    DecoderLayerState,
    DecoderState,
)
from foreblocks.modules.attention.cache.base import (
    cache_state_dict,
    load_cache_state_dict,
)
from foreblocks.modules.attention.cache.kv import StaticKVCache
from foreblocks.modules.attention.cache.paged import PagedKVCache


class DecoderCacheManager:
    def __init__(self, decoder: DecoderProtocol) -> None:
        self.decoder = decoder

    def reorder(self, state: DecoderState, beam_idx: torch.Tensor) -> DecoderState:
        def select(value: Any) -> Any:
            if isinstance(value, (StaticKVCache, PagedKVCache)):
                return value.batch_select(cast(torch.LongTensor, beam_idx))
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if beam_idx.numel() and value.size(0) > int(beam_idx.max().item()):
                    return value.index_select(0, beam_idx.to(value.device))
                return value
            if isinstance(value, AttentionCacheState):
                return AttentionCacheState(
                    {key: select(item) for key, item in value.items()}
                )
            if isinstance(value, DecoderLayerState):
                return DecoderLayerState(
                    self_attn=select(value.self_attention),
                    cross_attn=select(value.cross_attention),
                )
            if isinstance(value, dict):
                return {key: select(item) for key, item in value.items()}
            if isinstance(value, list):
                return [select(item) for item in value]
            if isinstance(value, tuple):
                return tuple(select(item) for item in value)
            return value

        selected = {key: select(value) for key, value in state.items()}
        return DecoderState(selected)

    def state_dict(self, state: DecoderState) -> dict[str, Any]:
        return cast(dict[str, Any], cache_state_dict(state))

    def load_state_dict(
        self, state: dict[str, Any], *, device: torch.device | str | None = None
    ) -> DecoderState:
        if device is None:
            device = next(self.decoder.parameters()).device
        loaded = load_cache_state_dict(state, device=device)
        return DecoderState.from_mapping(loaded, num_layers=self.decoder.num_layers)

    def offload(self, state: DecoderState) -> dict[str, Any]:
        return self.state_dict(state)

    def save(self, state: DecoderState, path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(state), path)

    def load(
        self, path: str | PathLike[str], *, device: torch.device | str | None = None
    ) -> DecoderState:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_state_dict(snapshot, device=device)


__all__ = ["DecoderCacheManager"]
