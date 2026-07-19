"""Common cache protocol used by transformer attention implementations."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class TransformerCache(Protocol):
    is_compileable: bool

    def get_seq_length(self, batch_idx: int | None = None) -> int: ...

    def get_seq_lengths(self) -> torch.Tensor: ...

    def get_max_cache_shape(self) -> int: ...

    def reset(self) -> None: ...

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None: ...

    def crop(self, max_length: int) -> None: ...


def map_cache_state(value, tensor_fn):
    """Recursively transform tensors and cache objects in incremental state."""
    from foreblocks.modules.attention.cache.kv import StaticKVCache
    from foreblocks.modules.attention.cache.paged import PagedKVCache

    if isinstance(value, (StaticKVCache, PagedKVCache)):
        return value.to(tensor_fn(value.get_seq_lengths()).device)
    if isinstance(value, torch.Tensor):
        return tensor_fn(value)
    if isinstance(value, dict):
        return {key: map_cache_state(item, tensor_fn) for key, item in value.items()}
    if isinstance(value, list):
        return [map_cache_state(item, tensor_fn) for item in value]
    if isinstance(value, tuple):
        return tuple(map_cache_state(item, tensor_fn) for item in value)
    return value


def cache_state_dict(value):
    """Create a CPU-portable snapshot of nested decoder cache state."""
    from foreblocks.modules.attention.cache.kv import StaticKVCache
    from foreblocks.modules.attention.cache.paged import PagedKVCache

    if isinstance(value, StaticKVCache):
        return {"__cache_type__": "static", "state": value.state_dict()}
    if isinstance(value, PagedKVCache):
        return {"__cache_type__": "paged", "state": value.state_dict()}
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cache_state_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cache_state_dict(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cache_state_dict(item) for item in value)
    return value


def load_cache_state_dict(value, *, device=None):
    """Restore a nested decoder cache snapshot on the requested device."""
    from foreblocks.modules.attention.cache.kv import StaticKVCache
    from foreblocks.modules.attention.cache.paged import PagedKVCache

    if isinstance(value, dict) and "__cache_type__" in value:
        cache_cls = StaticKVCache if value["__cache_type__"] == "static" else PagedKVCache
        return cache_cls.from_state_dict(value["state"], device=device)
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: load_cache_state_dict(item, device=device) for key, item in value.items()}
    if isinstance(value, list):
        return [load_cache_state_dict(item, device=device) for item in value]
    if isinstance(value, tuple):
        return tuple(load_cache_state_dict(item, device=device) for item in value)
    return value


__all__ = [
    "TransformerCache", "cache_state_dict", "load_cache_state_dict", "map_cache_state"
]
