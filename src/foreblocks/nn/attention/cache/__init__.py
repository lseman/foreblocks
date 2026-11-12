"""Lazy public exports for foreblocks.nn.attention.cache."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.attention.cache.base import (
        KVCacheProtocol as KVCacheProtocol,
        cache_state_dict as cache_state_dict,
        load_cache_state_dict as load_cache_state_dict,
        map_cache_state as map_cache_state,
    )
    from foreblocks.nn.attention.cache.decode_stream import (
        paged_stream_decode_standard as paged_stream_decode_standard,
    )
    from foreblocks.nn.attention.cache.kv import (
        DenseKVProvider as DenseKVProvider,
        KVProvider as KVProvider,
        PagedKVProvider as PagedKVProvider,
        StaticKVCache as StaticKVCache,
        StaticKVProvider as StaticKVProvider,
    )
    from foreblocks.nn.attention.cache.paged import PagedKVCache as PagedKVCache
    from foreblocks.nn.attention.cache.storage import (
        DensePagedStorage as DensePagedStorage,
        LatentPagedStorage as LatentPagedStorage,
        PagedStorage as PagedStorage,
    )

_EXPORTS = {
    "DenseKVProvider": "foreblocks.nn.attention.cache.kv",
    "DensePagedStorage": "foreblocks.nn.attention.cache.storage",
    "KVCacheProtocol": "foreblocks.nn.attention.cache.base",
    "KVProvider": "foreblocks.nn.attention.cache.kv",
    "LatentPagedStorage": "foreblocks.nn.attention.cache.storage",
    "PagedKVCache": "foreblocks.nn.attention.cache.paged",
    "PagedKVProvider": "foreblocks.nn.attention.cache.kv",
    "PagedStorage": "foreblocks.nn.attention.cache.storage",
    "StaticKVCache": "foreblocks.nn.attention.cache.kv",
    "StaticKVProvider": "foreblocks.nn.attention.cache.kv",
    "cache_state_dict": "foreblocks.nn.attention.cache.base",
    "load_cache_state_dict": "foreblocks.nn.attention.cache.base",
    "map_cache_state": "foreblocks.nn.attention.cache.base",
    "paged_stream_decode_standard": "foreblocks.nn.attention.cache.decode_stream",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
