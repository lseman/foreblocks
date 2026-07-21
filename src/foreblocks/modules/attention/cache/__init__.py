"""foreblocks.modules.attention.cache.

Package initializer that exposes the public symbols for this namespace.
It belongs to the attention key-value cache implementations area of Foreblocks.

"""

from foreblocks.modules.attention.cache.base import (
    KVCacheProtocol,
    cache_state_dict,
    load_cache_state_dict,
    map_cache_state,
)
from foreblocks.modules.attention.cache.decode_stream import (
    paged_stream_decode_standard,
)
from foreblocks.modules.attention.cache.kv import (
    DenseKVProvider,
    KVProvider,
    PagedKVProvider,
    StaticKVCache,
    StaticKVProvider,
)
from foreblocks.modules.attention.cache.paged import PagedKVCache
from foreblocks.modules.attention.cache.storage import (
    DensePagedStorage,
    LatentPagedStorage,
    PagedStorage,
)

__all__ = [
    "DenseKVProvider",
    "DensePagedStorage",
    "KVCacheProtocol",
    "KVProvider",
    "LatentPagedStorage",
    "PagedKVCache",
    "PagedKVProvider",
    "PagedStorage",
    "StaticKVCache",
    "StaticKVProvider",
    "cache_state_dict",
    "load_cache_state_dict",
    "map_cache_state",
    "paged_stream_decode_standard",
]
