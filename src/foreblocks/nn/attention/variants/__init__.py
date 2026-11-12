"""Lazy public exports for foreblocks.nn.attention.variants."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.attention.variants.base import AttentionImpl as AttentionImpl
    from foreblocks.nn.attention.variants.dilated_sliding_window import (
        DilatedSlidingWindowAttentionImpl as DilatedSlidingWindowAttentionImpl,
    )
    from foreblocks.nn.attention.variants.moba import (
        MoBAAttentionImpl as MoBAAttentionImpl,
    )
    from foreblocks.nn.attention.variants.nsa import (
        NSAAttentionImpl as NSAAttentionImpl,
    )
    from foreblocks.nn.attention.variants.prob_sparse import (
        ProbSparseAttentionImpl as ProbSparseAttentionImpl,
    )
    from foreblocks.nn.attention.variants.sliding_window import (
        SlidingWindowAttentionImpl as SlidingWindowAttentionImpl,
    )
    from foreblocks.nn.attention.variants.softpick import (
        SoftpickAttentionImpl as SoftpickAttentionImpl,
    )
    from foreblocks.nn.attention.variants.spectral import (
        SpectralAttentionImpl as SpectralAttentionImpl,
    )
    from foreblocks.nn.attention.variants.standard import (
        StandardAttentionImpl as StandardAttentionImpl,
    )

_EXPORTS = {
    "AttentionImpl": "foreblocks.nn.attention.variants.base",
    "DilatedSlidingWindowAttentionImpl": "foreblocks.nn.attention.variants.dilated_sliding_window",
    "MoBAAttentionImpl": "foreblocks.nn.attention.variants.moba",
    "NSAAttentionImpl": "foreblocks.nn.attention.variants.nsa",
    "ProbSparseAttentionImpl": "foreblocks.nn.attention.variants.prob_sparse",
    "SlidingWindowAttentionImpl": "foreblocks.nn.attention.variants.sliding_window",
    "SoftpickAttentionImpl": "foreblocks.nn.attention.variants.softpick",
    "SpectralAttentionImpl": "foreblocks.nn.attention.variants.spectral",
    "StandardAttentionImpl": "foreblocks.nn.attention.variants.standard",
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
