"""foreblocks.attention.variants.

Package initializer that exposes the public symbols for this namespace.
It belongs to the attention pattern variants area of Foreblocks.

"""

from foreblocks.attention.variants.base import AttentionImpl
from foreblocks.attention.variants.dilated_sliding_window import (
    DilatedSlidingWindowAttentionImpl,
)
from foreblocks.attention.variants.moba import MoBAAttentionImpl
from foreblocks.attention.variants.nsa import NSAAttentionImpl
from foreblocks.attention.variants.prob_sparse import ProbSparseAttentionImpl
from foreblocks.attention.variants.sliding_window import (
    SlidingWindowAttentionImpl,
)
from foreblocks.attention.variants.softpick import SoftpickAttentionImpl
from foreblocks.attention.variants.spectral import SpectralAttentionImpl
from foreblocks.attention.variants.standard import StandardAttentionImpl

__all__ = [
    "AttentionImpl",
    "DilatedSlidingWindowAttentionImpl",
    "MoBAAttentionImpl",
    "NSAAttentionImpl",
    "ProbSparseAttentionImpl",
    "SlidingWindowAttentionImpl",
    "SoftpickAttentionImpl",
    "SpectralAttentionImpl",
    "StandardAttentionImpl",
]
