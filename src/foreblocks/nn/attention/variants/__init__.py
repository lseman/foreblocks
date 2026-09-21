"""foreblocks.nn.attention.variants.

Package initializer that exposes the public symbols for this namespace.
It belongs to the attention pattern variants area of Foreblocks.

"""

from foreblocks.nn.attention.variants.base import AttentionImpl
from foreblocks.nn.attention.variants.dilated_sliding_window import (
    DilatedSlidingWindowAttentionImpl,
)
from foreblocks.nn.attention.variants.moba import MoBAAttentionImpl
from foreblocks.nn.attention.variants.nsa import NSAAttentionImpl
from foreblocks.nn.attention.variants.prob_sparse import ProbSparseAttentionImpl
from foreblocks.nn.attention.variants.sliding_window import (
    SlidingWindowAttentionImpl,
)
from foreblocks.nn.attention.variants.softpick import SoftpickAttentionImpl
from foreblocks.nn.attention.variants.spectral import SpectralAttentionImpl
from foreblocks.nn.attention.variants.standard import StandardAttentionImpl

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
