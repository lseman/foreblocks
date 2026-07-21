"""foreblocks.modules.attention.variants.

Package initializer that exposes the public symbols for this namespace.
It belongs to the attention pattern variants area of Foreblocks.

"""

from foreblocks.modules.attention.variants.base import AttentionImpl
from foreblocks.modules.attention.variants.dilated_sliding_window import (
    DilatedSlidingWindowAttentionImpl,
)
from foreblocks.modules.attention.variants.moba import MoBAAttentionImpl
from foreblocks.modules.attention.variants.nsa import NSAImpl
from foreblocks.modules.attention.variants.prob_sparse import ProbSparseAttentionImpl
from foreblocks.modules.attention.variants.sliding_window import (
    SlidingWindowAttentionImpl,
)
from foreblocks.modules.attention.variants.softpick import SoftpickAttentionImpl
from foreblocks.modules.attention.variants.spectral import SpectralAttentionImpl
from foreblocks.modules.attention.variants.standard import StandardAttentionImpl

__all__ = [
    "AttentionImpl",
    "DilatedSlidingWindowAttentionImpl",
    "MoBAAttentionImpl",
    "NSAImpl",
    "ProbSparseAttentionImpl",
    "SlidingWindowAttentionImpl",
    "SoftpickAttentionImpl",
    "SpectralAttentionImpl",
    "StandardAttentionImpl",
]
