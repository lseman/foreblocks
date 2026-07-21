"""foreblocks.modules.moe.

Package initializer that exposes the public symbols for this namespace.
It belongs to the mixture-of-experts layers and utilities area of Foreblocks.

"""

from foreblocks.modules.moe import experts, ff
from foreblocks.modules.moe.ff import FeedForwardBlock

__all__ = [
    "FeedForwardBlock",
    "experts",
    "ff",
]
