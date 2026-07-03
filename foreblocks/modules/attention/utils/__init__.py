"""foreblocks.modules.attention.utils.

Package initializer that exposes the public symbols for this namespace.
It belongs to the attention modules, variants, caches, and utilities area of Foreblocks.

"""

from foreblocks.modules.attention.utils.compaction import (
    AttentionMatchingCompactor,
    AttentionMatchingConfig,
)
from foreblocks.modules.attention.utils.position import PositionEncodingApplier
from foreblocks.modules.attention.utils.residuals import (
    AttentionResidual,
    BlockAttentionResidual,
    normalize_attention_residual_mode,
)

__all__ = [
    "AttentionMatchingConfig",
    "AttentionMatchingCompactor",
    "PositionEncodingApplier",
    "AttentionResidual",
    "BlockAttentionResidual",
    "normalize_attention_residual_mode",
]
