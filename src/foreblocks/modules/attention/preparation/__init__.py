"""Input projection, position encoding, and mask preparation."""

from foreblocks.modules.attention.preparation.masking import (
    AttentionMaskProcessor,
    build_attention_mask,
    normalize_blocked_mask,
    to_additive_mask,
)
from foreblocks.modules.attention.preparation.pipeline import QKVPipeline
from foreblocks.modules.attention.preparation.position import PositionEncodingApplier
from foreblocks.modules.attention.preparation.projections import QKVProjector

__all__ = [
    "AttentionMaskProcessor",
    "PositionEncodingApplier",
    "QKVPipeline",
    "QKVProjector",
    "build_attention_mask",
    "normalize_blocked_mask",
    "to_additive_mask",
]
