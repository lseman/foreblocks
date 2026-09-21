"""Input projection, position encoding, and mask preparation."""

from foreblocks.nn.attention.preparation.masking import (
    AttentionMaskProcessor,
    build_attention_mask,
    normalize_blocked_mask,
    to_additive_mask,
)
from foreblocks.nn.attention.preparation.pipeline import QKVPipeline
from foreblocks.nn.attention.preparation.position import PositionEncodingApplier
from foreblocks.nn.attention.preparation.projections import QKVProjector

__all__ = [
    "AttentionMaskProcessor",
    "PositionEncodingApplier",
    "QKVPipeline",
    "QKVProjector",
    "build_attention_mask",
    "normalize_blocked_mask",
    "to_additive_mask",
]
