"""base_blocks — re-export shim.

All classes live in dedicated sub-modules:

    bb_primitives   → RMSNorm, SwiGLUFFN
    bb_positional   → RotaryPositionalEncoding, PositionalEncoding
    bb_attention    → SelfAttention, AttentionBridge, LearnedPoolingBridge
    bb_moe          → DARTSFeedForward
    bb_transformers → LightweightTransformerEncoder,
                      LightweightTransformerDecoder
    bb_sequence     → ArchitectureNormalizer, SearchableDecomposition,
                      SequenceStateAdapter, BaseMixedSequenceBlock,
                      BaseFixedSequenceBlock
    bb_mixed        → MixedEncoder, MixedDecoder, ArchitectureConverter,
                      FixedEncoder, FixedDecoder
"""

from .bb_attention import AttentionBridge, LearnedPoolingBridge, SelfAttention
from .bb_mixed import (
    ArchitectureConverter,
    FixedDecoder,
    FixedEncoder,
    MixedDecoder,
    MixedEncoder,
)
from .bb_moe import DARTSFeedForward
from .bb_positional import PositionalEncoding, RotaryPositionalEncoding
from .bb_primitives import RMSNorm, SwiGLUFFN
from .bb_sequence import (
    ArchitectureNormalizer,
    BaseFixedSequenceBlock,
    BaseMixedSequenceBlock,
    SearchableDecomposition,
    SequenceStateAdapter,
)
from .bb_transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
)


__all__ = [
    "RMSNorm",
    "SwiGLUFFN",
    "RotaryPositionalEncoding",
    "PositionalEncoding",
    "SelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
    "DARTSFeedForward",
    "LightweightTransformerEncoder",
    "LightweightTransformerDecoder",
    "ArchitectureNormalizer",
    "SearchableDecomposition",
    "SequenceStateAdapter",
    "BaseMixedSequenceBlock",
    "BaseFixedSequenceBlock",
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]
