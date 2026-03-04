"""base_blocks — re-export shim.

All classes live in dedicated sub-modules:

    bb_primitives   → RMSNorm, SwiGLUFFN
    bb_positional   → RotaryPositionalEncoding, PositionalEncoding
    bb_attention    → LinearSelfAttention, AttentionBridge, LearnedPoolingBridge
    bb_transformers → LightweightTransformerEncoder, PatchTSTEncoder,
                      LightweightTransformerDecoder
    bb_sequence     → ArchitectureNormalizer, SearchableDecomposition,
                      SequenceStateAdapter, BaseMixedSequenceBlock,
                      BaseFixedSequenceBlock
    bb_mixed        → MixedEncoder, MixedDecoder, ArchitectureConverter,
                      FixedEncoder, FixedDecoder
    bb_mamba        → MambaBranch
"""
from .bb_attention import AttentionBridge, LearnedPoolingBridge, LinearSelfAttention
from .bb_mamba import MambaBranch
from .bb_mixed import (
    ArchitectureConverter,
    FixedDecoder,
    FixedEncoder,
    MixedDecoder,
    MixedEncoder,
)
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
    PatchTSTEncoder,
)

__all__ = [
    "RMSNorm",
    "SwiGLUFFN",
    "RotaryPositionalEncoding",
    "PositionalEncoding",
    "LinearSelfAttention",
    "AttentionBridge",
    "LearnedPoolingBridge",
    "LightweightTransformerEncoder",
    "PatchTSTEncoder",
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
    "MambaBranch",
]
