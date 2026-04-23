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

from .bb_attention import AttentionBridge
from .bb_attention import LearnedPoolingBridge
from .bb_attention import SelfAttention
from .bb_mixed import ArchitectureConverter
from .bb_mixed import FixedDecoder
from .bb_mixed import FixedEncoder
from .bb_mixed import MixedDecoder
from .bb_mixed import MixedEncoder
from .bb_moe import DARTSFeedForward
from .bb_positional import PositionalEncoding
from .bb_positional import RotaryPositionalEncoding
from .bb_primitives import RMSNorm
from .bb_primitives import SwiGLUFFN
from .bb_sequence import ArchitectureNormalizer
from .bb_sequence import BaseFixedSequenceBlock
from .bb_sequence import BaseMixedSequenceBlock
from .bb_sequence import SearchableDecomposition
from .bb_sequence import SequenceStateAdapter
from .bb_transformers import LightweightTransformerDecoder
from .bb_transformers import LightweightTransformerEncoder


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
