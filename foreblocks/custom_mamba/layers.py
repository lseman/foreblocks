"""Compatibility re-exports for hybrid Mamba layers.

The implementation now lives in focused modules plus ``custom_mamba.blocks``.
Importing from ``foreblocks.custom_mamba.layers`` remains supported.
"""

from .blocks import (
    CausalDepthwiseConv1d,
    FeedForward,
    HybridMamba2Block,
    HybridMambaBlock,
    RMSNorm,
    RMSNormWeightOnly,
    RotaryEmbedding,
    SlidingWindowAttention,
    StructuredStateSpaceDualityBranch,
    TinyHybridMamba2LM,
    TinyHybridMambaLM,
)

__all__ = [
    "CausalDepthwiseConv1d",
    "FeedForward",
    "HybridMambaBlock",
    "HybridMamba2Block",
    "RMSNorm",
    "RMSNormWeightOnly",
    "RotaryEmbedding",
    "SlidingWindowAttention",
    "StructuredStateSpaceDualityBranch",
    "TinyHybridMambaLM",
    "TinyHybridMamba2LM",
]
