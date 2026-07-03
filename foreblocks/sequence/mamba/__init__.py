"""foreblocks.sequence.mamba.

Mamba-style SSM blocks and state-space operator kernels with Triton acceleration.

Provides Mamba2 and Mamba3 block implementations, chunked SSD scan kernels,
causal depthwise convolutions, RMS normalization, and Triton availability flags.
Designed for high-throughput sequence modeling with optional fused Triton paths.

Core API:
- Mamba2Block: Mamba2-style SSM block with diagonal A and chunked scan
- Mamba3Block: Mamba3-style SSM block with blockwise rotary on B/C
- FeedForward: SwiGLU feed-forward block
- CausalDepthwiseConv1d: causal depthwise 1-D convolution
- RMSNorm, RMSNormWeightOnly: RMS normalization helpers
- TRITON_AVAILABLE, CHUNKED_SSD_TRITON_AVAILABLE, ...: Triton kernel availability flags

"""

from foreblocks.ops.mamba import (  # rotary_apply, rotary_apply_fallback stubs below
    CAUSAL_CONV1D_TRITON_AVAILABLE,
    CHUNKED_SSD_TRITON_AVAILABLE,
    RMS_NORM_TRITON_AVAILABLE,
    ROTARY_TRITON_AVAILABLE,
    TRITON_AVAILABLE,
    causal_depthwise_conv1d,
    causal_depthwise_conv1d_bwd_triton,
    causal_depthwise_conv1d_reference,
    causal_depthwise_conv1d_triton,
    chunked_ssd_backward_reference,
    chunked_ssd_forward,
    chunked_ssd_forward_reference,
    chunked_ssd_forward_triton,
    dt_prep,
    dt_prep_bwd_triton,
    dt_prep_fallback,
    dt_prep_triton,
    fused_out,
    fused_out_bwd_triton,
    fused_out_fallback,
    fused_out_triton,
    mamba2_split_conv1d_scan_combined,
    rms_norm,
    rms_norm_fallback,
)
from foreblocks.sequence.mamba.conv import CausalDepthwiseConv1d
from foreblocks.sequence.mamba.feedforward import FeedForward
from foreblocks.sequence.mamba.mamba2 import Mamba2Block
from foreblocks.sequence.mamba.mamba3 import Mamba3Block
from foreblocks.sequence.mamba.norms import RMSNorm, RMSNormWeightOnly


def rotary_apply(*args, **kwargs):
    """Stub — rotary Triton kernel not implemented."""
    raise NotImplementedError("rotary_apply not yet implemented")


def rotary_apply_fallback(*args, **kwargs):
    """Stub — rotary fallback not implemented."""
    raise NotImplementedError("rotary_apply_fallback not yet implemented")


# from foreblocks.sequence.mamba.attention import SlidingWindowAttention
class SlidingWindowAttention:
    """Stub — not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("SlidingWindowAttention not yet implemented")


# from foreblocks.sequence.mamba.hybrid import HybridMamba2Block
class HybridMamba2Block:
    """Stub — not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("HybridMamba2Block not yet implemented")


# from foreblocks.sequence.mamba.rotary import RotaryEmbedding
class RotaryEmbedding:
    """Stub — not yet implemented."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("RotaryEmbedding not yet implemented")


_HAS_TRITON = TRITON_AVAILABLE
__all__ = [
    "RotaryEmbedding",
    "SlidingWindowAttention",
    "TRITON_AVAILABLE",
    "_HAS_TRITON",
    "CAUSAL_CONV1D_TRITON_AVAILABLE",
    "RMS_NORM_TRITON_AVAILABLE",
    "ROTARY_TRITON_AVAILABLE",
    "CausalDepthwiseConv1d",
    "FeedForward",
    "causal_depthwise_conv1d",
    "causal_depthwise_conv1d_bwd_triton",
    "causal_depthwise_conv1d_reference",
    "causal_depthwise_conv1d_triton",
    "chunked_ssd_backward_reference",
    "chunked_ssd_forward",
    "chunked_ssd_forward_reference",
    "chunked_ssd_forward_triton",
    "dt_prep",
    "dt_prep_bwd_triton",
    "dt_prep_fallback",
    "dt_prep_triton",
    "fused_out",
    "fused_out_bwd_triton",
    "fused_out_fallback",
    "fused_out_triton",
    "mamba2_split_conv1d_scan_combined",
    "rms_norm",
    "rms_norm_fallback",
    "rotary_apply",
    "rotary_apply_fallback",
    "CHUNKED_SSD_TRITON_AVAILABLE",
    "HybridMamba2Block",
    "Mamba2Block",
    "Mamba3Block",
    "RMSNorm",
    "RMSNormWeightOnly",
    "SlidingWindowAttention",
]
