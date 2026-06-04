from foreblocks.sequence.mamba_hybrid.attention import SlidingWindowAttention
from foreblocks.sequence.mamba_hybrid.conv import CausalDepthwiseConv1d
from foreblocks.sequence.mamba_hybrid.feedforward import FeedForward
from foreblocks.sequence.mamba_hybrid.hybrid import HybridMamba2Block
from foreblocks.sequence.mamba_hybrid.mamba2 import Mamba2Block
from foreblocks.sequence.mamba_hybrid.mamba3 import Mamba3Block
from foreblocks.sequence.mamba_hybrid.norms import RMSNorm, RMSNormWeightOnly
from foreblocks.sequence.mamba_hybrid.rotary import RotaryEmbedding
from foreblocks.ops.mamba import (
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
    rotary_apply,
    rotary_apply_fallback,
)


_HAS_TRITON = TRITON_AVAILABLE
__all__ = [
    "RotaryEmbedding",
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
