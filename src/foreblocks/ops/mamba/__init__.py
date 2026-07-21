"""foreblocks.ops.mamba.

Mamba2 and state-space operator kernels for Foreblocks.

Provides causal conv1d, fused dt projection, chunked SSM (SSD) forward/backward,
RMSNorm, and Triton helpers. All modules gracefully degrade when Triton is
unavailable.

Core API (entry points):
- mamba2_split_conv1d_scan_combined: full Mamba2 block path
- chunked_ssd_forward: chunked SSM forward with triton/pytorch toggle
- causal_depthwise_conv1d: causal depthwise conv1d
- fused_dt: fused dt_proj + softplus + clamp
- fused_out: RMSNormGated (RMSNorm + silu gate)
- dt_prep: softplus + clamp time-step preparation

"""

from foreblocks.ops.mamba.causal_conv1d import (
    CAUSAL_CONV1D_TRITON_AVAILABLE,
    causal_depthwise_conv1d,
    causal_depthwise_conv1d_bwd_triton,
    causal_depthwise_conv1d_reference,
    causal_depthwise_conv1d_triton,
)
from foreblocks.ops.mamba.fused_dt import (
    FUSED_DT_TRITON_AVAILABLE,
    fused_dt,
    fused_dt_bwd_fallback,
    fused_dt_fallback,
    fused_dt_triton,
)
from foreblocks.ops.mamba.mamba2_combined import mamba2_split_conv1d_scan_combined
from foreblocks.ops.kernels.rms_norm import (
    TRITON_AVAILABLE as RMS_NORM_TRITON_AVAILABLE,
    rms_norm,
    rms_norm_fallback,
)

# rotary_apply, rotary_apply_fallback
from foreblocks.ops.mamba.ssd import (
    CHUNKED_SSD_TRITON_AVAILABLE,
    chunked_ssd_backward_reference,
    chunked_ssd_forward,
    chunked_ssd_forward_reference,
    chunked_ssd_forward_triton,
    chunked_ssd_forward_triton_parallel,
    chunked_ssd_forward_triton_tiled,
    segment_sum,
)
from foreblocks.ops.mamba.triton_ops import (
    TRITON_AVAILABLE,
    dt_prep,
    dt_prep_bwd_triton,
    dt_prep_fallback,
    dt_prep_triton,
    fused_out,
    fused_out_bwd_triton,
    fused_out_fallback,
    fused_out_triton,
)

ROTARY_TRITON_AVAILABLE = False  # rotary Triton kernels not yet implemented

__all__ = [
    "CAUSAL_CONV1D_TRITON_AVAILABLE",
    "FUSED_DT_TRITON_AVAILABLE",
    "RMS_NORM_TRITON_AVAILABLE",
    "ROTARY_TRITON_AVAILABLE",
    "causal_depthwise_conv1d",
    "causal_depthwise_conv1d_bwd_triton",
    "causal_depthwise_conv1d_reference",
    "causal_depthwise_conv1d_triton",
    "fused_dt",
    "fused_dt_bwd_fallback",
    "fused_dt_fallback",
    "fused_dt_triton",
    "mamba2_split_conv1d_scan_combined",
    "CHUNKED_SSD_TRITON_AVAILABLE",
    "chunked_ssd_backward_reference",
    "chunked_ssd_forward",
    "chunked_ssd_forward_reference",
    "chunked_ssd_forward_triton",
    "chunked_ssd_forward_triton_parallel",
    "chunked_ssd_forward_triton_tiled",
    "segment_sum",
    "TRITON_AVAILABLE",
    "dt_prep",
    "dt_prep_bwd_triton",
    "dt_prep_fallback",
    "dt_prep_triton",
    "fused_out",
    "fused_out_bwd_triton",
    "fused_out_fallback",
    "fused_out_triton",
    "rms_norm",
    "rms_norm_fallback",
    # "rotary_apply",
    # "rotary_apply_fallback",
]
