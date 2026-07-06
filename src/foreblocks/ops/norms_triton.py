"""foreblocks.ops.norms_triton.

Backward-compatibility shim: re-exports Triton norm kernels.

Triton norm kernels have been moved to `foreblocks.ops.kernels.layer_norm`
and `foreblocks.ops.kernels.rms_norm`. This file re-exports the public
symbols so that legacy imports continue to work unchanged.

Core API (re-exports):
- LayerNormTritonFunction: Triton-accelerated layer normalization
- RMSNormTritonFunction: Triton-accelerated RMS normalization
- FusedAddRMSNormFunction: fused add + RMSNorm
- fused_add_rmsnorm: fused add + RMSNorm helper

"""

# triton_backend.py - backward-compatibility shim
# Triton norm kernels have been moved to foreblocks/transformer/kernels/.
# Import from the canonical locations directly:
#   from foreblocks.ops.kernels.layer_norm import LayerNormTritonFunction
#   from foreblocks.ops.kernels.rms_norm import RMSNormTritonFunction, ...
from foreblocks.ops.kernels.layer_norm import (  # noqa: F401
    TRITON_AVAILABLE,
    LayerNormTritonFunction,
    layernorm_fwd_kernel,
)
from foreblocks.ops.kernels.rms_norm import (  # noqa: F401
    FusedAddRMSNormFunction,
    RMSNormTritonFunction,
    _should_use_triton,
    fused_add_rmsnorm,
    fused_add_rmsnorm_fwd_kernel,
    triton_fused_rmsnorm_scale_bias,
    triton_scale_bias,
)
