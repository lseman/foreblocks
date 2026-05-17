# triton_backend.py - backward-compatibility shim
# Triton norm kernels have been moved to foreblocks/transformer/kernels/.
# Import from the canonical locations directly:
#   from foreblocks.transformer.kernels.layer_norm import LayerNormTritonFunction
#   from foreblocks.transformer.kernels.rms_norm import RMSNormTritonFunction, ...
from ..kernels.layer_norm import (  # noqa: F401
    TRITON_AVAILABLE,
    LayerNormTritonFunction,
    layernorm_bwd_dwdb_row_kernel,
    layernorm_bwd_dx_kernel,
    layernorm_fwd_kernel,
)
from ..kernels.rms_norm import (  # noqa: F401
    FusedAddRMSNormFunction,
    RMSNormTritonFunction,
    _should_use_triton,
    fused_add_rmsnorm,
    fused_add_rmsnorm_fwd_kernel,
    triton_fused_rmsnorm_scale_bias,
    triton_scale_bias,
)
