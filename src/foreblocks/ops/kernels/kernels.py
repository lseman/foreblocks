"""foreblocks.ops.kernels.kernels.

Backward-compatibility shim: re-exports from per-kernel modules.

All kernel implementations have been split into dedicated modules
(grouped_gemm, swiglu, etc.). This file re-exports the public symbols so that
code importing `foreblocks.ops.kernels.kernels` continues to work unchanged.

Core API (re-exports):
- grouped_mm_varM / _GroupedMMVarMFunction / _split_by_offsets: grouped GEMM with variable M
- swiglu_gate / TritonSwiGLUGate / grouped_mlp_swiglu: SwiGLU gate and grouped MLP
- HAS_TRITON / TRITON_AVAILABLE: Triton availability flags

"""

# kernels.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from foreblocks.ops.kernels.grouped_gemm import *  # noqa: F403
from foreblocks.ops.kernels.grouped_gemm import (  # noqa: F401
    TRITON_AVAILABLE,
    _foreach_mm,
    _GroupedMMVarMFunction,
    _split_by_offsets,
    grouped_mm_varM,
)
from foreblocks.ops.kernels.swiglu import *  # noqa: F403
from foreblocks.ops.kernels.swiglu import (  # noqa: F401
    HAS_TRITON,
    TritonSwiGLUGate,
    _weights_from_swiglu_experts,
    grouped_mlp_swiglu,
    swiglu_gate,
)
