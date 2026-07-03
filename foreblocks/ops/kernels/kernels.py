"""foreblocks.ops.kernels.kernels.

This module implements the kernels pieces for its package.
It belongs to the low-level optimized operations and kernel wrappers area of Foreblocks.
"""

# kernels.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from foreblocks.ops.kernels.grouped_gemm import *  # noqa: F401, F403
from foreblocks.ops.kernels.grouped_gemm import (  # noqa: F401
    TRITON_AVAILABLE,
    _foreach_mm,
    _GroupedMMVarMFunction,
    _split_by_offsets,
    grouped_mm_varM,
)
from foreblocks.ops.kernels.swiglu import *  # noqa: F401, F403
from foreblocks.ops.kernels.swiglu import (  # noqa: F401
    HAS_TRITON,
    TritonSwiGLUGate,
    _weights_from_swiglu_experts,
    grouped_mlp_swiglu,
    swiglu_gate,
)
