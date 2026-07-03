"""foreblocks.ops.kernels.triton_helpers.

This module implements the triton helpers pieces for its package.
It belongs to the low-level optimized operations and kernel wrappers area of Foreblocks.
"""

# triton_helpers.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from foreblocks.ops.kernels.swiglu import (  # noqa: F401
    HAS_TRITON,
    TritonSwiGLUGate,
    swiglu_gate,
)
