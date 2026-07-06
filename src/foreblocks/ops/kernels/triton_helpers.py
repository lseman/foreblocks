"""foreblocks.ops.kernels.triton_helpers.

Backward-compatibility shim: re-exports SwiGLU Triton kernels.

Re-exports from `foreblocks.ops.kernels.swiglu` so that code importing
`foreblocks.ops.kernels.triton_helpers` continues to work unchanged.

Core API (re-exports):
- swiglu_gate: SwiGLU gate function
- TritonSwiGLUGate: Triton-accelerated SwiGLU gate
- HAS_TRITON: Triton availability flag

"""

# triton_helpers.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from foreblocks.ops.kernels.swiglu import (  # noqa: F401
    HAS_TRITON,
    TritonSwiGLUGate,
    swiglu_gate,
)
