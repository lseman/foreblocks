# kernels.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from foreblocks.ops.kernels.grouped_gemm import *  # noqa: F401, F403
from foreblocks.ops.kernels.grouped_gemm import TRITON_AVAILABLE, _GroupedMMVarMFunction, _split_by_offsets, _foreach_mm, grouped_mm_varM  # noqa: F401
from foreblocks.ops.kernels.swiglu import *  # noqa: F401, F403
from foreblocks.ops.kernels.swiglu import HAS_TRITON, TritonSwiGLUGate, swiglu_gate, _weights_from_swiglu_experts, grouped_mlp_swiglu  # noqa: F401
