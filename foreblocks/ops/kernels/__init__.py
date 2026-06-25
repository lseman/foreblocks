# Legacy shim modules (kept for backward compatibility)
from foreblocks.ops.kernels import kernels  # noqa: F401
from foreblocks.ops.kernels import triton_helpers  # noqa: F401
from foreblocks.ops.kernels.grouped_gemm import *  # noqa: F403
from foreblocks.ops.kernels.layer_norm import *  # noqa: F403
from foreblocks.ops.kernels.rms_norm import *  # noqa: F403
from foreblocks.ops.kernels.swiglu import *  # noqa: F403
