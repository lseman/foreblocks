"""foreblocks.ops.kernels.

Package initializer that exposes the public symbols for this namespace.
It belongs to the low-level optimized operations and kernel wrappers area of Foreblocks.

"""

# Legacy shim modules (kept for backward compatibility)
from foreblocks.ops.kernels import kernels  # noqa: F401
from foreblocks.ops.kernels.grouped_gemm import *  # noqa: F403
from foreblocks.ops.kernels.layer_norm import *  # noqa: F403
from foreblocks.ops.kernels.rms_norm import *  # noqa: F403
from foreblocks.ops.kernels.swiglu import *  # noqa: F403
from foreblocks.ops.kernels.softmax import *  # noqa: F403
from foreblocks.ops.kernels.gelu import *  # noqa: F403
