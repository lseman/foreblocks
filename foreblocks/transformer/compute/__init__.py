# Legacy shim modules (kept for backward compatibility)
from . import (
    kernels,  # noqa: F401
    triton_helpers,  # noqa: F401
)
from .grouped_gemm import *  # noqa: F403
from .layer_norm import *  # noqa: F403
from .linear_attention import *  # noqa: F403
from .rms_norm import *  # noqa: F403
from .swiglu import *  # noqa: F403
