# triton_helpers.py - backward-compatibility shim
# All content has been split into per-kernel modules.
from .swiglu import HAS_TRITON, TritonSwiGLUGate, swiglu_gate  # noqa: F401
from .linear_attention import chunked_causal_linear_attn  # noqa: F401
