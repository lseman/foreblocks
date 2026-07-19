"""Backward-compatible imports for the former monolithic Triton helpers."""

from foreblocks.ops.kernels.swiglu import TritonSwiGLUGate, swiglu_gate

__all__ = ["TritonSwiGLUGate", "swiglu_gate"]
