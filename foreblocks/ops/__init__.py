"""Low-level compute kernels (Triton/CUDA) for foreblocks.

Subpackages:
  - kernels:   general transformer kernels (grouped_gemm, swiglu, rms/layer norm)
  - attention: attention kernels (fla_*, fused_rope, paged decode, chunked linear)
  - mamba:     Mamba/SSD ops (ssd, causal_conv1d, mamba2_combined)
  - raven:     Raven backend ops
"""
from foreblocks.ops import attention, kernels, mamba, raven

__all__ = ["attention", "kernels", "mamba", "raven"]
