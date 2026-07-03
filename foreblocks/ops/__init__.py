"""foreblocks.ops.

Specialized compute kernels for inference performance (Triton/CUDA).

Ops exposes hand-tuned kernels that replace standard PyTorch ops where
performance is critical: grouped matrix multiply (for MoE), fused attention
(FLA-style tiling), state-space model updates (Mamba), and graph operations.
Used internally by foreblocks layers; generally not called directly.

Subpackages:
  - kernels:   general compute (grouped_gemm, swiglu, rms/layer norm)
  - attention: attention kernels (fla, fused_rope, paged decode, chunked linear)
  - mamba:     SSD/Mamba ops (selective state update, conv, scans)
  - raven:     Raven/FLA backend kernels

"""

from foreblocks.ops import attention, kernels, mamba, raven

__all__ = ["attention", "kernels", "mamba", "raven"]
