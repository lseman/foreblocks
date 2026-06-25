"""Alternative sequence backbones.

Subpackages:
  - mamba:         original Mamba backbone (MoE, positional, eval tools)
  - mamba_hybrid:  hybrid Mamba/Mamba2 blocks (formerly custom_mamba)
  - raven:         Raven/FLA-inspired sequence blocks (formerly custom_raven)

Modules:
  - block_stack:   Nemotron-style heterogeneous block stack (Mamba/Attn/MoE)
"""

from foreblocks.sequence.block_stack import BLOCK_TYPES, BlockStack


__all__ = ["BlockStack", "BLOCK_TYPES"]
