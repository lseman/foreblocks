"""foreblocks.sequence.

Stateful sequence models: Mamba, S4, and linear-RNN variants.

Sequence exports sequence-modeling backbones (Mamba 2/3, hybrid Mamba-Attn blocks,
FLA-based models) optimized for long-context forecasting. These replace or augment
Transformer backbones for efficiency and expressiveness. BlockStack enables
heterogeneous architectures (Mamba layers, attention layers, MoE experts mixed
in a single stack).

Subpackages:
  - mamba:         original Mamba (MoE-ready, positional embeddings, diagnostics)
  - raven:         Raven/FLA-inspired blocks (linear attention via gates)
  - block_stack:   heterogeneous block composition (Nemotron-style)

Core API:
- BlockStack: configurable heterogeneous block stack
- BLOCK_TYPES: registry of available block implementations
"""

from foreblocks.sequence.block_stack import BLOCK_TYPES, BlockStack


__all__ = ["BlockStack", "BLOCK_TYPES"]
