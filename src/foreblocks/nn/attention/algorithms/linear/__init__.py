"""foreblocks.nn.attention.algorithms.linear.

Modular linear attention with swappable backends.

Provides a unified interface to six linear attention backends (RDABackend,
GLABackend, DeltaNetBackend, GatedDeltaNetBackend, GatedDeltaNet2Backend,
KimiAttentionBackend), each implementing O(L·d²) sequence modeling with
recurrent state. Use ModernLinearAttention for runtime backend selection, or
import individual backends directly.

Core API:
- ModernLinearAttention: swappable multi-backend linear attention wrapper
- RDABackend, GLABackend, DeltaNetBackend: standard linear attention backends
- GatedDeltaNetBackend, GatedDeltaNet2Backend: gated delta network backends
- KimiAttentionBackend: Kimi Delta Attention (KDA) with per-channel forget gates
- RoPEMixin, FeatureMapRegistry: shared utilities and feature map factory

"""

from __future__ import annotations

from foreblocks.nn.attention.algorithms.linear.base import (
    FeatureMapRegistry,
    RoPEMixin,
)
from foreblocks.nn.attention.algorithms.linear.deltanet import (
    DeltaNetBackend,
)
from foreblocks.nn.attention.algorithms.linear.gated_delta import (
    GatedDeltaNetBackend,
)
from foreblocks.nn.attention.algorithms.linear.gated_deltanet2 import (
    GatedDeltaNet2Backend,
)
from foreblocks.nn.attention.algorithms.linear.gla import GLABackend
from foreblocks.nn.attention.algorithms.linear.kimi import (
    KimiAttentionBackend,
)
from foreblocks.nn.attention.algorithms.linear.rda import RDABackend
from foreblocks.nn.attention.algorithms.linear.wrapper import (
    ModernLinearAttention,
)

__all__ = [
    "DeltaNetBackend",
    "FeatureMapRegistry",
    "GLABackend",
    "GatedDeltaNetBackend",
    "GatedDeltaNet2Backend",
    "KimiAttentionBackend",
    "ModernLinearAttention",
    "RDABackend",
    "RoPEMixin",
]
