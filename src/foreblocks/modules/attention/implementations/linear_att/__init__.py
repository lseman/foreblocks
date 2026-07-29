"""foreblocks.modules.attention.implementations.linear_att.

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

from foreblocks.modules.attention.implementations.linear_att.base import (
    FeatureMapRegistry,
    RoPEMixin,
)
from foreblocks.modules.attention.implementations.linear_att.deltanet import (
    DeltaNetBackend,
)
from foreblocks.modules.attention.implementations.linear_att.gated_delta import (
    GatedDeltaNetBackend,
)
from foreblocks.modules.attention.implementations.linear_att.gated_deltanet2 import (
    GatedDeltaNet2Backend,
)
from foreblocks.modules.attention.implementations.linear_att.gla import GLABackend
from foreblocks.modules.attention.implementations.linear_att.kimi import (
    KimiAttentionBackend,
)
from foreblocks.modules.attention.implementations.linear_att.rda import RDABackend
from foreblocks.modules.attention.implementations.linear_att.wrapper import (
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
