"""foreblocks.modules.attention.modules.linear_att.

Modular linear attention with swappable backends.

Provides a unified interface to six linear attention backends (RDA, GLA,
DeltaNet, GatedDeltaNet, GatedDeltaNet2, KimiAttention), each implementing
O(L·d²) sequence modeling with recurrent state. Use ModernLinearAttention for
runtime backend selection, or import individual backends directly.

Core API:
- ModernLinearAttention: swappable multi-backend linear attention wrapper
- RDABackend, GLABackend, DeltaNetBackend: standard linear attention backends
- GatedDeltaNet, GatedDeltaNet2: gated delta network backends
- KimiAttention: Kimi Delta Attention (KDA) with per-channel forget gates
- RoPEMixin, FeatureMapRegistry: shared utilities and feature map factory

"""

from __future__ import annotations

from foreblocks.modules.attention.modules.linear_att.base import (
    FeatureMapRegistry,
    RoPEMixin,
)
from foreblocks.modules.attention.modules.linear_att.deltanet import DeltaNetBackend
from foreblocks.modules.attention.modules.linear_att.gated_delta import GatedDeltaNet
from foreblocks.modules.attention.modules.linear_att.gated_deltanet2 import (
    GatedDeltaNet2,
)
from foreblocks.modules.attention.modules.linear_att.gla import GLABackend
from foreblocks.modules.attention.modules.linear_att.kimi import KimiAttention
from foreblocks.modules.attention.modules.linear_att.rda import RDABackend
from foreblocks.modules.attention.modules.linear_att.wrapper import (
    GatedDeltaBackend,
    KimiBackend,
    ModernLinearAttention,
)

__all__ = [
    "FeatureMapRegistry",
    "RoPEMixin",
    "RDABackend",
    "GLABackend",
    "DeltaNetBackend",
    "GatedDeltaBackend",
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "KimiBackend",
    "KimiAttention",
    "ModernLinearAttention",
]
