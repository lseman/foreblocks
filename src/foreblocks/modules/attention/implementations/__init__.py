"""Canonical namespace for concrete attention implementations.

The older ``attention.modules`` path remains available for compatibility, but
new callers should import concrete algorithms from this namespace.
"""

from foreblocks.modules.attention.modules import (
    AutoCorrelation,
    AutoCorrelationLayer,
    DeltaNetBackend,
    DWTAttention,
    FeatureMapRegistry,
    FrequencyAttention,
    GatedDeltaBackend,
    GatedDeltaNet,
    GatedDeltaNet2,
    GLABackend,
    KimiAttention,
    KimiBackend,
    ModernLinearAttention,
    RDABackend,
    RoPEMixin,
)

__all__ = [
    "AutoCorrelation",
    "AutoCorrelationLayer",
    "DWTAttention",
    "DeltaNetBackend",
    "FeatureMapRegistry",
    "FrequencyAttention",
    "GLABackend",
    "GatedDeltaBackend",
    "GatedDeltaNet",
    "GatedDeltaNet2",
    "KimiAttention",
    "KimiBackend",
    "ModernLinearAttention",
    "RDABackend",
    "RoPEMixin",
]
