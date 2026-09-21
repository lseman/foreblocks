"""foreblocks.nn.attention.algorithms.

Package initializer that exposes the public symbols for this namespace.
It belongs to the reusable attention, block, head, MoE, and skip modules area of Foreblocks.

"""

from foreblocks.nn.attention.algorithms.spectral.autocorrelation import (
    AutoCorrelation,
    AutoCorrelationLayer,
)
from foreblocks.nn.attention.algorithms.spectral.wavelet import DWTAttention
from foreblocks.nn.attention.algorithms.spectral.frequency import (
    FourierBlock,
    FourierModeSelector,
    FrequencyAttention,
)
from foreblocks.nn.attention.algorithms.linear import (
    DeltaNetBackend,
    FeatureMapRegistry,
    GatedDeltaNetBackend,
    GatedDeltaNet2Backend,
    GLABackend,
    KimiAttentionBackend,
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
    "FourierBlock",
    "FourierModeSelector",
    "FrequencyAttention",
    "GLABackend",
    "GatedDeltaNetBackend",
    "GatedDeltaNet2Backend",
    "KimiAttentionBackend",
    "ModernLinearAttention",
    "RDABackend",
    "RoPEMixin",
]
