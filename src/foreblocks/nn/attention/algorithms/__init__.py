"""Lazy public exports for foreblocks.nn.attention.algorithms."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.attention.algorithms.linear import (
        DeltaNetBackend as DeltaNetBackend,
        FeatureMapRegistry as FeatureMapRegistry,
        GatedDeltaNet2Backend as GatedDeltaNet2Backend,
        GatedDeltaNetBackend as GatedDeltaNetBackend,
        GLABackend as GLABackend,
        KimiAttentionBackend as KimiAttentionBackend,
        ModernLinearAttention as ModernLinearAttention,
        RDABackend as RDABackend,
        RoPEMixin as RoPEMixin,
    )
    from foreblocks.nn.attention.algorithms.spectral.autocorrelation import (
        AutoCorrelation as AutoCorrelation,
        AutoCorrelationLayer as AutoCorrelationLayer,
    )
    from foreblocks.nn.attention.algorithms.spectral.frequency import (
        FourierBlock as FourierBlock,
        FourierModeSelector as FourierModeSelector,
        FrequencyAttention as FrequencyAttention,
    )
    from foreblocks.nn.attention.algorithms.spectral.wavelet import (
        DWTAttention as DWTAttention,
    )

_EXPORTS = {
    "AutoCorrelation": "foreblocks.nn.attention.algorithms.spectral.autocorrelation",
    "AutoCorrelationLayer": "foreblocks.nn.attention.algorithms.spectral.autocorrelation",
    "DWTAttention": "foreblocks.nn.attention.algorithms.spectral.wavelet",
    "DeltaNetBackend": "foreblocks.nn.attention.algorithms.linear",
    "FeatureMapRegistry": "foreblocks.nn.attention.algorithms.linear",
    "FourierBlock": "foreblocks.nn.attention.algorithms.spectral.frequency",
    "FourierModeSelector": "foreblocks.nn.attention.algorithms.spectral.frequency",
    "FrequencyAttention": "foreblocks.nn.attention.algorithms.spectral.frequency",
    "GLABackend": "foreblocks.nn.attention.algorithms.linear",
    "GatedDeltaNetBackend": "foreblocks.nn.attention.algorithms.linear",
    "GatedDeltaNet2Backend": "foreblocks.nn.attention.algorithms.linear",
    "KimiAttentionBackend": "foreblocks.nn.attention.algorithms.linear",
    "ModernLinearAttention": "foreblocks.nn.attention.algorithms.linear",
    "RDABackend": "foreblocks.nn.attention.algorithms.linear",
    "RoPEMixin": "foreblocks.nn.attention.algorithms.linear",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
