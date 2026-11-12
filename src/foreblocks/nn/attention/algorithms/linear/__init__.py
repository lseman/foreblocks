"""Lazy public exports for foreblocks.nn.attention.algorithms.linear."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.attention.algorithms.linear.base import (
        FeatureMapRegistry as FeatureMapRegistry,
        RoPEMixin as RoPEMixin,
    )
    from foreblocks.nn.attention.algorithms.linear.deltanet import (
        DeltaNetBackend as DeltaNetBackend,
    )
    from foreblocks.nn.attention.algorithms.linear.gated_delta import (
        GatedDeltaNetBackend as GatedDeltaNetBackend,
    )
    from foreblocks.nn.attention.algorithms.linear.gated_deltanet2 import (
        GatedDeltaNet2Backend as GatedDeltaNet2Backend,
    )
    from foreblocks.nn.attention.algorithms.linear.gla import GLABackend as GLABackend
    from foreblocks.nn.attention.algorithms.linear.kimi import (
        KimiAttentionBackend as KimiAttentionBackend,
    )
    from foreblocks.nn.attention.algorithms.linear.rda import RDABackend as RDABackend
    from foreblocks.nn.attention.algorithms.linear.wrapper import (
        ModernLinearAttention as ModernLinearAttention,
    )

_EXPORTS = {
    "DeltaNetBackend": "foreblocks.nn.attention.algorithms.linear.deltanet",
    "FeatureMapRegistry": "foreblocks.nn.attention.algorithms.linear.base",
    "GLABackend": "foreblocks.nn.attention.algorithms.linear.gla",
    "GatedDeltaNetBackend": "foreblocks.nn.attention.algorithms.linear.gated_delta",
    "GatedDeltaNet2Backend": "foreblocks.nn.attention.algorithms.linear.gated_deltanet2",
    "KimiAttentionBackend": "foreblocks.nn.attention.algorithms.linear.kimi",
    "ModernLinearAttention": "foreblocks.nn.attention.algorithms.linear.wrapper",
    "RDABackend": "foreblocks.nn.attention.algorithms.linear.rda",
    "RoPEMixin": "foreblocks.nn.attention.algorithms.linear.base",
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
