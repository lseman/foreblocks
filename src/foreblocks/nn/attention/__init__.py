"""Lazy public exports for foreblocks.nn.attention."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.attention.cache.base import KVCacheProtocol as KVCacheProtocol
    from foreblocks.nn.attention.config import (
        AttentionCacheConfig as AttentionCacheConfig,
        AttentionConfig as AttentionConfig,
        AttentionFeatureConfig as AttentionFeatureConfig,
        AttentionPositionConfig as AttentionPositionConfig,
        AttentionShapeConfig as AttentionShapeConfig,
        AttentionVariantConfig as AttentionVariantConfig,
    )
    from foreblocks.nn.attention.enums import (
        AttentionOutputNorm as AttentionOutputNorm,
        GatedAttentionMode as GatedAttentionMode,
        PositionEncoding as PositionEncoding,
        QKNorm as QKNorm,
        RopeScaling as RopeScaling,
        SubqueryNorm as SubqueryNorm,
    )
    from foreblocks.nn.attention.execution.backends import (
        AttentionBackendRegistry as AttentionBackendRegistry,
        AttentionBackendSpec as AttentionBackendSpec,
        register_attention_backend as register_attention_backend,
    )
    from foreblocks.nn.attention.multihead import MultiAttention as MultiAttention
    from foreblocks.nn.attention.variants.base import (
        AttentionImpl as AttentionImpl,
        AttentionOwner as AttentionOwner,
    )
    from foreblocks.nn.attention.variants.registry import (
        AttentionVariantRegistry as AttentionVariantRegistry,
        register_attention_variant as register_attention_variant,
    )

_EXPORTS = {
    "AttentionBackendRegistry": "foreblocks.nn.attention.execution.backends",
    "AttentionBackendSpec": "foreblocks.nn.attention.execution.backends",
    "AttentionCacheConfig": "foreblocks.nn.attention.config",
    "AttentionConfig": "foreblocks.nn.attention.config",
    "AttentionOwner": "foreblocks.nn.attention.variants.base",
    "AttentionFeatureConfig": "foreblocks.nn.attention.config",
    "AttentionImpl": "foreblocks.nn.attention.variants.base",
    "AttentionPositionConfig": "foreblocks.nn.attention.config",
    "AttentionOutputNorm": "foreblocks.nn.attention.enums",
    "AttentionShapeConfig": "foreblocks.nn.attention.config",
    "AttentionVariantConfig": "foreblocks.nn.attention.config",
    "AttentionVariantRegistry": "foreblocks.nn.attention.variants.registry",
    "KVCacheProtocol": "foreblocks.nn.attention.cache.base",
    "GatedAttentionMode": "foreblocks.nn.attention.enums",
    "MultiAttention": "foreblocks.nn.attention.multihead",
    "PositionEncoding": "foreblocks.nn.attention.enums",
    "QKNorm": "foreblocks.nn.attention.enums",
    "RopeScaling": "foreblocks.nn.attention.enums",
    "SubqueryNorm": "foreblocks.nn.attention.enums",
    "register_attention_backend": "foreblocks.nn.attention.execution.backends",
    "register_attention_variant": "foreblocks.nn.attention.variants.registry",
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
