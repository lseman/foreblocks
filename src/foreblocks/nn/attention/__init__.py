"""Stable public facade for Foreblocks attention.

Concrete attention algorithms and cache implementations intentionally live in
the ``implementations`` and ``cache`` subpackages. Keeping this facade small avoids
loading optional implementations merely to import the core attention API.
"""

from foreblocks.nn.attention.cache.base import KVCacheProtocol
from foreblocks.nn.attention.config import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionFeatureConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)
from foreblocks.nn.attention.enums import (
    AttentionOutputNorm,
    GatedAttentionMode,
    PositionEncoding,
    QKNorm,
    RopeScaling,
    SubqueryNorm,
)
from foreblocks.nn.attention.execution.backends import (
    AttentionBackendRegistry,
    AttentionBackendSpec,
    register_attention_backend,
)
from foreblocks.nn.attention.multihead import MultiAttention
from foreblocks.nn.attention.variants.base import AttentionImpl, AttentionOwner
from foreblocks.nn.attention.variants.registry import (
    AttentionVariantRegistry,
    register_attention_variant,
)

__all__ = [
    "AttentionBackendRegistry",
    "AttentionBackendSpec",
    "AttentionCacheConfig",
    "AttentionConfig",
    "AttentionOwner",
    "AttentionFeatureConfig",
    "AttentionImpl",
    "AttentionPositionConfig",
    "AttentionOutputNorm",
    "AttentionShapeConfig",
    "AttentionVariantConfig",
    "AttentionVariantRegistry",
    "KVCacheProtocol",
    "GatedAttentionMode",
    "MultiAttention",
    "PositionEncoding",
    "QKNorm",
    "RopeScaling",
    "SubqueryNorm",
    "register_attention_backend",
    "register_attention_variant",
]
