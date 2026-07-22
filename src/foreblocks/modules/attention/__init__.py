"""Stable public facade for Foreblocks attention.

Concrete attention algorithms and cache implementations intentionally live in
the ``implementations`` and ``cache`` subpackages. Keeping this facade small avoids
loading optional implementations merely to import the core attention API.
"""

from foreblocks.modules.attention.cache.base import KVCacheProtocol
from foreblocks.modules.attention.config import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionFeatureConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)
from foreblocks.modules.attention.enums import (
    AttentionOutputNorm,
    GatedAttentionMode,
    PositionEncoding,
    QKNorm,
    RopeScaling,
    SubqueryNorm,
)
from foreblocks.modules.attention.execution.backends import (
    AttentionBackendRegistry,
    AttentionBackendSpec,
    register_attention_backend,
)
from foreblocks.modules.attention.multi_att import MultiAttention
from foreblocks.modules.attention.variants.base import AttentionContext, AttentionImpl
from foreblocks.modules.attention.variants.registry import (
    AttentionVariantRegistry,
    register_attention_variant,
)

__all__ = [
    "AttentionBackendRegistry",
    "AttentionBackendSpec",
    "AttentionCacheConfig",
    "AttentionConfig",
    "AttentionContext",
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
