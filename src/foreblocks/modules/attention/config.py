"""Grouped configuration objects for multi-backend attention."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttentionShapeConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int | None = None
    dropout: float = 0.1


@dataclass(frozen=True)
class AttentionCacheConfig:
    use_paged_cache: bool = True
    block_size: int = 128
    max_blocks: int = 2048
    use_mla: bool = True
    kv_latent_dim: int | None = None


@dataclass(frozen=True)
class AttentionPositionConfig:
    encoding: str = "rope"
    rope_base: float = 10000.0
    rope_scaling_type: str = "none"
    rope_scaling_factor: float = 1.0


@dataclass(frozen=True)
class AttentionVariantConfig:
    name: str = "standard"
    backend: str = "auto"
    window_size: int = 64
    chunk_size: int = 1024
    probability_factor: float = 0.4


@dataclass(frozen=True)
class AttentionFeatureConfig:
    qk_norm: bool = False
    qk_norm_type: str = "rms"
    logit_softcap: float | None = None
    learned_temperature: bool = False
    gated_attention: bool = False
    normalized_output: bool = False
    head_importance: bool = False


@dataclass(frozen=True)
class MultiAttentionConfig:
    shape: AttentionShapeConfig
    cache: AttentionCacheConfig = field(default_factory=AttentionCacheConfig)
    position: AttentionPositionConfig = field(default_factory=AttentionPositionConfig)
    variant: AttentionVariantConfig = field(default_factory=AttentionVariantConfig)
    features: AttentionFeatureConfig = field(default_factory=AttentionFeatureConfig)

    def to_legacy_kwargs(self) -> dict[str, object]:
        return {
            "d_model": self.shape.d_model,
            "n_heads": self.shape.n_heads,
            "n_kv_heads": self.shape.n_kv_heads,
            "dropout": self.shape.dropout,
            "attention_type": self.variant.name,
            "attn_implementation": self.variant.backend,
            "window_size": self.variant.window_size,
            "chunk_size": self.variant.chunk_size,
            "prob_sparse_factor": self.variant.probability_factor,
            "use_paged_cache": self.cache.use_paged_cache,
            "cache_block_size": self.cache.block_size,
            "max_cache_blocks": self.cache.max_blocks,
            "use_mla": self.cache.use_mla,
            "kv_latent_dim": self.cache.kv_latent_dim,
            "pos_encoding_type": self.position.encoding,
            "rope_base": self.position.rope_base,
            "rope_scaling_type": self.position.rope_scaling_type,
            "rope_scaling_factor": self.position.rope_scaling_factor,
            "qk_norm": self.features.qk_norm,
            "qk_norm_type": self.features.qk_norm_type,
            "logit_softcap": self.features.logit_softcap,
            "use_learned_temp": self.features.learned_temperature,
            "use_gated_attention": self.features.gated_attention,
            "use_normalized_attn_out": self.features.normalized_output,
            "use_head_importance": self.features.head_importance,
        }


__all__ = [
    "AttentionCacheConfig",
    "AttentionFeatureConfig",
    "AttentionPositionConfig",
    "AttentionShapeConfig",
    "AttentionVariantConfig",
    "MultiAttentionConfig",
]
