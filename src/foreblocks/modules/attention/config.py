"""Grouped configuration objects for multi-backend attention."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AttentionShapeConfig:
    d_model: int
    n_heads: int
    n_kv_heads: int | None = None
    dropout: float = 0.1
    max_seq_len: int = 4096
    cross_attention: bool = False


@dataclass(frozen=True)
class AttentionCacheConfig:
    use_paged_cache: bool = True
    block_size: int = 128
    max_blocks: int = 2048
    use_mla: bool = True
    kv_latent_dim: int | None = None
    attention_matching: bool = False
    matching_keep_ratio: float = 0.25
    matching_trigger_len: int = 512
    matching_min_keep: int = 64
    matching_query_budget: int = 64
    matching_force_single_step: bool = False


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
    frequency_modes: int = 32
    softpick_chunk_size: int = 128
    global_attention_ratio: float = 0.1
    use_flash_sliding: bool = True
    use_swiglu: bool = True
    nsa_block_size: int | None = None
    nsa_topk_ratio: float | None = None
    moba_block_size: int | None = None
    moba_topk: int = 4
    dilation: int = 2
    dilated_window_size: int | None = None


@dataclass(frozen=True)
class AttentionFeatureConfig:
    qk_norm: bool = False
    qk_norm_type: str = "rms"
    logit_softcap: float | None = None
    learned_temperature: bool = False
    gated_attention: bool = False
    normalized_output: bool = False
    head_importance: bool = False
    gated_attention_mode: str = "per_head"
    gated_attention_bias: bool = True
    temperature_init: float = 1.0
    subquery_norm: bool = False
    subquery_norm_mode: str = "learned"
    multiscale_mask: bool = False
    multiscale_window_ratio: float = 0.2
    multiscale_topk: int = 16
    normalized_output_type: str = "rms"
    head_importance_sparsity: float = 0.1
    verbose_init: bool = False


@dataclass(frozen=True)
class AttentionConfig:
    shape: AttentionShapeConfig
    architecture: str | None = None
    cache: AttentionCacheConfig = field(default_factory=AttentionCacheConfig)
    position: AttentionPositionConfig = field(default_factory=AttentionPositionConfig)
    variant: AttentionVariantConfig = field(default_factory=AttentionVariantConfig)
    features: AttentionFeatureConfig = field(default_factory=AttentionFeatureConfig)

    @property
    def backend(self) -> str:
        """Compatibility view of the configured kernel backend."""
        return self.variant.backend

    @property
    def position_encoding(self) -> str:
        return self.position.encoding

    def to_legacy_kwargs(self) -> dict[str, object]:
        return {
            "d_model": self.shape.d_model,
            "n_heads": self.shape.n_heads,
            "n_kv_heads": self.shape.n_kv_heads,
            "dropout": self.shape.dropout,
            "max_seq_len": self.shape.max_seq_len,
            "cross_attention": self.shape.cross_attention,
            "attention_type": self.variant.name,
            "attn_implementation": self.variant.backend,
            "window_size": self.variant.window_size,
            "chunk_size": self.variant.chunk_size,
            "prob_sparse_factor": self.variant.probability_factor,
            "freq_modes": self.variant.frequency_modes,
            "softpick_chunk_size": self.variant.softpick_chunk_size,
            "global_attention_ratio": self.variant.global_attention_ratio,
            "use_flash_sliding": self.variant.use_flash_sliding,
            "use_swiglu": self.variant.use_swiglu,
            "nsa_block_size": self.variant.nsa_block_size,
            "nsa_topk_ratio": self.variant.nsa_topk_ratio,
            "moba_block_size": self.variant.moba_block_size,
            "moba_topk": self.variant.moba_topk,
            "attention_dilation": self.variant.dilation,
            "dilated_window_size": self.variant.dilated_window_size,
            "use_paged_cache": self.cache.use_paged_cache,
            "cache_block_size": self.cache.block_size,
            "max_cache_blocks": self.cache.max_blocks,
            "use_mla": self.cache.use_mla,
            "kv_latent_dim": self.cache.kv_latent_dim,
            "use_attention_matching_compaction": self.cache.attention_matching,
            "attention_matching_keep_ratio": self.cache.matching_keep_ratio,
            "attention_matching_trigger_len": self.cache.matching_trigger_len,
            "attention_matching_min_keep": self.cache.matching_min_keep,
            "attention_matching_query_budget": self.cache.matching_query_budget,
            "attention_matching_force_single_step": (
                self.cache.matching_force_single_step
            ),
            "pos_encoding_type": self.position.encoding,
            "rope_base": self.position.rope_base,
            "rope_scaling_type": self.position.rope_scaling_type,
            "rope_scaling_factor": self.position.rope_scaling_factor,
            "qk_norm": self.features.qk_norm,
            "qk_norm_type": self.features.qk_norm_type,
            "logit_softcap": self.features.logit_softcap,
            "use_learned_temp": self.features.learned_temperature,
            "use_gated_attention": self.features.gated_attention,
            "gated_attn_mode": self.features.gated_attention_mode,
            "gated_attn_bias": self.features.gated_attention_bias,
            "temp_init": self.features.temperature_init,
            "use_subquery_norm": self.features.subquery_norm,
            "subquery_norm_mode": self.features.subquery_norm_mode,
            "use_multiscale_mask": self.features.multiscale_mask,
            "multiscale_window_ratio": self.features.multiscale_window_ratio,
            "multiscale_topk": self.features.multiscale_topk,
            "use_normalized_attn_out": self.features.normalized_output,
            "norm_attn_type": self.features.normalized_output_type,
            "use_head_importance": self.features.head_importance,
            "head_importance_sparsity": self.features.head_importance_sparsity,
            "verbose_init": self.features.verbose_init,
        }


# Compatibility name retained for callers that adopted the initial grouped API.
MultiAttentionConfig = AttentionConfig


__all__ = [
    "AttentionCacheConfig",
    "AttentionConfig",
    "AttentionFeatureConfig",
    "AttentionPositionConfig",
    "AttentionShapeConfig",
    "AttentionVariantConfig",
    "MultiAttentionConfig",
]
