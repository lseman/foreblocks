"""Grouped configuration objects for multi-backend attention."""

from __future__ import annotations

from dataclasses import dataclass, field

from foreblocks.modules.attention.enums import (
    AttentionOutputNorm,
    GatedAttentionMode,
    PositionEncoding,
    QKNorm,
    RopeScaling,
    SubqueryNorm,
)


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
    encoding: PositionEncoding = PositionEncoding.ROPE
    rope_base: float = 10000.0
    rope_scaling_type: RopeScaling = RopeScaling.NONE
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
    qk_norm_type: QKNorm = QKNorm.RMS
    logit_softcap: float | None = None
    learned_temperature: bool = False
    gated_attention: bool = False
    normalized_output: bool = False
    head_importance: bool = False
    gated_attention_mode: GatedAttentionMode = GatedAttentionMode.PER_HEAD
    gated_attention_bias: bool = True
    temperature_init: float = 1.0
    subquery_norm: bool = False
    subquery_norm_mode: SubqueryNorm = SubqueryNorm.LEARNED
    multiscale_mask: bool = False
    multiscale_window_ratio: float = 0.2
    multiscale_topk: int = 16
    normalized_output_type: AttentionOutputNorm = AttentionOutputNorm.RMS
    head_importance_sparsity: float = 0.1
    verbose_init: bool = False


@dataclass(frozen=True)
class AttentionConfig:
    shape: AttentionShapeConfig
    architecture: str = "standard"
    cache: AttentionCacheConfig = field(default_factory=AttentionCacheConfig)
    position: AttentionPositionConfig = field(default_factory=AttentionPositionConfig)
    variant: AttentionVariantConfig = field(default_factory=AttentionVariantConfig)
    features: AttentionFeatureConfig = field(default_factory=AttentionFeatureConfig)

    @classmethod
    def from_legacy_kwargs(cls, **values: object) -> AttentionConfig:
        """Load the former flat ``MultiAttention`` constructor representation."""
        from foreblocks.modules.attention.compat import (
            attention_config_from_legacy_kwargs,
        )

        return attention_config_from_legacy_kwargs(**values)

__all__ = [
    "AttentionCacheConfig",
    "AttentionConfig",
    "AttentionFeatureConfig",
    "AttentionPositionConfig",
    "AttentionShapeConfig",
    "AttentionVariantConfig",
]
