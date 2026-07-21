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
    architecture: str = "standard"
    cache: AttentionCacheConfig = field(default_factory=AttentionCacheConfig)
    position: AttentionPositionConfig = field(default_factory=AttentionPositionConfig)
    variant: AttentionVariantConfig = field(default_factory=AttentionVariantConfig)
    features: AttentionFeatureConfig = field(default_factory=AttentionFeatureConfig)

    @classmethod
    def from_legacy_kwargs(cls, **values: object) -> AttentionConfig:
        """Load the former flat ``MultiAttention`` constructor representation."""
        values = dict(values)
        shape = AttentionShapeConfig(
            d_model=int(values.pop("d_model")),
            n_heads=int(values.pop("n_heads")),
            n_kv_heads=values.pop("n_kv_heads", None),
            dropout=float(values.pop("dropout", 0.1)),
            max_seq_len=int(values.pop("max_seq_len", 4096)),
            cross_attention=bool(values.pop("cross_attention", False)),
        )
        cache = AttentionCacheConfig(
            use_paged_cache=bool(values.pop("use_paged_cache", True)),
            block_size=int(values.pop("cache_block_size", 128)),
            max_blocks=int(values.pop("max_cache_blocks", 2048)),
            use_mla=bool(values.pop("use_mla", True)),
            kv_latent_dim=values.pop("kv_latent_dim", None),
            attention_matching=bool(
                values.pop("use_attention_matching_compaction", False)
            ),
            matching_keep_ratio=float(
                values.pop("attention_matching_keep_ratio", 0.25)
            ),
            matching_trigger_len=int(
                values.pop("attention_matching_trigger_len", 512)
            ),
            matching_min_keep=int(values.pop("attention_matching_min_keep", 64)),
            matching_query_budget=int(
                values.pop("attention_matching_query_budget", 64)
            ),
            matching_force_single_step=bool(
                values.pop("attention_matching_force_single_step", False)
            ),
        )
        position = AttentionPositionConfig(
            encoding=str(values.pop("pos_encoding_type", "rope")),
            rope_base=float(values.pop("rope_base", 10000.0)),
            rope_scaling_type=str(values.pop("rope_scaling_type", "none")),
            rope_scaling_factor=float(values.pop("rope_scaling_factor", 1.0)),
        )
        variant = AttentionVariantConfig(
            name=str(values.pop("attention_type", "standard")),
            backend=str(values.pop("attn_implementation", "auto")),
            window_size=int(values.pop("window_size", 64)),
            chunk_size=int(values.pop("chunk_size", 1024)),
            probability_factor=float(values.pop("prob_sparse_factor", 0.4)),
            frequency_modes=int(values.pop("freq_modes", 32)),
            softpick_chunk_size=int(values.pop("softpick_chunk_size", 128)),
            global_attention_ratio=float(
                values.pop("global_attention_ratio", 0.1)
            ),
            use_flash_sliding=bool(values.pop("use_flash_sliding", True)),
            use_swiglu=bool(values.pop("use_swiglu", True)),
            nsa_block_size=values.pop("nsa_block_size", None),
            nsa_topk_ratio=values.pop("nsa_topk_ratio", None),
            moba_block_size=values.pop("moba_block_size", None),
            moba_topk=int(values.pop("moba_topk", 4)),
            dilation=int(values.pop("attention_dilation", 2)),
            dilated_window_size=values.pop("dilated_window_size", None),
        )
        features = AttentionFeatureConfig(
            qk_norm=bool(values.pop("qk_norm", False)),
            qk_norm_type=str(values.pop("qk_norm_type", "rms")),
            logit_softcap=values.pop("logit_softcap", None),
            learned_temperature=bool(values.pop("use_learned_temp", False)),
            gated_attention=bool(values.pop("use_gated_attention", False)),
            normalized_output=bool(values.pop("use_normalized_attn_out", False)),
            head_importance=bool(values.pop("use_head_importance", False)),
            gated_attention_mode=str(values.pop("gated_attn_mode", "per_head")),
            gated_attention_bias=bool(values.pop("gated_attn_bias", True)),
            temperature_init=float(values.pop("temp_init", 1.0)),
            subquery_norm=bool(values.pop("use_subquery_norm", False)),
            subquery_norm_mode=str(values.pop("subquery_norm_mode", "learned")),
            multiscale_mask=bool(values.pop("use_multiscale_mask", False)),
            multiscale_window_ratio=float(
                values.pop("multiscale_window_ratio", 0.2)
            ),
            multiscale_topk=int(values.pop("multiscale_topk", 16)),
            normalized_output_type=str(values.pop("norm_attn_type", "rms")),
            head_importance_sparsity=float(
                values.pop("head_importance_sparsity", 0.1)
            ),
            verbose_init=bool(values.pop("verbose_init", False)),
        )
        if values:
            raise TypeError(f"unknown legacy attention options: {sorted(values)}")
        return cls(
            shape=shape,
            cache=cache,
            position=position,
            variant=variant,
            features=features,
        )

__all__ = [
    "AttentionCacheConfig",
    "AttentionConfig",
    "AttentionFeatureConfig",
    "AttentionPositionConfig",
    "AttentionShapeConfig",
    "AttentionVariantConfig",
]
