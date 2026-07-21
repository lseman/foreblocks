"""Serializable configuration for Foreblocks transformer models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

from foreblocks.modules.attention.config import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionFeatureConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)

_ATTENTION_MODES: set[str] = {
    "standard",
    "linear",
    "sype",
    "hybrid",
    "kimi",
    "hybrid_kimi",
    "kimi_3to1",
    "gated_delta",
    "hybrid_gdn",
    "gdn_3to1",
    "gla",
    "gla_hybrid",
    "gla_3to1",
    "deltanet",
    "deltanet_hybrid",
    "deltanet_3to1",
    "gated_deltanet",
    "gated_deltanet_hybrid",
    "gated_deltanet_3to1",
}

_SUPPORTED_OPTIONS: set[str] = {
    "pos_encoder",
    "mod_budget_scheduler",
    "layer_dropout_schedule",
}


@dataclass(frozen=True)
class ResidualConfig:
    policy: Literal["standard", "gateskip", "mhc", "attention_residual", "mod"]
    norm_strategy: Literal["pre_norm", "post_norm", "sandwich_norm"]
    norm_type: Literal["rms", "layer", "layernorm", "rmsnorm"]


@dataclass(frozen=True)
class CacheConfig:
    implementation: Literal["auto", "dynamic", "paged", "static"]


@dataclass
class TransformerConfig:
    input_size: int = 1
    output_size: int = 1
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 5000
    attention: AttentionConfig | None = None
    model_type: str = "transformer"
    norm_strategy: str = "pre_norm"
    custom_norm: str = "rms"
    layer_norm_eps: float = 1e-5
    pos_encoding_scale: float = 1.0
    patch_encoder: bool = True
    patch_len: int = 16
    patch_stride: int = 8
    patch_pad_end: bool = True
    use_gradient_checkpointing: bool = False
    share_layers: bool = False
    use_final_norm: bool = True
    use_moe: bool = False
    num_experts: int = 8
    top_k: int = 2
    moe_use_latent: bool = False
    moe_latent_dim: int | None = None
    moe_latent_d_ff: int | None = None
    moe_aux_lambda: float = 1.0
    use_gateskip: bool = False
    gate_budget: float | None = None
    gate_lambda: float = 0.1
    use_mhc: bool = False
    mhc_n_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_collapse: Literal["first", "mean"] = "first"
    use_mod: bool = False
    mod_mode: Literal["token", "seq"] = "token"
    mod_lambda: float = 0.05
    use_attention_residual: bool = False
    attn_residual_type: str = "full"
    attention_residual_block_size: int = 8
    initializer_range: float = 0.02
    depth_scaled_init: bool = True
    cache_implementation: Literal["auto", "dynamic", "paged", "static"] = "auto"
    label_len: int = 0
    informer_like: bool = True
    use_time_encoding: bool = False
    ct_patchtst: bool = False
    ct_patch_len: int = 16
    ct_patch_stride: int = 8
    ct_patch_pad_end: bool = True
    ct_patch_fuse: Literal["mean", "linear"] = "linear"
    output_hidden_states: bool = False
    output_attentions: bool = False
    return_dict: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.attention is None:
            self.attention = AttentionConfig(
                shape=AttentionShapeConfig(
                    d_model=self.d_model,
                    n_heads=self.nhead,
                    dropout=self.dropout,
                    max_seq_len=self.max_seq_len,
                )
            )

        if self.input_size <= 0 or self.output_size <= 0:
            raise ValueError("input_size and output_size must be positive")
        if self.d_model <= 0 or self.nhead <= 0 or self.num_layers <= 0:
            raise ValueError("d_model, nhead, and num_layers must be positive")
        if self.d_model % self.nhead:
            raise ValueError("d_model must be divisible by nhead")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.cache_implementation not in {"auto", "dynamic", "paged", "static"}:
            raise ValueError("unsupported cache_implementation")
        if self.attention.shape.d_model != self.d_model:
            raise ValueError("attention.shape.d_model must match d_model")
        if self.attention.shape.n_heads != self.nhead:
            raise ValueError("attention.shape.n_heads must match nhead")
        if self.attention.architecture not in _ATTENTION_MODES:
            raise ValueError(
                f"unsupported attention architecture: {self.attention.architecture}"
            )
        if self.norm_strategy not in {"pre_norm", "post_norm", "sandwich_norm"}:
            raise ValueError(f"unsupported norm_strategy: {self.norm_strategy}")
        if self.custom_norm not in {"rms", "layer", "layernorm", "rmsnorm"}:
            raise ValueError(f"unsupported custom_norm: {self.custom_norm}")
        if self.attention.position.encoding not in {
            "rope",
            "alibi",
            "sinusoidal",
            "learnable",
        }:
            raise ValueError(
                f"unsupported position encoding: {self.attention.position.encoding}"
            )
        if self.attention.position.rope_scaling_type not in {
            "none",
            "yarn",
            "ntk",
            "linear",
        }:
            raise ValueError(
                "unsupported rope scaling type: "
                f"{self.attention.position.rope_scaling_type}"
            )
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")
        if self.attention.variant.frequency_modes <= 0:
            raise ValueError("freq_modes must be positive")
        if self.gate_budget is not None and not 0.0 <= self.gate_budget <= 1.0:
            raise ValueError("gate_budget must be in [0, 1]")
        if self.mhc_n_streams <= 0 or self.mhc_sinkhorn_iters <= 0:
            raise ValueError("mhc_n_streams and mhc_sinkhorn_iters must be positive")
        if self.attention_residual_block_size <= 0:
            raise ValueError("attention_residual_block_size must be positive")
        unsupported = sorted(set(self.options) - _SUPPORTED_OPTIONS)
        if unsupported:
            raise ValueError(
                "unsupported Transformer options: " + ", ".join(unsupported)
            )
        self.validate_compatibility()

    @property
    def residual(self) -> ResidualConfig:
        if self.use_attention_residual:
            policy = "attention_residual"
        elif self.use_mhc:
            policy = "mhc"
        elif self.use_mod:
            policy = "mod"
        elif self.use_gateskip:
            policy = "gateskip"
        else:
            policy = "standard"
        return ResidualConfig(policy, self.norm_strategy, self.custom_norm)

    @property
    def cache(self) -> CacheConfig:
        return CacheConfig(self.cache_implementation)

    def option(self, name: str, default: Any = None) -> Any:
        return self.options.get(name, default)

    def validate_compatibility(self, *, role: str | None = None) -> None:
        attention_residual = self.use_attention_residual
        mod_mode = self.mod_mode
        if attention_residual and self.use_gateskip:
            raise ValueError("use_attention_residual is incompatible with use_gateskip")
        if attention_residual and self.use_mhc:
            raise ValueError("use_attention_residual is incompatible with use_mhc")
        if attention_residual and self.use_mod:
            raise ValueError("use_attention_residual is incompatible with use_mod")
        if self.use_gradient_checkpointing and attention_residual:
            raise ValueError(
                "use_gradient_checkpointing is incompatible with use_attention_residual"
            )
        if self.use_gradient_checkpointing and self.use_mhc:
            raise ValueError("use_gradient_checkpointing is incompatible with use_mhc")
        if self.use_mod and self.use_gateskip:
            raise ValueError("use_mod is incompatible with use_gateskip")
        if self.use_mod and self.use_mhc:
            raise ValueError("use_mod is incompatible with use_mhc")
        if self.use_mod and mod_mode != "token":
            raise ValueError("use_mod currently requires mod_mode='token'")
        if (
            role == "decoder"
            and self.use_mhc
            and self.cache_implementation in {"static", "paged"}
        ):
            raise ValueError(
                "decoder use_mhc does not support static/paged KV caching; "
                "use dynamic full-sequence execution"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> TransformerConfig:
        return replace(self, **overrides)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TransformerConfig:
        values = dict(values)
        attention = values.get("attention")
        if isinstance(attention, dict):
            values["attention"] = AttentionConfig(
                shape=AttentionShapeConfig(**attention["shape"]),
                architecture=attention.get("architecture", "standard"),
                cache=AttentionCacheConfig(**attention.get("cache", {})),
                position=AttentionPositionConfig(**attention.get("position", {})),
                variant=AttentionVariantConfig(**attention.get("variant", {})),
                features=AttentionFeatureConfig(**attention.get("features", {})),
            )
        return cls(**values)

    @classmethod
    def from_legacy_dict(
        cls, values: dict[str, Any] | None = None, **kwargs: Any
    ) -> TransformerConfig:
        """Load the former flat attention configuration representation."""
        values = {**(values or {}), **kwargs}
        legacy_options = dict(values.pop("options", {}))
        known_fields = {item.name for item in fields(cls)} - {"options"}
        for name in known_fields & legacy_options.keys():
            values.setdefault(name, legacy_options.pop(name))
        attention = AttentionConfig(
            shape=AttentionShapeConfig(
                d_model=values.get("d_model", 256),
                n_heads=values.get("nhead", 8),
                dropout=values.get("dropout", 0.1),
                max_seq_len=values.get("max_seq_len", 5000),
            ),
            architecture=values.pop("attention_mode", "standard"),
            cache=AttentionCacheConfig(
                use_paged_cache=values.get("cache_implementation", "auto")
                in {"auto", "paged"},
                use_mla=not values.pop("use_attention_matching_compaction", False),
                matching_keep_ratio=values.pop(
                    "attention_matching_keep_ratio", 0.25
                ),
                matching_trigger_len=values.pop(
                    "attention_matching_trigger_len", 512
                ),
                matching_min_keep=values.pop("attention_matching_min_keep", 64),
                matching_query_budget=values.pop(
                    "attention_matching_query_budget", 64
                ),
                matching_force_single_step=values.pop(
                    "attention_matching_force_single_step", False
                ),
            ),
            position=AttentionPositionConfig(
                encoding=values.pop("pos_encoding_type", "rope"),
                rope_base=values.pop("rope_base", 10000.0),
                rope_scaling_type=values.pop("rope_scaling_type", "none"),
                rope_scaling_factor=values.pop("rope_scaling_factor", 1.0),
            ),
            variant=AttentionVariantConfig(
                name=values.pop("att_type", "standard"),
                backend=values.pop("attn_implementation", "auto"),
                frequency_modes=values.pop("freq_modes", 32),
                use_swiglu=values.pop("use_swiglu", True),
                moba_block_size=values.pop("moba_block_size", None),
                moba_topk=values.pop("moba_topk", 4),
            ),
        )
        attention_matching = not attention.cache.use_mla
        attention = replace(
            attention,
            cache=replace(
                attention.cache, attention_matching=attention_matching
            ),
        )
        values["attention"] = attention
        known = {item.name for item in fields(cls)} - {"options"}
        options = legacy_options
        options.update(
            (name, values.pop(name)) for name in list(values) if name not in known
        )
        values["options"] = options
        return cls(**values)

    def model_kwargs(self) -> dict[str, Any]:
        excluded = {
            "input_size",
            "output_size",
            "model_type",
            "label_len",
            "informer_like",
            "use_time_encoding",
            "cache_implementation",
            "ct_patchtst",
            "ct_patch_len",
            "ct_patch_stride",
            "ct_patch_pad_end",
            "ct_patch_fuse",
            "output_hidden_states",
            "output_attentions",
            "return_dict",
            "options",
        }
        values = self.to_dict()
        kwargs = {key: value for key, value in values.items() if key not in excluded}
        kwargs.update(self.options)
        return kwargs


__all__ = ["AttentionConfig", "CacheConfig", "ResidualConfig", "TransformerConfig"]
