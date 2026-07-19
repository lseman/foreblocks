"""Serializable configuration for Foreblocks transformer models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal


_ATTENTION_ALIASES = {
    "hybrid_linear": "hybrid",
    "kimi_hybrid": "hybrid_kimi",
    "gdn_hybrid": "hybrid_gdn",
    "hybrid_gla": "gla_hybrid",
    "hybrid_deltanet": "deltanet_hybrid",
    "hybrid_gated_deltanet": "gated_deltanet_hybrid",
}

_ATTENTION_MODES = {
    "standard", "linear", "sype", "hybrid", "kimi", "hybrid_kimi",
    "kimi_3to1", "gated_delta", "hybrid_gdn", "gdn_3to1", "gla",
    "gla_hybrid", "gla_3to1", "deltanet", "deltanet_hybrid",
    "deltanet_3to1", "gated_deltanet", "gated_deltanet_hybrid",
    "gated_deltanet_3to1",
}

_SUPPORTED_OPTIONS = {
    "att_type", "layer_norm_eps", "pos_encoding_scale", "pos_encoder",
    "share_layers", "use_final_norm", "use_swiglu", "freq_modes",
    "gate_budget", "gate_lambda", "mhc_n_streams", "mhc_sinkhorn_iters",
    "mhc_collapse", "mod_mode", "mod_lambda", "mod_budget_scheduler",
    "moe_aux_lambda", "use_attention_residual", "attn_residual_type",
    "attention_residual_block_size", "layer_dropout_schedule",
    "initializer_range", "depth_scaled_init", "moe_use_latent",
    "moe_latent_dim", "moe_latent_d_ff", "use_attention_matching_compaction",
    "attention_matching_keep_ratio", "attention_matching_trigger_len",
    "attention_matching_min_keep", "attention_matching_query_budget",
    "attention_matching_force_single_step", "moba_block_size", "moba_topk",
}


@dataclass(frozen=True)
class AttentionConfig:
    architecture: str
    backend: str
    position: str


@dataclass(frozen=True)
class ResidualConfig:
    policy: Literal["standard", "gateskip", "mhc", "attention_residual", "mod"]
    norm_strategy: str
    norm_type: str


@dataclass(frozen=True)
class CacheConfig:
    implementation: Literal["auto", "dynamic", "paged", "static"]


@dataclass
class TransformerConfig:
    """Single source of truth for encoder and decoder construction.

    Less common experimental options remain supported through ``options`` while
    stable architecture and output controls have first-class fields.
    """

    input_size: int = 1
    output_size: int = 1
    d_model: int = 256
    nhead: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 5000
    model_type: str = "transformer"
    attention_mode: str = "standard"
    attn_implementation: str = "auto"
    norm_strategy: str = "pre_norm"
    custom_norm: str = "rms"
    pos_encoding_type: str = "rope"
    rope_base: float = 10000.0
    rope_scaling_type: Literal["none", "yarn", "ntk", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    patch_encoder: bool = True
    patch_len: int = 16
    patch_stride: int = 8
    patch_pad_end: bool = True
    use_gradient_checkpointing: bool = False
    use_moe: bool = False
    num_experts: int = 8
    top_k: int = 2
    use_gateskip: bool = False
    use_mhc: bool = False
    use_mod: bool = False
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
        self.attention_mode = _ATTENTION_ALIASES.get(
            self.attention_mode, self.attention_mode
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
        if self.attention_mode not in _ATTENTION_MODES:
            raise ValueError(f"unsupported attention_mode: {self.attention_mode}")
        if self.norm_strategy not in {"pre_norm", "post_norm", "sandwich_norm"}:
            raise ValueError(f"unsupported norm_strategy: {self.norm_strategy}")
        if self.custom_norm not in {"rms", "layer", "layernorm", "rmsnorm"}:
            raise ValueError(f"unsupported custom_norm: {self.custom_norm}")
        if self.pos_encoding_type not in {"rope", "alibi", "sinusoidal", "learnable"}:
            raise ValueError(f"unsupported pos_encoding_type: {self.pos_encoding_type}")
        if self.rope_scaling_type not in {"none", "yarn", "ntk", "linear"}:
            raise ValueError(f"unsupported rope_scaling_type: {self.rope_scaling_type}")
        unsupported = sorted(set(self.options) - _SUPPORTED_OPTIONS)
        if unsupported:
            raise ValueError(
                "unsupported Transformer options: " + ", ".join(unsupported)
            )
        self.validate_compatibility()

    @property
    def attention(self) -> AttentionConfig:
        return AttentionConfig(
            architecture=self.attention_mode,
            backend=self.attn_implementation,
            position=self.pos_encoding_type,
        )

    @property
    def residual(self) -> ResidualConfig:
        if self.option("use_attention_residual", False):
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
        """Read an experimental option without leaking the options mapping."""
        return self.options.get(name, default)

    def validate_compatibility(self, *, role: str | None = None) -> None:
        """Reject unsupported feature combinations before model execution."""
        attention_residual = bool(self.option("use_attention_residual", False))
        mod_mode = str(self.option("mod_mode", "token"))
        if attention_residual and self.use_gateskip:
            raise ValueError("use_attention_residual is incompatible with use_gateskip")
        if attention_residual and self.use_mhc:
            raise ValueError("use_attention_residual is incompatible with use_mhc")
        if attention_residual and self.use_mod:
            raise ValueError("use_attention_residual is incompatible with use_mod")
        if self.use_mod and self.use_gateskip:
            raise ValueError("use_mod is incompatible with use_gateskip")
        if self.use_mod and self.use_mhc:
            raise ValueError("use_mod is incompatible with use_mhc")
        if self.use_mod and mod_mode != "token":
            raise ValueError("use_mod currently requires mod_mode='token'")
        if role == "decoder" and self.use_mhc and self.cache_implementation in {
            "static", "paged"
        }:
            raise ValueError(
                "decoder use_mhc does not support static/paged KV caching; "
                "use dynamic full-sequence execution"
            )

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> "TransformerConfig":
        known = {item.name for item in fields(cls)} - {"options"}
        values = {key: value for key, value in kwargs.items() if key in known}
        values["options"] = {
            key: value for key, value in kwargs.items() if key not in known
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TransformerConfig":
        return cls(**values)

    def model_kwargs(self) -> dict[str, Any]:
        excluded = {
            "input_size", "output_size", "model_type", "label_len",
            "informer_like", "use_time_encoding", "cache_implementation",
            "ct_patchtst", "ct_patch_len", "ct_patch_stride", "ct_patch_pad_end",
            "ct_patch_fuse",
            "output_hidden_states", "output_attentions", "return_dict", "options",
        }
        values = self.to_dict()
        kwargs = {key: value for key, value in values.items() if key not in excluded}
        kwargs.update(self.options)
        return kwargs


__all__ = [
    "AttentionConfig", "CacheConfig", "ResidualConfig", "TransformerConfig"
]
