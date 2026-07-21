"""Serializable configuration for Foreblocks transformer models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

_ATTENTION_ALIASES: dict[str, str] = {
    "hybrid_linear": "hybrid",
    "kimi_hybrid": "hybrid_kimi",
    "gdn_hybrid": "hybrid_gdn",
    "hybrid_gla": "gla_hybrid",
    "hybrid_deltanet": "deltanet_hybrid",
    "hybrid_gated_deltanet": "gated_deltanet_hybrid",
}

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


def _resolve_attention_mode(mode: str) -> str:
    return _ATTENTION_ALIASES.get(mode, mode)


@dataclass(frozen=True)
class AttentionConfig:
    architecture: Literal[
        "standard",
        "linear",
        "sype",
        "hybrid",
        "kimi",
        "gated_delta",
        "gla",
        "deltanet",
        "gated_deltanet",
        "kimi_hybrid",
        "hybrid_kimi",
        "hybrid_gdn",
        "gdn_hybrid",
        "gla_hybrid",
        "hybrid_gla",
        "deltanet_hybrid",
        "hybrid_deltanet",
        "gated_deltanet_hybrid",
        "hybrid_gated_deltanet",
        "kimi_3to1",
        "gdn_3to1",
        "gla_3to1",
        "deltanet_3to1",
        "gated_deltanet_3to1",
    ]
    backend: Literal["auto", "sdpa", "flash", "cudnn"]
    position: Literal["rope", "alibi", "sinusoidal", "learnable"]


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
    model_type: str = "transformer"
    attention_mode: str = "standard"
    attn_implementation: str = "auto"
    att_type: str = "standard"
    norm_strategy: str = "pre_norm"
    custom_norm: str = "rms"
    layer_norm_eps: float = 1e-5
    pos_encoding_scale: float = 1.0
    pos_encoding_type: str = "rope"
    rope_base: float = 10000.0
    rope_scaling_type: Literal["none", "yarn", "ntk", "linear"] = "none"
    rope_scaling_factor: float = 1.0
    patch_encoder: bool = True
    patch_len: int = 16
    patch_stride: int = 8
    patch_pad_end: bool = True
    use_gradient_checkpointing: bool = False
    share_layers: bool = False
    use_final_norm: bool = True
    use_swiglu: bool = True
    freq_modes: int = 32
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
    use_attention_matching_compaction: bool = False
    attention_matching_keep_ratio: float = 0.25
    attention_matching_trigger_len: int = 512
    attention_matching_min_keep: int = 64
    attention_matching_query_budget: int = 64
    attention_matching_force_single_step: bool = False
    moba_block_size: int | None = None
    moba_topk: int = 4
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
        # Resolve attention-mode aliases before validation.
        # This runs for both direct construction (TransformerConfig(...))
        # and from_dict(), ensuring canonical values everywhere.
        self.attention_mode = _resolve_attention_mode(self.attention_mode)

        # Accept the pre-first-class representation when loading or directly
        # constructing older configs. New code should pass these as fields.
        known = {item.name for item in fields(self)} - {"options"}
        promoted = known & self.options.keys()
        for name in promoted:
            setattr(self, name, self.options[name])
        if promoted:
            self.options = {
                name: value
                for name, value in self.options.items()
                if name not in promoted
            }

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
        if self.layer_norm_eps <= 0:
            raise ValueError("layer_norm_eps must be positive")
        if self.freq_modes <= 0:
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
    def attention(self) -> AttentionConfig:
        return AttentionConfig(
            architecture=self.attention_mode,
            backend=self.attn_implementation,
            position=self.pos_encoding_type,
        )

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

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> TransformerConfig:
        known = {item.name for item in fields(cls)} - {"options"}
        values = {key: value for key, value in kwargs.items() if key in known}
        values["options"] = {
            key: value for key, value in kwargs.items() if key not in known
        }
        return cls(**values)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, **overrides: Any) -> TransformerConfig:
        values = {
            item.name: getattr(self, item.name)
            for item in fields(self)
            if item.name != "options"
        }
        options = dict(self.options)
        values.update(options)
        values.update(overrides)
        return type(self).from_kwargs(**values)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> TransformerConfig:
        values = dict(values)
        options = dict(values.get("options", {}))
        known = {item.name for item in fields(cls)} - {"options"}
        for name in known & options.keys():
            values.setdefault(name, options.pop(name))
        values["options"] = options
        if "attention_mode" in values:
            values["attention_mode"] = _resolve_attention_mode(values["attention_mode"])
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
