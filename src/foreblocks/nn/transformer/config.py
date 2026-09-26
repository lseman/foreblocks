"""Flat, serializable configuration for Foreblocks transformers.

Every setting is a plain keyword. Modules accept a ``TransformerConfig``, keyword
overrides, or both::

    encoder = TransformerEncoder(input_size=4, d_model=64, attention="linear")

    config = TransformerConfig(d_model=64, n_heads=4, residual="mhc")
    encoder = TransformerEncoder(config, input_size=4)
    decoder = TransformerDecoder(config, input_size=2, output_size=2)

Live collaborators (a custom positional encoder, budget schedulers) are module
constructor arguments, never config fields, so a config is always plain data.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any, Literal

from foreblocks.nn.attention.config import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionFeatureConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
)

AttentionPattern = Literal["uniform", "hybrid", "3to1"]
Position = Literal["rope", "alibi", "sinusoidal", "learnable", "none"]
RopeScaling = Literal["none", "yarn", "ntk", "linear"]
Norm = Literal["rms", "layer"]
NormPlacement = Literal["pre", "post", "sandwich"]
Residual = Literal["standard", "gateskip", "mhc", "mod", "attention"]
Patching = Literal["none", "shared", "channel"]
KVCache = Literal["auto", "dynamic", "paged", "static"]
Role = Literal["encoder", "decoder"]

# Advanced attention settings accepted through ``attention_options``: every
# field of the nested attention groups that has no flat config field.
_ATTENTION_OPTION_GROUPS: dict[str, type] = {
    "cache": AttentionCacheConfig,
    "variant": AttentionVariantConfig,
    "features": AttentionFeatureConfig,
}
_FLAT_ATTENTION_FIELDS = {
    "name",
    "backend",
    "frequency_modes",
    "use_swiglu",
    "use_paged_cache",
}
ATTENTION_OPTIONS: dict[str, str] = {
    item.name: group
    for group, cls in _ATTENTION_OPTION_GROUPS.items()
    for item in fields(cls)
    if item.name not in _FLAT_ATTENTION_FIELDS
}


@dataclass(frozen=True)
class TransformerConfig:
    """Settings for transformer stacks and their layers.

    Attention
        ``attention`` names the backend used by every layer: a recurrent or
        linear backend (``linear``, ``gla``, ``deltanet``, ``gated_deltanet``,
        ``gated_delta``, ``kimi``) or a softmax variant (``standard``, ``sype``,
        ``prob_sparse``, ``frequency``, ...). ``attention_pattern`` mixes it
        with standard attention across depth: ``hybrid`` uses it for every
        layer but the last, ``3to1`` for three of every four layers.
        ``attention_options`` sets any remaining field of the nested attention
        groups by name, e.g. ``{"window_size": 128, "qk_norm": True}``.

    Mixture of experts
        ``moe_experts > 0`` replaces the feed-forward with routed experts.
        ``moe_options`` passes further ``FeedForwardBlock`` settings by name,
        e.g. ``{"num_shared": 1, "router_type": "noisy_topk"}``.

    Residual
        ``residual`` selects one residual policy: ``standard``, ``gateskip``
        (learned token gates, ``gate_*``), ``mhc`` (manifold hyper-connections,
        ``mhc_*``), ``mod`` (Mixture-of-Depths token routing), or
        ``attention`` (attention over earlier layer outputs,
        ``attention_residual_*``).

    Encoder input
        ``patching`` tokenizes the series: ``shared`` embeds patches of the
        projected input, ``channel`` embeds each channel's patches and fuses
        them (``channel_fuse``), ``none`` keeps one token per step.
        ``variate_attention`` keeps each variate as its own token stream and
        mixes across variates in every layer.

    Decoder
        ``informer`` makes self-attention non-causal and masks positions after
        ``label_len`` as padding. ``kv_cache`` selects the incremental cache;
        ``auto`` and ``paged`` use a paged cache.
    """

    # Shape.
    input_size: int = 1
    output_size: int = 1
    d_model: int = 256
    n_heads: int = 8
    n_kv_heads: int | None = None
    num_layers: int = 6
    ff_dim: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    swiglu: bool = True
    max_seq_len: int = 5000
    # Attention.
    attention: str = "standard"
    attention_pattern: AttentionPattern = "uniform"
    attention_kernel: str = "auto"
    frequency_modes: int = 32
    attention_options: Mapping[str, Any] = field(default_factory=dict)
    # Positions.
    position: Position = "rope"
    position_scale: float = 1.0
    rope_base: float = 10000.0
    rope_scaling: RopeScaling = "none"
    rope_scaling_factor: float = 1.0
    time_encoding: bool = False
    # Normalization.
    norm: Norm = "rms"
    norm_placement: NormPlacement = "pre"
    norm_eps: float = 1e-5
    final_norm: bool = True
    # Feed-forward mixture of experts; 0 experts means a dense feed-forward.
    moe_experts: int = 0
    moe_top_k: int = 2
    moe_latent: bool = False
    moe_latent_dim: int | None = None
    moe_latent_ff_dim: int | None = None
    moe_aux_weight: float = 1.0
    moe_options: Mapping[str, Any] = field(default_factory=dict)
    # Residual policy.
    residual: Residual = "standard"
    gate_budget: float | None = None
    gate_aux_weight: float = 0.1
    mhc_streams: int = 4
    mhc_sinkhorn_iters: int = 20
    mhc_collapse: Literal["first", "mean"] = "first"
    mod_aux_weight: float = 0.05
    attention_residual_mode: Literal["full", "block"] = "full"
    attention_residual_block_size: int = 8
    # Stack execution and initialization.
    share_layers: bool = False
    gradient_checkpointing: bool = False
    init_std: float = 0.02
    depth_scaled_init: bool = True
    # Encoder input.
    patching: Patching = "shared"
    patch_len: int = 16
    patch_stride: int = 8
    patch_pad_end: bool = True
    channel_fuse: Literal["mean", "linear"] = "linear"
    variate_attention: bool = False
    variate_fuse: Literal["mean", "linear", "none"] = "linear"
    variate_position: bool = False
    contiguous_decoding: bool = False
    quantiles: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    # Decoder.
    informer: bool = False
    label_len: int = 0
    kv_cache: KVCache = "auto"

    def __post_init__(self) -> None:
        object.__setattr__(self, "attention_options", dict(self.attention_options))
        object.__setattr__(self, "moe_options", dict(self.moe_options))
        object.__setattr__(self, "quantiles", tuple(float(q) for q in self.quantiles))
        _check_positive(self, "input_size", "output_size", "d_model", "n_heads")
        _check_positive(self, "num_layers", "ff_dim", "max_seq_len")
        _check_positive(self, "mhc_streams", "mhc_sinkhorn_iters")
        _check_positive(self, "patch_len", "patch_stride")
        _check_positive(self, "attention_residual_block_size", "frequency_modes")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if self.moe_experts < 0 or (self.moe_experts and self.moe_top_k <= 0):
            raise ValueError("moe_experts must be >= 0 and moe_top_k positive")
        if self.gate_budget is not None and not 0.0 <= self.gate_budget <= 1.0:
            raise ValueError("gate_budget must be in [0, 1]")
        if not self.quantiles or any(not 0.0 < q < 1.0 for q in self.quantiles):
            raise ValueError("quantiles must be non-empty and strictly in (0, 1)")
        if tuple(sorted(self.quantiles)) != self.quantiles:
            raise ValueError("quantiles must be sorted")
        _check_choice(self, "attention_pattern", AttentionPattern)
        _check_choice(self, "position", Position)
        _check_choice(self, "rope_scaling", RopeScaling)
        _check_choice(self, "norm", Norm)
        _check_choice(self, "norm_placement", NormPlacement)
        _check_choice(self, "residual", Residual)
        _check_choice(self, "mhc_collapse", Literal["first", "mean"])
        _check_choice(self, "attention_residual_mode", Literal["full", "block"])
        _check_choice(self, "patching", Patching)
        _check_choice(self, "channel_fuse", Literal["mean", "linear"])
        _check_choice(self, "variate_fuse", Literal["mean", "linear", "none"])
        _check_choice(self, "kv_cache", KVCache)
        unknown = sorted(set(self.attention_options) - ATTENTION_OPTIONS.keys())
        if unknown:
            raise ValueError("unknown attention_options: " + ", ".join(unknown))
        _check_attention_name(self.attention)
        _check_moe_options(self.moe_options)
        if self.gradient_checkpointing and self.residual in {"mhc", "attention"}:
            raise ValueError(
                f"gradient_checkpointing is incompatible with residual={self.residual!r}"
            )
        if self.variate_attention:
            conflicts = [
                name
                for name, conflict in {
                    "patching='channel'": self.patching == "channel",
                    f"residual={self.residual!r}": self.residual != "standard",
                    "moe_experts": self.moe_experts > 0,
                    f"attention_pattern={self.attention_pattern!r}": (
                        self.share_layers and self.attention_pattern != "uniform"
                    ),
                }.items()
                if conflict
            ]
            if conflicts:
                raise ValueError(
                    "variate_attention is incompatible with " + ", ".join(conflicts)
                )
        if self.contiguous_decoding:
            if not self.variate_attention or self.patching != "shared":
                raise ValueError(
                    "contiguous_decoding requires variate_attention=True and "
                    "patching='shared'"
                )
            if self.patch_stride != self.patch_len:
                raise ValueError(
                    "contiguous_decoding requires patch_stride == patch_len"
                )

    # ---- Construction ---------------------------------------------------------
    @classmethod
    def resolve(
        cls, config: TransformerConfig | None = None, **overrides: Any
    ) -> TransformerConfig:
        """``config`` (or the defaults) with keyword overrides applied."""
        if config is None:
            return cls(**overrides)
        if not isinstance(config, cls):
            raise TypeError(f"expected TransformerConfig, got {type(config).__name__}")
        return replace(config, **overrides) if overrides else config

    def validate_for(self, role: Role) -> None:
        """Checks that depend on whether the config builds an encoder or decoder."""
        if role == "decoder":
            if self.variate_attention:
                raise ValueError("variate_attention is only supported by the encoder")
            if self.residual == "mhc" and self.kv_cache in {"static", "paged"}:
                raise ValueError(
                    "residual='mhc' does not support static or paged KV caches"
                )

    # ---- Derived settings -----------------------------------------------------
    @property
    def use_moe(self) -> bool:
        return self.moe_experts > 0

    def layer_attention(self, index: int) -> str:
        """Attention backend of the layer at depth ``index``."""
        if self.attention_pattern == "uniform":
            return self.attention
        if self.attention_pattern == "hybrid":
            use_backend = index < self.num_layers - 1
        else:
            use_backend = index % 4 < 3
        return self.attention if use_backend else "standard"

    def for_layer(
        self, index: int, *, dropout: float | None = None
    ) -> TransformerConfig:
        """The uniform config of the layer at depth ``index``."""
        changes: dict[str, Any] = {}
        if self.attention_pattern != "uniform":
            changes["attention"] = self.layer_attention(index)
            changes["attention_pattern"] = "uniform"
        if dropout is not None and dropout != self.dropout:
            changes["dropout"] = dropout
        return replace(self, **changes) if changes else self

    def attention_config(self, *, cross: bool = False) -> AttentionConfig:
        """Nested configuration consumed by ``foreblocks.nn.attention``."""
        options: dict[str, dict[str, Any]] = {
            group: {} for group in _ATTENTION_OPTION_GROUPS
        }
        for name, value in self.attention_options.items():
            options[ATTENTION_OPTIONS[name]][name] = value
        # Attention-matching compaction works on full K/V, not an MLA latent.
        if options["cache"].get("attention_matching"):
            options["cache"].setdefault("use_mla", False)
        return AttentionConfig(
            shape=AttentionShapeConfig(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_kv_heads=self.n_kv_heads,
                dropout=self.dropout,
                max_seq_len=self.max_seq_len,
                cross_attention=cross,
            ),
            cache=AttentionCacheConfig(
                use_paged_cache=self.kv_cache in {"auto", "paged"},
                **options["cache"],
            ),
            position=AttentionPositionConfig(
                encoding=self.position,
                rope_base=self.rope_base,
                rope_scaling_type=self.rope_scaling,
                rope_scaling_factor=self.rope_scaling_factor,
            ),
            variant=AttentionVariantConfig(
                name=self.attention,
                backend=self.attention_kernel,
                frequency_modes=self.frequency_modes,
                use_swiglu=self.swiglu,
                **options["variant"],
            ),
            features=AttentionFeatureConfig(**options["features"]),
        )

    # ---- Serialization --------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> TransformerConfig:
        return cls(**values)


def _check_positive(config: TransformerConfig, *names: str) -> None:
    for name in names:
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive")


def _check_choice(config: TransformerConfig, name: str, choices: Any) -> None:
    allowed = choices.__args__
    value = getattr(config, name)
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _check_attention_name(name: str) -> None:
    # Imported lazily: the backend registries pull in the attention modules.
    from foreblocks.nn.attention.variants.registry import ATTENTION_VARIANTS
    from foreblocks.nn.transformer.attention_backends import LAYER_ATTENTION_BACKENDS

    known = set(LAYER_ATTENTION_BACKENDS) | set(ATTENTION_VARIANTS.names())
    if name not in known:
        raise ValueError(
            f"unknown attention {name!r}; expected one of: {', '.join(sorted(known))}"
        )


# FeedForwardBlock arguments owned by flat fields, or live objects.
_MOE_RESERVED = {
    "self",
    "d_model",
    "dim_ff",
    "dropout",
    "use_swiglu",
    "activation",
    "use_moe",
    "num_experts",
    "top_k",
    "moe_use_latent",
    "moe_latent_dim",
    "moe_latent_d_ff",
    "moe_logger",
    "step_getter",
}


def _check_moe_options(options: Mapping[str, Any]) -> None:
    if not options:
        return
    import inspect

    from foreblocks.nn.moe.feedforward import FeedForwardBlock

    known = set(inspect.signature(FeedForwardBlock.__init__).parameters) - _MOE_RESERVED
    unknown = sorted(set(options) - known)
    if unknown:
        raise ValueError("unknown moe_options: " + ", ".join(unknown))


@dataclass(frozen=True)
class GenerationConfig:
    """Generation-time configuration, independent from decoder construction."""

    max_new_tokens: int = 1
    return_dict: bool = True
    use_cache: bool = True

    def __post_init__(self) -> None:
        if self.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")


__all__ = ["ATTENTION_OPTIONS", "GenerationConfig", "TransformerConfig"]
