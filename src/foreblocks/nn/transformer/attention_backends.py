"""Layer-level attention construction from a single grouped configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

import torch.nn as nn

from foreblocks.nn.attention.config import AttentionConfig

if TYPE_CHECKING:
    from foreblocks.nn.transformer.config import TransformerConfig

AttentionKwargsFactory = Callable[[AttentionConfig], dict[str, object]]
_MULTI_ATTENTION_PATH = "foreblocks.nn.attention.multihead.MultiAttention"


class LazyAttentionOwner(Protocol):
    config: TransformerConfig
    layer_attention_type: str
    training: bool

    def parameters(self, recurse: bool = True): ...
    def add_module(self, name: str, module: nn.Module | None) -> None: ...


@dataclass(frozen=True)
class LayerAttentionBackendSpec:
    """Construct a backend from a concrete class or deferred absolute class path."""

    name: str
    module_cls: type[nn.Module] | str
    extra_kwargs: AttentionKwargsFactory = field(default=lambda config: {})

    def build(self, config: AttentionConfig) -> nn.Module:
        module_cls = self.module_cls
        if isinstance(module_cls, str):
            grouped_config = module_cls == _MULTI_ATTENTION_PATH
            module_name, class_name = module_cls.rsplit(".", 1)
            module_cls = getattr(import_module(module_name), class_name)
        else:
            from foreblocks.nn.attention.multihead import MultiAttention

            grouped_config = module_cls is MultiAttention
        shape = config.shape
        if grouped_config:
            configured = replace(
                config,
                variant=replace(config.variant, name=self.name),
            )
            return module_cls(configured)
        return module_cls(
            d_model=shape.d_model,
            n_heads=shape.n_heads,
            dropout=shape.dropout,
            cross_attention=shape.cross_attention,
            pos_encoding_type=config.position.encoding,
            **self.extra_kwargs(config),
        )


def _linear_kwargs(config: AttentionConfig) -> dict[str, object]:
    return {"backend": "rda", "state": "elu"}


def _gla_kwargs(config: AttentionConfig) -> dict[str, object]:
    return {"backend": "gla", "mode": "chunk", "chunk_size": 64}


def _deltanet_kwargs(config: AttentionConfig) -> dict[str, object]:
    return {"backend": "deltanet", "mode": "chunk", "chunk_size": 64}


def _gdn_modern_kwargs(config: AttentionConfig) -> dict[str, object]:
    return {"chunk_size": 64}


LAYER_ATTENTION_BACKENDS: dict[str, LayerAttentionBackendSpec] = {
    "standard": LayerAttentionBackendSpec("standard", _MULTI_ATTENTION_PATH),
    "sype": LayerAttentionBackendSpec("sype", _MULTI_ATTENTION_PATH),
    "linear": LayerAttentionBackendSpec(
        "linear",
        "foreblocks.nn.attention.algorithms.linear.wrapper.ModernLinearAttention",
        _linear_kwargs,
    ),
    "gla": LayerAttentionBackendSpec(
        "gla",
        "foreblocks.nn.attention.algorithms.linear.wrapper.ModernLinearAttention",
        _gla_kwargs,
    ),
    "deltanet": LayerAttentionBackendSpec(
        "deltanet",
        "foreblocks.nn.attention.algorithms.linear.wrapper.ModernLinearAttention",
        _deltanet_kwargs,
    ),
    "gated_deltanet": LayerAttentionBackendSpec(
        "gated_deltanet",
        "foreblocks.nn.attention.algorithms.linear.gated_delta.GatedDeltaNetBackend",
        _gdn_modern_kwargs,
    ),
    "kimi": LayerAttentionBackendSpec(
        "kimi", "foreblocks.nn.attention.algorithms.linear.kimi.KimiAttentionBackend"
    ),
    "gated_delta": LayerAttentionBackendSpec(
        "gated_delta",
        "foreblocks.nn.attention.algorithms.linear.gated_delta.GatedDeltaNetBackend",
    ),
}


def build_layer_attention_backend(name: str, config: AttentionConfig) -> nn.Module:
    spec = LAYER_ATTENTION_BACKENDS.get(name)
    if spec is not None:
        return spec.build(config)
    # MultiAttention validates registered variants; never silently fall back.
    return LayerAttentionBackendSpec(name, _MULTI_ATTENTION_PATH).build(config)


class LazyAttentionBackendMixin:
    """Lazily construct only the attention implementations selected by a layer."""

    layer_attention_type: str = "standard"

    def _ensure_attn_backend(self, name: str) -> nn.Module:
        owner = cast(LazyAttentionOwner, cast(object, self))
        cache: dict[str, nn.Module] = self.__dict__.setdefault("_attn_backends", {})
        module = cache.get(name)
        if module is None:
            parameter = next(owner.parameters())
            attention = owner.config.attention_config()
            module = build_layer_attention_backend(name, attention).to(
                device=parameter.device, dtype=parameter.dtype
            )
            module.train(owner.training)
            cache[name] = module
            owner.add_module(f"_attn_backend_{name}", module)
        return module

    def _self_attn(self) -> nn.Module:
        return self._ensure_attn_backend(self.layer_attention_type)

    def set_layer_attention_type(self, layer_attention_type: str) -> None:
        self.layer_attention_type = str(layer_attention_type)

    def materialize_attention_type(
        self, layer_attention_type: str | None = None
    ) -> nn.Module:
        previous = self.layer_attention_type
        if layer_attention_type is not None:
            self.layer_attention_type = str(layer_attention_type)
        try:
            return self._self_attn()
        finally:
            self.layer_attention_type = previous


__all__ = [
    "LAYER_ATTENTION_BACKENDS",
    "LayerAttentionBackendSpec",
    "LazyAttentionBackendMixin",
    "build_layer_attention_backend",
]
