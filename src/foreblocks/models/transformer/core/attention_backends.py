"""Layer-level attention construction from a single grouped configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Protocol, cast

import torch.nn as nn

from foreblocks.modules.attention.config import AttentionConfig
from foreblocks.modules.attention.implementations import (
    GatedDeltaNet,
    KimiAttention,
    ModernLinearAttention,
)
from foreblocks.modules.attention.multi_att import MultiAttention

AttentionKwargsFactory = Callable[[AttentionConfig], dict[str, object]]


class LazyAttentionOwner(Protocol):
    _attention_config: AttentionConfig
    layer_attention_type: str

    def parameters(self, recurse: bool = True): ...
    def add_module(self, name: str, module: nn.Module | None) -> None: ...


@dataclass(frozen=True)
class LayerAttentionBackendSpec:
    name: str
    module_cls: type[nn.Module]
    extra_kwargs: AttentionKwargsFactory = field(default=lambda config: {})

    def build(self, config: AttentionConfig) -> nn.Module:
        shape = config.shape
        if self.module_cls is MultiAttention:
            configured = replace(
                config,
                variant=replace(config.variant, name=self.name),
            )
            return MultiAttention(configured)
        return self.module_cls(
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
    "standard": LayerAttentionBackendSpec("standard", MultiAttention),
    "sype": LayerAttentionBackendSpec("sype", MultiAttention),
    "linear": LayerAttentionBackendSpec(
        "linear", ModernLinearAttention, _linear_kwargs
    ),
    "gla": LayerAttentionBackendSpec("gla", ModernLinearAttention, _gla_kwargs),
    "deltanet": LayerAttentionBackendSpec(
        "deltanet", ModernLinearAttention, _deltanet_kwargs
    ),
    "gated_deltanet": LayerAttentionBackendSpec(
        "gated_deltanet", GatedDeltaNet, _gdn_modern_kwargs
    ),
    "kimi": LayerAttentionBackendSpec("kimi", KimiAttention),
    "gated_delta": LayerAttentionBackendSpec("gated_delta", GatedDeltaNet),
}


def build_layer_attention_backend(name: str, config: AttentionConfig) -> nn.Module:
    spec = LAYER_ATTENTION_BACKENDS.get(name, LAYER_ATTENTION_BACKENDS["standard"])
    return spec.build(config)


class LazyAttentionBackendMixin:
    """Lazily construct only the attention implementations selected by a layer."""

    layer_attention_type: str = "standard"

    def _ensure_attn_backend(self, name: str) -> nn.Module:
        owner = cast(LazyAttentionOwner, cast(object, self))
        cache: dict[str, nn.Module] = self.__dict__.setdefault("_attn_backends", {})
        module = cache.get(name)
        if module is None:
            parameter = next(owner.parameters())
            module = build_layer_attention_backend(name, owner._attention_config).to(
                parameter.device
            )
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
