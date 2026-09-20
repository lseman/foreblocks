"""Transformer encoder, decoder, and shared base implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.forecasting.transformer.core.attention_backends import (
        LAYER_ATTENTION_BACKENDS,
        LayerAttentionBackendSpec,
        build_layer_attention_backend,
    )
    from foreblocks.forecasting.transformer.core.base import (
        BaseTransformer,
        BaseTransformerLayer,
    )
    from foreblocks.forecasting.transformer.core.decoder import (
        TransformerDecoder,
        TransformerDecoderLayer,
    )
    from foreblocks.forecasting.transformer.core.encoder import (
        TransformerEncoder,
        TransformerEncoderLayer,
    )
    from foreblocks.forecasting.transformer.core.mixing import (
        MixingTransformer,
        StackedMixingTransformer,
    )

__all__ = [
    "LAYER_ATTENTION_BACKENDS",
    "BaseTransformer",
    "BaseTransformerLayer",
    "LayerAttentionBackendSpec",
    "MixingTransformer",
    "StackedMixingTransformer",
    "TransformerDecoder",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerEncoderLayer",
    "build_layer_attention_backend",
]

_MODULE_BY_NAME = {
    "LAYER_ATTENTION_BACKENDS": "attention_backends",
    "LayerAttentionBackendSpec": "attention_backends",
    "build_layer_attention_backend": "attention_backends",
    "BaseTransformer": "base",
    "BaseTransformerLayer": "base",
    "TransformerDecoder": "decoder",
    "TransformerDecoderLayer": "decoder",
    "TransformerEncoder": "encoder",
    "TransformerEncoderLayer": "encoder",
    "MixingTransformer": "mixing",
    "StackedMixingTransformer": "mixing",
}


def __getattr__(name: str):
    # Lazy re-export (PEP 562): this package sits in the middle of a
    # transformer <-> attention module import cycle, so eagerly importing
    # submodules here at package-load time deadlocks that cycle. Resolve on
    # first attribute access instead.
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
