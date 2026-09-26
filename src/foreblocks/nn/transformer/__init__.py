"""Transformer configuration, encoders, decoders, layers, and outputs."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.transformer.base import BaseTransformer as BaseTransformer
    from foreblocks.nn.transformer.config import (
        GenerationConfig as GenerationConfig,
        TransformerConfig as TransformerConfig,
    )
    from foreblocks.nn.transformer.decoder import (
        TransformerDecoder as TransformerDecoder,
    )
    from foreblocks.nn.transformer.encoder import (
        TransformerEncoder as TransformerEncoder,
    )
    from foreblocks.nn.transformer.layers.base import (
        BaseTransformerLayer as BaseTransformerLayer,
    )
    from foreblocks.nn.transformer.layers.decoder import (
        TransformerDecoderLayer as TransformerDecoderLayer,
    )
    from foreblocks.nn.transformer.layers.encoder import (
        TransformerEncoderLayer as TransformerEncoderLayer,
    )
    from foreblocks.nn.transformer.layers.mixing import (
        MixingTransformer as MixingTransformer,
        StackedMixingTransformer as StackedMixingTransformer,
    )
    from foreblocks.nn.transformer.runtime.outputs import (
        TransformerDecoderOutput as TransformerDecoderOutput,
        TransformerEncoderOutput as TransformerEncoderOutput,
        TransformerGenerationOutput as TransformerGenerationOutput,
    )

_EXPORTS = {
    "TransformerConfig": "foreblocks.nn.transformer.config",
    "GenerationConfig": "foreblocks.nn.transformer.config",
    "TransformerEncoder": "foreblocks.nn.transformer.encoder",
    "TransformerDecoder": "foreblocks.nn.transformer.decoder",
    "BaseTransformer": "foreblocks.nn.transformer.base",
    "TransformerEncoderLayer": "foreblocks.nn.transformer.layers.encoder",
    "TransformerDecoderLayer": "foreblocks.nn.transformer.layers.decoder",
    "BaseTransformerLayer": "foreblocks.nn.transformer.layers.base",
    "MixingTransformer": "foreblocks.nn.transformer.layers.mixing",
    "StackedMixingTransformer": "foreblocks.nn.transformer.layers.mixing",
    "TransformerEncoderOutput": "foreblocks.nn.transformer.runtime.outputs",
    "TransformerDecoderOutput": "foreblocks.nn.transformer.runtime.outputs",
    "TransformerGenerationOutput": "foreblocks.nn.transformer.runtime.outputs",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
