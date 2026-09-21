"""Transformer configuration, encoders, decoders, mixing, and generation outputs."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.nn.transformer.config import TransformerConfig as TransformerConfig
    from foreblocks.nn.transformer.config import AttentionMode as AttentionMode
    from foreblocks.nn.transformer.config import GenerationConfig as GenerationConfig
    from foreblocks.nn.transformer.config import ResidualConfig as ResidualConfig
    from foreblocks.nn.transformer.config import CacheConfig as CacheConfig
    from foreblocks.nn.transformer.encoder import TransformerEncoder as TransformerEncoder
    from foreblocks.nn.transformer.encoder import TransformerEncoderLayer as TransformerEncoderLayer
    from foreblocks.nn.transformer.decoder import TransformerDecoder as TransformerDecoder
    from foreblocks.nn.transformer.decoder import TransformerDecoderLayer as TransformerDecoderLayer
    from foreblocks.nn.transformer.base import BaseTransformer as BaseTransformer
    from foreblocks.nn.transformer.base import BaseTransformerLayer as BaseTransformerLayer
    from foreblocks.nn.transformer.mixing import MixingTransformer as MixingTransformer
    from foreblocks.nn.transformer.mixing import StackedMixingTransformer as StackedMixingTransformer
    from foreblocks.nn.transformer.runtime.outputs import TransformerEncoderOutput as TransformerEncoderOutput
    from foreblocks.nn.transformer.runtime.outputs import TransformerDecoderOutput as TransformerDecoderOutput
    from foreblocks.nn.transformer.runtime.outputs import TransformerGenerationOutput as TransformerGenerationOutput
    from foreblocks.nn.transformer.tuner import TransformerTuner as TransformerTuner

_EXPORTS = {
    'TransformerConfig': 'foreblocks.nn.transformer.config',
    'AttentionMode': 'foreblocks.nn.transformer.config',
    'GenerationConfig': 'foreblocks.nn.transformer.config',
    'ResidualConfig': 'foreblocks.nn.transformer.config',
    'CacheConfig': 'foreblocks.nn.transformer.config',
    'TransformerEncoder': 'foreblocks.nn.transformer.encoder',
    'TransformerEncoderLayer': 'foreblocks.nn.transformer.encoder',
    'TransformerDecoder': 'foreblocks.nn.transformer.decoder',
    'TransformerDecoderLayer': 'foreblocks.nn.transformer.decoder',
    'BaseTransformer': 'foreblocks.nn.transformer.base',
    'BaseTransformerLayer': 'foreblocks.nn.transformer.base',
    'MixingTransformer': 'foreblocks.nn.transformer.mixing',
    'StackedMixingTransformer': 'foreblocks.nn.transformer.mixing',
    'TransformerEncoderOutput': 'foreblocks.nn.transformer.runtime.outputs',
    'TransformerDecoderOutput': 'foreblocks.nn.transformer.runtime.outputs',
    'TransformerGenerationOutput': 'foreblocks.nn.transformer.runtime.outputs',
    'TransformerTuner': 'foreblocks.nn.transformer.tuner',
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
