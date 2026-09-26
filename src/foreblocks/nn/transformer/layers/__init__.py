"""Individual transformer layers, loaded without importing model stacks."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseTransformerLayer as BaseTransformerLayer
    from .decoder import TransformerDecoderLayer as TransformerDecoderLayer
    from .encoder import TransformerEncoderLayer as TransformerEncoderLayer
    from .mixing import (
        MixingTransformer as MixingTransformer,
        StackedMixingTransformer as StackedMixingTransformer,
    )

_EXPORTS = {
    "BaseTransformerLayer": "base",
    "TransformerEncoderLayer": "encoder",
    "TransformerDecoderLayer": "decoder",
    "MixingTransformer": "mixing",
    "StackedMixingTransformer": "mixing",
}
__all__ = list(_EXPORTS)


def __getattr__(name: str):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{module_name}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
