"""Stable public contracts for transformer execution and generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from foreblocks.models.transformer.runtime.cache import DecoderCacheManager
    from foreblocks.models.transformer.runtime.decoding.engine import GenerationEngine
    from foreblocks.models.transformer.runtime.outputs import (
        TransformerDecoderOutput,
        TransformerEncoderOutput,
        TransformerGenerationOutput,
        TransformerOutput,
    )
    from foreblocks.models.transformer.runtime.residual_state import (
        AttentionResidualState,
    )
    from foreblocks.models.transformer.runtime.state import (
        AttentionCacheState,
        DecoderLayerState,
        DecoderState,
    )

__all__ = [
    "AttentionCacheState",
    "AttentionResidualState",
    "DecoderCacheManager",
    "DecoderLayerState",
    "DecoderState",
    "GenerationEngine",
    "TransformerDecoderOutput",
    "TransformerEncoderOutput",
    "TransformerGenerationOutput",
    "TransformerOutput",
]

_MODULE_BY_NAME = {
    "AttentionCacheState": "state",
    "AttentionResidualState": "residual_state",
    "DecoderCacheManager": "cache",
    "DecoderLayerState": "state",
    "DecoderState": "state",
    "GenerationEngine": "decoding.engine",
    "TransformerDecoderOutput": "outputs",
    "TransformerEncoderOutput": "outputs",
    "TransformerGenerationOutput": "outputs",
    "TransformerOutput": "outputs",
}


def __getattr__(name: str):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(f"{__name__}.{module_name}")
    return getattr(module, name)
