"""
Mixed encoder/decoder blocks - re-exports from split modules.

This module re-exports all mixed encoder/decoder classes and functions
from their dedicated modules for backward compatibility.
"""

from __future__ import annotations

from .converter import ArchitectureConverter
from .fixed_encoder_decoder import FixedDecoder, FixedEncoder
from .freeze_utils import (
    _freeze_transformer_cross_attention,
    _freeze_transformer_cross_attention_position,
    _freeze_transformer_decoder_style,
    _freeze_transformer_ffn_mode,
    _freeze_transformer_patch_mode,
    _freeze_transformer_self_attention,
    _freeze_transformer_self_attention_position,
    _resolve_searchable_cross_attention_position,
    _resolve_searchable_cross_attention_type,
    _resolve_searchable_decoder_style,
    _resolve_searchable_ffn_mode,
    _resolve_searchable_patch_mode,
    _resolve_searchable_self_attention_position,
    _resolve_searchable_self_attention_type,
)
from .helpers import (
    _collect_layer_components,
    _layer_component,
    _mean_component_mode_probs,
)
from .mixed_encoder_decoder import MixedDecoder, MixedEncoder


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
    "_layer_component",
    "_collect_layer_components",
    "_mean_component_mode_probs",
    "_resolve_searchable_self_attention_type",
    "_freeze_transformer_self_attention",
    "_resolve_searchable_self_attention_position",
    "_freeze_transformer_self_attention_position",
    "_resolve_searchable_cross_attention_type",
    "_freeze_transformer_cross_attention",
    "_resolve_searchable_cross_attention_position",
    "_freeze_transformer_cross_attention_position",
    "_resolve_searchable_ffn_mode",
    "_freeze_transformer_ffn_mode",
    "_resolve_searchable_patch_mode",
    "_freeze_transformer_patch_mode",
    "_resolve_searchable_decoder_style",
    "_freeze_transformer_decoder_style",
]
