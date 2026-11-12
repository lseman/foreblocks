"""Stack-level layer execution and collection of runtime diagnostics.

Preparation and output helpers are re-exported for existing callers; their
implementations live in preparation.py and outputs.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from foreblocks.nn.transformer.runtime.checkpointing import ModelLayerInvokeStrategy
from foreblocks.nn.transformer.runtime.outputs import (
    build_decoder_output as build_decoder_output,
)
from foreblocks.nn.transformer.runtime.preparation import (
    EncoderPreparationOwner as EncoderPreparationOwner,
    PreparedDecoderState as PreparedDecoderState,
    PreparedEncoderInput as PreparedEncoderInput,
    positions_from_offset as positions_from_offset,
    prepare_decoder_state as prepare_decoder_state,
    prepare_encoder_input as prepare_encoder_input,
    validate_memory_padding_mask as validate_memory_padding_mask,
)
from foreblocks.nn.transformer.runtime.residual_state import AttentionResidualState
from foreblocks.nn.transformer.runtime.state import DecoderLayerState


class StackOwner(Protocol):
    def _resolve_layer(self, layer_index: int) -> Any: ...


@dataclass(frozen=True)
class LayerResult:
    """One layer invocation's output state and runtime diagnostics."""

    hidden_states: torch.Tensor
    streams: torch.Tensor | None = None
    layer_state: DecoderLayerState | None = None
    router_state: object | None = None
    self_attention: torch.Tensor | None = None
    cross_attention: torch.Tensor | None = None


@dataclass
class StackTrace:
    """Diagnostics collected while one forward pass runs the layer stack."""

    hidden_states: list[torch.Tensor] | None
    used_indices: list[int] = field(default_factory=list)
    router_states: list[object] = field(default_factory=list)
    attentions: list[torch.Tensor] = field(default_factory=list)
    cross_attentions: list[torch.Tensor] = field(default_factory=list)

    @classmethod
    def start(cls, x: torch.Tensor, output_hidden_states: bool) -> StackTrace:
        return cls(hidden_states=[x] if output_hidden_states else None)

    def record(self, index: int, result: LayerResult) -> None:
        self.used_indices.append(index)
        if self.hidden_states is not None:
            self.hidden_states.append(result.hidden_states)
        if result.router_state is not None:
            self.router_states.append(result.router_state)
        if result.self_attention is not None:
            self.attentions.append(result.self_attention)
        if result.cross_attention is not None:
            self.cross_attentions.append(result.cross_attention)


def _router_state(layer: Any) -> object | None:
    ff_block = getattr(getattr(layer, "feed_forward", None), "block", None)
    return getattr(ff_block, "last_routing_state", None)


def execute_encoder_layer(
    owner: StackOwner,
    invoke: ModelLayerInvokeStrategy,
    *,
    layer_index: int,
    hidden_states: torch.Tensor,
    src_mask: torch.Tensor | None,
    src_key_padding_mask: torch.Tensor | None,
    budget: float | None,
    streams: torch.Tensor | None,
    attention_residual_state: AttentionResidualState | None,
    active_mask: torch.Tensor | None,
    output_attentions: bool,
) -> LayerResult:
    """Execute one ordinary encoder layer and collect its runtime diagnostics."""
    layer = owner._resolve_layer(layer_index)
    attention_module = layer._self_attn()
    if hasattr(attention_module, "output_attentions"):
        attention_module.output_attentions = output_attentions
    hidden_states, streams = invoke.run_encoder_layer(
        layer,
        hidden_states,
        src_mask=src_mask,
        src_key_padding_mask=src_key_padding_mask,
        gate_budget=budget,
        streams=streams,
        attention_residual_state=attention_residual_state,
        active_mask=active_mask,
    )
    return LayerResult(
        hidden_states=hidden_states,
        streams=streams,
        router_state=_router_state(layer),
        self_attention=getattr(attention_module, "last_attn_weights", None),
    )


def execute_decoder_layer(
    owner: StackOwner,
    invoke: ModelLayerInvokeStrategy,
    *,
    layer_index: int,
    hidden_states: torch.Tensor,
    memory: torch.Tensor,
    tgt_mask: torch.Tensor | None,
    memory_mask: torch.Tensor | None,
    tgt_key_padding_mask: torch.Tensor | None,
    memory_key_padding_mask: torch.Tensor | None,
    layer_state: DecoderLayerState | None,
    previous_state: DecoderLayerState | None,
    budget: float | None,
    streams: torch.Tensor | None,
    mtp_targets: torch.Tensor | None,
    attention_residual_state: AttentionResidualState | None,
    active_mask: torch.Tensor | None,
    output_attentions: bool,
) -> LayerResult:
    """Execute one ordinary decoder layer and collect its runtime diagnostics."""
    layer = owner._resolve_layer(layer_index)
    self_attention_module = layer._self_attn()
    cross_attention_module = layer.cross_attn
    if hasattr(self_attention_module, "output_attentions"):
        self_attention_module.output_attentions = output_attentions
    cross_attention_module.output_attentions = output_attentions
    hidden_states, layer_state, streams = invoke.run_decoder_layer(
        layer,
        hidden_states,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        layer_state=layer_state,
        prev_layer_state=previous_state,
        gate_budget=budget,
        streams=streams,
        mtp_targets=mtp_targets,
        attention_residual_state=attention_residual_state,
        active_mask=active_mask,
    )
    return LayerResult(
        hidden_states=hidden_states,
        layer_state=layer_state,
        streams=streams,
        router_state=_router_state(layer),
        self_attention=getattr(self_attention_module, "last_attn_weights", None),
        cross_attention=getattr(cross_attention_module, "last_attn_weights", None),
    )


__all__ = [
    "StackTrace",
    "LayerResult",
    "StackOwner",
    "EncoderPreparationOwner",
    "PreparedDecoderState",
    "PreparedEncoderInput",
    "build_decoder_output",
    "execute_decoder_layer",
    "execute_encoder_layer",
    "positions_from_offset",
    "prepare_decoder_state",
    "prepare_encoder_input",
    "validate_memory_padding_mask",
]
