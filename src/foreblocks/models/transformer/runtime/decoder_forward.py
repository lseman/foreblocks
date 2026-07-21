"""Preparation and output helpers for decoder forward execution."""

from __future__ import annotations

from typing import Any

import torch

from foreblocks.models.transformer.runtime.outputs import TransformerDecoderOutput
from foreblocks.models.transformer.runtime.state import (
    DecoderLayerState,
    DecoderState,
    load_legacy_decoder_state,
)


def resolve_output_options(config, hidden_states, attentions, return_dict):
    return (
        config.output_hidden_states if hidden_states is None else hidden_states,
        config.output_attentions if attentions is None else attentions,
        config.return_dict if return_dict is None else return_dict,
    )


def coerce_decoder_state(
    state: dict[str, Any] | DecoderState | None, *, num_layers: int
) -> DecoderState | None:
    if state is None or isinstance(state, DecoderState):
        return state
    return load_legacy_decoder_state(state, num_layers=num_layers)


def prepare_layer_states(
    state: DecoderState | None, *, num_layers: int
) -> list[DecoderLayerState | None]:
    if state is None:
        return [None] * num_layers
    raw_layers = state.get("layers")
    if raw_layers is None:
        return [DecoderLayerState.from_mapping(None) for _ in range(num_layers)]
    layers = [DecoderLayerState.from_mapping(item) for item in raw_layers]
    if len(layers) != num_layers:
        raise ValueError(
            f"incremental_state['layers'] length {len(layers)} "
            f"must equal num_layers={num_layers}"
        )
    return layers


def build_decoder_output(
    last_hidden_state: torch.Tensor,
    *,
    hidden_states: list[torch.Tensor] | None,
    state: DecoderState | None,
    aux_loss: torch.Tensor,
    router_states: list[object],
    attentions: list[torch.Tensor],
    cross_attentions: list[torch.Tensor],
) -> TransformerDecoderOutput:
    return TransformerDecoderOutput(
        last_hidden_state=last_hidden_state,
        hidden_states=tuple(hidden_states) if hidden_states is not None else None,
        past_key_values=state,
        aux_loss=aux_loss,
        router_states=tuple(router_states) if router_states else None,
        attentions=tuple(attentions) if attentions else None,
        cross_attentions=tuple(cross_attentions) if cross_attentions else None,
    )


__all__ = [
    "build_decoder_output",
    "coerce_decoder_state",
    "prepare_layer_states",
    "resolve_output_options",
]
