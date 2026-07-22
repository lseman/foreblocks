"""Preparation, layer invocation, and output helpers for decoder forward execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from foreblocks.models.transformer.runtime.execution import ModelLayerInvokeStrategy
from foreblocks.models.transformer.runtime.outputs import TransformerDecoderOutput
from foreblocks.models.transformer.runtime.residual_state import AttentionResidualState
from foreblocks.models.transformer.runtime.state import DecoderLayerState, DecoderState
from foreblocks.modules.attention.cache.kv import StaticKVCache


@dataclass(frozen=True)
class PreparedDecoderState:
    state: DecoderState | None
    layer_states: list[DecoderLayerState | None]
    cache_position: torch.Tensor | None
    cache_update_mask: torch.Tensor | None
    position_offset: torch.Tensor | int | None


class DecoderStackOwner(Protocol):
    def _resolve_layer(self, layer_index: int) -> Any: ...


@dataclass(frozen=True)
class DecoderLayerResult:
    hidden_states: torch.Tensor
    layer_state: DecoderLayerState | None
    streams: torch.Tensor | None
    router_state: object | None
    self_attention: torch.Tensor | None
    cross_attention: torch.Tensor | None


def execute_decoder_layer(
    owner: DecoderStackOwner,
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
) -> DecoderLayerResult:
    """Execute one ordinary decoder layer and collect its runtime diagnostics."""
    layer = owner._resolve_layer(layer_index)
    self_attention_module = layer._self_attn()
    cross_attention_module = layer.cross_attn
    if hasattr(self_attention_module, "output_attentions"):
        self_attention_module.output_attentions = output_attentions
    cross_attention_module.output_attentions = output_attentions
    hidden_states, layer_state, streams = invoke.run_decoder_layer(
        layer=layer,
        x=hidden_states,
        memory=memory,
        tgt_mask=tgt_mask,
        memory_mask=memory_mask,
        tgt_key_padding_mask=tgt_key_padding_mask,
        memory_key_padding_mask=memory_key_padding_mask,
        layer_state=layer_state,
        prev_state=previous_state,
        budget=budget,
        streams=streams,
        mtp_targets=mtp_targets,
        attention_residual_state=attention_residual_state,
        gateskip_active_mask=active_mask,
    )
    ff_block = getattr(getattr(layer, "feed_forward", None), "block", None)
    return DecoderLayerResult(
        hidden_states=hidden_states,
        layer_state=layer_state,
        streams=streams,
        router_state=getattr(ff_block, "last_routing_state", None),
        self_attention=getattr(self_attention_module, "last_attn_weights", None),
        cross_attention=getattr(cross_attention_module, "last_attn_weights", None),
    )


def prepare_decoder_state(
    state: dict[str, Any] | DecoderState | None,
    *,
    num_layers: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    cache_position: torch.Tensor | None,
    cache_update_mask: torch.Tensor | None,
    position_offset: torch.Tensor | int | None,
) -> PreparedDecoderState:
    """Normalize incremental state and attach per-call cache coordinates."""
    state = DecoderState.coerce(state, num_layers=num_layers)
    if cache_position is None and state is not None and state.layers:
        cache = state.layers[0].self_attention.cache
        if isinstance(cache, StaticKVCache):
            cache_position = cache.lengths[:, None] + torch.arange(
                sequence_length, device=device, dtype=torch.long
            )

    if cache_position is not None:
        cache_position = cache_position.to(device=device, dtype=torch.long)
        if cache_position.ndim == 1:
            cache_position = cache_position.unsqueeze(0).expand(batch_size, -1)
        if cache_position.shape != (batch_size, sequence_length):
            raise ValueError(
                "cache_position must be [T] or [B,T], got "
                f"{tuple(cache_position.shape)}"
            )
        if position_offset is None:
            position_offset = cache_position[:, 0]

    if cache_update_mask is not None:
        cache_update_mask = cache_update_mask.to(device=device, dtype=torch.bool)
        if cache_update_mask.shape != (batch_size,):
            raise ValueError("cache_update_mask must have shape [B]")

    layer_states: list[DecoderLayerState | None] = (
        list(state.layers) if state is not None else [None] * num_layers
    )
    if cache_position is not None:
        for layer_state in layer_states:
            if layer_state is None:
                continue
            self_state = layer_state.self_attention
            self_state.cache_position = cache_position
            self_state.cache_update_mask = cache_update_mask

    return PreparedDecoderState(
        state=state,
        layer_states=layer_states,
        cache_position=cache_position,
        cache_update_mask=cache_update_mask,
        position_offset=position_offset,
    )


def validate_memory_padding_mask(
    memory: torch.Tensor, mask: torch.Tensor | None
) -> None:
    if mask is not None and mask.shape[:2] != memory.shape[:2]:
        raise ValueError(
            f"memory_key_padding_mask shape {tuple(mask.shape)} must match "
            f"memory [B,Tm]=[{memory.shape[0]},{memory.shape[1]}]"
        )


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
    "DecoderLayerResult",
    "DecoderStackOwner",
    "execute_decoder_layer",
    "prepare_decoder_state",
    "PreparedDecoderState",
    "validate_memory_padding_mask",
]
