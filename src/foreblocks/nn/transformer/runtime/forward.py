"""Encoder and decoder forward-execution preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch
import torch.nn as nn

from foreblocks.nn.attention.cache.kv import StaticKVCache
from foreblocks.nn.embeddings.patching import (
    PatchInfo,
    PatchTokenizer,
    patchify_padding_mask,
)
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.transformer.runtime.execution import ModelLayerInvokeStrategy
from foreblocks.nn.transformer.runtime.outputs import TransformerDecoderOutput
from foreblocks.nn.transformer.runtime.residual_state import AttentionResidualState
from foreblocks.nn.transformer.runtime.routing import patchify_gateskip_active_mask
from foreblocks.nn.transformer.runtime.state import DecoderLayerState, DecoderState


@dataclass(frozen=True)
class PreparedDecoderState:
    state: DecoderState | None
    layer_states: list[DecoderLayerState | None]
    cache_position: torch.Tensor | None
    cache_update_mask: torch.Tensor | None
    position_offset: torch.Tensor | int | None


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


def positions_from_offset(
    offset: torch.Tensor | int,
    *,
    batch_size: int,
    length: int,
    device: torch.device,
) -> torch.Tensor:
    """Absolute ``[B, T]`` positions for a chunk starting at ``offset``."""
    steps = torch.arange(length, device=device, dtype=torch.long)
    if isinstance(offset, int):
        return (steps + offset).unsqueeze(0).expand(batch_size, -1)
    start = offset.to(device=device, dtype=torch.long)
    if start.dim() == 0:
        start = start.view(1).expand(batch_size)
    if start.dim() != 1 or start.shape[0] != batch_size:
        raise ValueError(
            f"position_offset tensor must be scalar or [B], got {tuple(start.shape)}"
        )
    return start.unsqueeze(1) + steps.unsqueeze(0)


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


class EncoderPreparationOwner(Protocol):
    config: TransformerConfig
    input_adapter: nn.Module
    patcher: PatchTokenizer
    missing_token: torch.Tensor | None

    def _channel_patchify(
        self, src: torch.Tensor
    ) -> tuple[torch.Tensor, PatchInfo]: ...


@dataclass(frozen=True)
class PreparedEncoderInput:
    hidden_states: torch.Tensor
    padding_mask: torch.Tensor | None
    active_mask: torch.Tensor | None
    patch_info: PatchInfo | None


def prepare_encoder_input(
    owner: EncoderPreparationOwner,
    src: torch.Tensor,
    padding_mask: torch.Tensor | None,
    active_mask: torch.Tensor | None,
    value_mask: torch.Tensor | None = None,
) -> PreparedEncoderInput:
    """Project or patch encoder input and transform its token-aligned masks."""
    if src.ndim != 3:
        raise ValueError(f"encoder expects src [B,T,C], got {tuple(src.shape)}")
    _, sequence_length, input_size = src.shape
    config = owner.config
    if config.input_size != input_size:
        raise ValueError(f"Expected input size {config.input_size}, got {input_size}")
    if sequence_length > config.max_seq_len and config.patching == "none":
        raise ValueError(
            f"Sequence length {sequence_length} exceeds max {config.max_seq_len}"
        )

    if config.variate_attention:
        batch_size, _, num_variates = src.shape
        value_mask_bvt = None
        if value_mask is not None:
            if value_mask.shape == (batch_size, sequence_length):
                value_mask_bvt = value_mask[:, None, :].expand(-1, num_variates, -1)
            elif value_mask.shape == (batch_size, num_variates, sequence_length):
                value_mask_bvt = value_mask
            elif value_mask.shape == (batch_size, sequence_length, num_variates):
                value_mask_bvt = value_mask.transpose(1, 2)
            else:
                raise ValueError(
                    "value_mask must be [B,T], [B,V,T], or [B,T,V], got "
                    f"{tuple(value_mask.shape)}"
                )
            value_mask_bvt = value_mask_bvt.to(device=src.device, dtype=torch.bool)
            src = src.masked_fill(value_mask_bvt.transpose(1, 2), 0.0)

        hidden_states = src.transpose(1, 2).reshape(
            batch_size * num_variates, sequence_length, 1
        )
        hidden_states = owner.input_adapter(hidden_states)
        if value_mask_bvt is not None:
            if owner.missing_token is None:
                raise RuntimeError("variate value masking requires a missing token")
            flat_value_mask = value_mask_bvt.reshape(
                batch_size * num_variates, sequence_length, 1
            )
            hidden_states = (
                hidden_states
                + flat_value_mask.to(dtype=hidden_states.dtype) * owner.missing_token
            )

        def expand_variate_mask(mask: torch.Tensor | None, name: str):
            if mask is None:
                return None
            if mask.shape == (batch_size, sequence_length):
                mask = mask[:, None, :].expand(-1, num_variates, -1)
            elif mask.shape == (batch_size, sequence_length, num_variates):
                mask = mask.transpose(1, 2)
            elif mask.shape != (batch_size, num_variates, sequence_length):
                raise ValueError(
                    f"{name} must be [B,T], [B,V,T], or [B,T,V], "
                    f"got {tuple(mask.shape)}"
                )
            return mask.reshape(batch_size * num_variates, sequence_length)

        padding_mask = expand_variate_mask(padding_mask, "src_key_padding_mask")
        active_mask = expand_variate_mask(active_mask, "gateskip_active_mask")
        patch_info = None
        if config.patching == "shared":
            hidden_states, patch_info = owner.patcher(hidden_states)
            padding_mask, active_mask = _patchify_masks(
                config, padding_mask, active_mask, sequence_length
            )

        token_length = hidden_states.shape[1]
        if token_length > config.max_seq_len:
            raise ValueError(
                f"Encoder token length {token_length} exceeds "
                f"max_seq_len={config.max_seq_len}"
            )
        hidden_states = hidden_states.reshape(
            batch_size, num_variates, token_length, -1
        )
        if padding_mask is not None:
            padding_mask = padding_mask.reshape(batch_size, num_variates, token_length)
        if active_mask is not None:
            active_mask = active_mask.reshape(batch_size, num_variates, token_length)
        return PreparedEncoderInput(
            hidden_states, padding_mask, active_mask, patch_info
        )

    if value_mask is not None:
        raise ValueError("value_mask requires variate_attention=True")
    if config.patching == "channel":
        hidden_states, patch_info = owner._channel_patchify(src)
    else:
        hidden_states = owner.input_adapter(src)
        if config.patching == "none":
            return PreparedEncoderInput(hidden_states, padding_mask, active_mask, None)
        hidden_states, patch_info = owner.patcher(hidden_states)

    if hidden_states.shape[1] > config.max_seq_len:
        raise ValueError(
            f"Encoder patch token length {hidden_states.shape[1]} exceeds "
            f"max_seq_len={config.max_seq_len}. Increase max_seq_len or adjust "
            "patch_len/patch_stride."
        )
    padding_mask, active_mask = _patchify_masks(
        config, padding_mask, active_mask, sequence_length
    )
    return PreparedEncoderInput(hidden_states, padding_mask, active_mask, patch_info)


def _patchify_masks(
    config: TransformerConfig,
    padding_mask: torch.Tensor | None,
    active_mask: torch.Tensor | None,
    sequence_length: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    geometry = dict(
        T=sequence_length,
        patch_len=config.patch_len,
        stride=config.patch_stride,
        pad_end=config.patch_pad_end,
    )
    return (
        patchify_padding_mask(padding_mask, **geometry),
        patchify_gateskip_active_mask(active_mask, **geometry),
    )


__all__ = [
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
