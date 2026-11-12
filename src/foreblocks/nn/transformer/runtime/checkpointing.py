"""Stack-level layer invocation and activation-checkpoint replay.

Replay captures the selected attention backend and MoE routing buffers for each
invocation, while keeping differentiable auxiliary losses outside the closure.
This module does not import concrete layers or model stacks.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from foreblocks.nn.transformer.runtime.residual_state import AttentionResidualState


@contextmanager
def _selected_attention(layer, attention_type: str):
    """Replay the original backend even after a shared layer changes routes."""
    previous = layer.layer_attention_type
    layer.set_layer_attention_type(attention_type)
    try:
        yield
    finally:
        layer.set_layer_attention_type(previous)


def _moe_checkpoint_replay(layer):
    """Replay routing buffers as they were before this particular invocation."""
    block = getattr(getattr(layer, "feed_forward", None), "block", None)
    snapshots = []
    if block is not None and hasattr(block, "expert_usage"):
        for module in block.modules():
            for name, value in module.named_buffers(recurse=False):
                if name in {"expert_bias", "expert_usage", "adaptive_k_baseline"}:
                    snapshots.append((module, name, value.detach().clone()))
    initial_step = getattr(block, "_current_step", None)
    first_call = True

    @contextmanager
    def replay():
        nonlocal first_call
        if first_call:
            first_call = False
            yield
            return
        current = [
            (module, name, getattr(module, name)) for module, name, _ in snapshots
        ]
        current_step = getattr(block, "_current_step", None)
        try:
            for module, name, value in snapshots:
                setattr(module, name, value.clone())
            if initial_step is not None:
                block._current_step = initial_step
            yield
        finally:
            for module, name, value in current:
                setattr(module, name, value)
            if current_step is not None:
                block._current_step = current_step

    return replay


class LayerInvokeOwner(Protocol):
    def _record_layer_aux_loss(self, loss: torch.Tensor) -> None: ...

    def _run_with_checkpoint(
        self,
        fn: Callable[..., Any],
        *inputs: torch.Tensor,
        use_checkpoint: bool,
    ) -> Any: ...


@dataclass(frozen=True)
class ModelLayerInvokeStrategy:
    """Invoke layers for a stack, optionally under activation checkpointing.

    Checkpointing re-runs the layer in backward, so it replays the attention
    backend selected at call time and records the layer's auxiliary loss
    outside the checkpointed closure.
    """

    owner: LayerInvokeOwner
    use_checkpoint: bool

    def run_encoder_layer(
        self,
        layer: Any,
        x: torch.Tensor,
        *,
        src_mask: torch.Tensor | None,
        src_key_padding_mask: torch.Tensor | None,
        gate_budget: float | None,
        streams: torch.Tensor | None,
        attention_residual_state: AttentionResidualState | None,
        active_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        common = dict(gate_budget=gate_budget, active_mask=active_mask)
        if self.use_checkpoint:
            attention_type = layer.layer_attention_type
            replay_moe = _moe_checkpoint_replay(layer)

            def checkpointed(value):
                with _selected_attention(layer, attention_type), replay_moe():
                    result, _ = layer(value, src_mask, src_key_padding_mask, **common)
                    return result, layer.aux_loss

            result, aux_loss = self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            )
            self.owner._record_layer_aux_loss(aux_loss)
            return result, streams
        result = layer(
            x,
            src_mask,
            src_key_padding_mask,
            streams=streams,
            attention_residual_state=attention_residual_state,
            **common,
        )
        self.owner._record_layer_aux_loss(layer.aux_loss)
        return result

    def run_decoder_layer(
        self,
        layer: Any,
        x: torch.Tensor,
        *,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        memory_mask: torch.Tensor | None,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        layer_state: Any,
        prev_layer_state: Any,
        gate_budget: float | None,
        streams: torch.Tensor | None,
        mtp_targets: torch.Tensor | None,
        attention_residual_state: AttentionResidualState | None,
        active_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, Any, torch.Tensor | None]:
        args = (
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
        )
        common = dict(
            layer_state=layer_state,
            prev_layer_state=prev_layer_state,
            gate_budget=gate_budget,
            mtp_targets=mtp_targets,
            active_mask=active_mask,
        )
        if self.use_checkpoint:
            attention_type = layer.layer_attention_type
            replay_moe = _moe_checkpoint_replay(layer)

            def checkpointed(value):
                with _selected_attention(layer, attention_type), replay_moe():
                    result, _, _ = layer(value, *args, **common)
                    return result, layer.aux_loss

            result, aux_loss = self.owner._run_with_checkpoint(
                checkpointed, x, use_checkpoint=True
            )
            self.owner._record_layer_aux_loss(aux_loss)
            return result, layer_state, streams
        result = layer(
            x,
            *args,
            streams=streams,
            attention_residual_state=attention_residual_state,
            **common,
        )
        self.owner._record_layer_aux_loss(layer.aux_loss)
        return result


__all__ = ["LayerInvokeOwner", "ModelLayerInvokeStrategy"]
