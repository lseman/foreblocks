"""Attention-residual state transitions shared by encoder and decoder stacks."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class AttentionResidualState:
    mode: str
    current: torch.Tensor
    history: list[torch.Tensor] = field(default_factory=list)
    running_sum: torch.Tensor | None = None
    blocks: list[torch.Tensor] = field(default_factory=list)
    partial: torch.Tensor | None = None
    block_size: int = 1
    sublayers_in_block: int = 0


def init_attention_residual_state(
    x: torch.Tensor, mode: str, block_size: int
) -> AttentionResidualState:
    if mode == "full":
        return AttentionResidualState(mode=mode, current=x, history=[x], running_sum=x)
    return AttentionResidualState(
        mode=mode,
        current=x,
        blocks=[x],
        block_size=int(block_size),
    )


def attention_residual_input(
    carrier: torch.Tensor,
    state: AttentionResidualState | None,
    module: nn.Module | None,
) -> torch.Tensor:
    if state is None or module is None:
        return carrier
    if state.mode == "full":
        return module(state.history)
    return module(state.blocks, state.partial)


def append_attention_residual_update(
    state: AttentionResidualState | None,
    update: torch.Tensor,
) -> None:
    if state is None:
        return
    if state.mode == "full":
        state.history.append(update)
        if state.running_sum is None:
            raise RuntimeError("full attention residual state has no running sum")
        state.running_sum = state.running_sum + update
        state.current = state.running_sum
        return
    partial = update if state.partial is None else state.partial + update
    state.partial = partial
    state.sublayers_in_block += 1
    if state.sublayers_in_block >= state.block_size:
        state.blocks.append(partial)
        state.partial = None
        state.sublayers_in_block = 0
    state.current = partial


def attention_residual_values(state: AttentionResidualState) -> list[torch.Tensor]:
    if state.mode == "full":
        return list(state.history)
    values = list(state.blocks)
    if state.partial is not None:
        values.append(state.partial)
    return values


__all__ = [
    "AttentionResidualState",
    "append_attention_residual_update",
    "attention_residual_input",
    "attention_residual_values",
    "init_attention_residual_state",
]
