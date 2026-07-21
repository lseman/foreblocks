"""Attention-residual state transitions shared by encoder and decoder stacks."""

from __future__ import annotations


import torch
import torch.nn as nn


def init_attention_residual_state(x: torch.Tensor, mode: str, block_size: int) -> dict:
    state = {"mode": mode, "current": x}
    if mode == "full":
        state.update(history=[x], running_sum=x)
    else:
        state.update(
            blocks=[x], partial=None, block_size=int(block_size), sub_layers_in_block=0
        )
    return state


def attention_residual_input(
    carrier: torch.Tensor, state: dict | None, module: nn.Module | None
) -> torch.Tensor:
    if state is None or module is None:
        return carrier
    if state["mode"] == "full":
        return module(state["history"])
    return module(state["blocks"], state["partial"])


def append_attention_residual_update(
    state: dict | None,
    update: torch.Tensor,
) -> None:
    if state is None:
        return
    if state["mode"] == "full":
        state["history"].append(update)
        state["running_sum"] = state["running_sum"] + update
        state["current"] = state["running_sum"]
        return
    partial = update if state["partial"] is None else state["partial"] + update
    state["partial"] = partial
    state["sub_layers_in_block"] += 1
    if state["sub_layers_in_block"] >= state["block_size"]:
        state["blocks"].append(partial)
        state["partial"] = None
        state["sub_layers_in_block"] = 0
    state["current"] = partial


def attention_residual_values(state: dict) -> list[torch.Tensor]:
    if state["mode"] == "full":
        return list(state["history"])
    values = list(state["blocks"])
    if state["partial"] is not None:
        values.append(state["partial"])
    return values


# Private compatibility names used by the existing stack.
_init_attention_residual_state = init_attention_residual_state
_attention_residual_input = attention_residual_input
_append_attention_residual_update = append_attention_residual_update
_attention_residual_values = attention_residual_values

__all__ = [
    "append_attention_residual_update",
    "attention_residual_input",
    "attention_residual_values",
    "init_attention_residual_state",
]
