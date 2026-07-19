"""Structured outputs for Foreblocks transformer stacks."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TransformerEncoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    aux_loss: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    router_states: tuple[object, ...] | None = None


@dataclass
class TransformerDecoderOutput:
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    past_key_values: dict | None = None
    aux_loss: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    router_states: tuple[object, ...] | None = None


@dataclass
class TransformerGenerationOutput:
    sequences: torch.Tensor
    past_key_values: dict | None = None


__all__ = [
    "TransformerDecoderOutput",
    "TransformerEncoderOutput",
    "TransformerGenerationOutput",
]
