"""Structured outputs for Foreblocks transformer stacks."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, fields
from typing import Any, overload

import torch

from foreblocks.models.transformer.runtime.state import DecoderState


class TransformerOutput:
    def to_tuple(self) -> tuple[object, ...]:
        return tuple(
            value
            for item in fields(self)  # pyright: ignore[reportArgumentType]
            if (value := getattr(self, item.name)) is not None
        )

    def __iter__(self) -> Iterator[object]:
        return iter(self.to_tuple())

    @overload
    def __getitem__(self, key: str) -> Any: ...

    @overload
    def __getitem__(self, key: int | slice) -> object: ...

    def __getitem__(self, key: str | int | slice) -> Any:
        if isinstance(key, str):
            return getattr(self, key)
        return self.to_tuple()[key]


@dataclass
class TransformerEncoderOutput(TransformerOutput):
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    aux_loss: torch.Tensor | None = None
    padding_mask: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    router_states: tuple[object, ...] | None = None


@dataclass
class TransformerDecoderOutput(TransformerOutput):
    last_hidden_state: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None
    past_key_values: DecoderState | None = None
    aux_loss: torch.Tensor | None = None
    attentions: tuple[torch.Tensor, ...] | None = None
    cross_attentions: tuple[torch.Tensor, ...] | None = None
    router_states: tuple[object, ...] | None = None


@dataclass
class TransformerGenerationOutput(TransformerOutput):
    sequences: torch.Tensor
    past_key_values: DecoderState | None = None


def resolve_output_options(config, hidden_states, attentions, return_dict):
    return (
        config.output_hidden_states if hidden_states is None else hidden_states,
        config.output_attentions if attentions is None else attentions,
        config.return_dict if return_dict is None else return_dict,
    )


__all__ = [
    "TransformerDecoderOutput",
    "TransformerEncoderOutput",
    "TransformerGenerationOutput",
    "TransformerOutput",
    "resolve_output_options",
]
