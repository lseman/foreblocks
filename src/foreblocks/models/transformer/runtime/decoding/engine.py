"""Greedy generation facade and decoder compilation entry points."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from foreblocks.models.transformer.generation import GenerationConfig
from foreblocks.models.transformer.runtime.cache import DecoderCacheManager
from foreblocks.models.transformer.runtime.contracts import DecoderProtocol
from foreblocks.models.transformer.runtime.decoding.beam import beam_search
from foreblocks.models.transformer.runtime.decoding.speculative import (
    speculative_decode,
)
from foreblocks.models.transformer.runtime.outputs import TransformerGenerationOutput
from foreblocks.models.transformer.runtime.state import DecoderState


class GenerationEngine:
    def __init__(self, decoder: DecoderProtocol, cache: DecoderCacheManager) -> None:
        self.decoder = decoder
        self.cache = cache

    def speculative_decode(
        self,
        draft_tokens: torch.Tensor,
        memory: torch.Tensor,
        incremental_state: DecoderState,
        *,
        verifier_fn: Callable[[torch.Tensor, torch.Tensor], int] | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, DecoderState, int]:
        return speculative_decode(
            self.decoder,
            draft_tokens,
            memory,
            incremental_state,
            verifier_fn=verifier_fn,
            **kwargs,
        )

    def compile_prefill(self, **options: Any):
        return torch.compile(self.decoder.prefill, **options)

    def compile_decode(self, **options: Any):
        return torch.compile(self.decoder.decode, **options)

    @torch.no_grad()
    def generate(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int | None = None,
        *,
        generation_config: GenerationConfig | None = None,
        incremental_state: DecoderState | None = None,
        feedback_fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None,
        memory_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> torch.Tensor | TransformerGenerationOutput:
        if generation_config is None:
            generation_config = GenerationConfig(
                max_new_tokens=1 if max_new_tokens is None else max_new_tokens
            )
        elif max_new_tokens is not None:
            raise ValueError(
                "max_new_tokens belongs to GenerationConfig; do not pass both"
            )
        max_new_tokens = generation_config.max_new_tokens
        return_dict = (
            generation_config.return_dict if return_dict is None else return_dict
        )
        if feedback_fn is None and self.decoder.output_size != initial_tgt.size(-1):
            raise ValueError(
                "output_size must match decoder input width unless feedback_fn is provided"
            )
        state = incremental_state
        if max_new_tokens == 0:
            sequences = initial_tgt.new_empty(
                initial_tgt.size(0), 0, self.decoder.output_size
            )
        else:
            output, state = self.decoder.forward_one_step(
                initial_tgt,
                memory,
                incremental_state=state,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            generated = []
            for step in range(max_new_tokens):
                prediction = output[:, -1:, :]
                generated.append(prediction)
                if step + 1 == max_new_tokens:
                    break
                next_input = (
                    feedback_fn(prediction, step) if feedback_fn else prediction
                )
                output, state = self.decoder.forward_one_step(
                    next_input,
                    memory,
                    incremental_state=state,
                    memory_mask=memory_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
            sequences = torch.cat(generated, dim=1)
        return (
            TransformerGenerationOutput(sequences, state) if return_dict else sequences
        )

    def beam_search(
        self,
        initial_tgt: torch.Tensor,
        memory: torch.Tensor,
        max_new_tokens: int,
        num_beams: int,
        proposal_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor, DecoderState]:
        return beam_search(
            self.decoder,
            self.cache,
            initial_tgt,
            memory,
            max_new_tokens,
            num_beams,
            proposal_fn,
        )


__all__ = ["GenerationEngine"]
