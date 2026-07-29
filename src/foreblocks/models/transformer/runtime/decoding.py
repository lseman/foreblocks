"""Autoregressive, beam, and speculative transformer decoding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from foreblocks.models.transformer.generation import GenerationConfig
from foreblocks.models.transformer.runtime.cache import DecoderCacheManager
from foreblocks.models.transformer.runtime.contracts import DecoderOwner
from foreblocks.models.transformer.runtime.outputs import TransformerGenerationOutput
from foreblocks.models.transformer.runtime.state import DecoderState


@torch.no_grad()
def beam_search(
    decoder: DecoderOwner,
    cache_manager: DecoderCacheManager,
    initial_tgt: torch.Tensor,
    memory: torch.Tensor,
    max_new_tokens: int,
    num_beams: int,
    proposal_fn: Callable[[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, DecoderState]:
    if num_beams < 1 or max_new_tokens < 1:
        raise ValueError("num_beams and max_new_tokens must be positive")
    batch_size = initial_tgt.size(0)
    prediction, state = decoder.forward_one_step(initial_tgt, memory)
    prediction = prediction[:, -1:, :]
    beam_scores = prediction.new_zeros(batch_size, 1)
    histories = None
    beam_memory = memory
    for step in range(max_new_tokens):
        candidates, candidate_scores = proposal_fn(prediction, step)
        if candidates.ndim != 3 or candidate_scores.shape != candidates.shape[:2]:
            raise ValueError("proposal_fn must return [N,K,C] values and [N,K] scores")
        current_beams, choices = beam_scores.size(1), candidates.size(1)
        total = candidate_scores.view(batch_size, current_beams, choices)
        total = total + beam_scores.unsqueeze(-1)
        keep = min(num_beams, current_beams * choices)
        beam_scores, flat_indices = total.flatten(1).topk(keep, dim=1)
        parent = torch.div(flat_indices, choices, rounding_mode="floor")
        choice = flat_indices.remainder(choices)
        offsets = (
            torch.arange(batch_size, device=parent.device)[:, None] * current_beams
        )
        parent_global = (parent + offsets).flatten()
        rows = candidates.view(batch_size, current_beams, choices, -1)
        batch_rows = torch.arange(batch_size, device=parent.device)[:, None]
        selected = rows[batch_rows, parent, choice].reshape(batch_size * keep, 1, -1)
        if histories is None:
            histories = selected.view(batch_size, keep, 1, -1)
        else:
            histories = torch.cat(
                [
                    histories[batch_rows, parent],
                    selected.view(batch_size, keep, 1, -1),
                ],
                dim=2,
            )
        state = cache_manager.reorder(state, parent_global)
        beam_memory = beam_memory.index_select(0, parent_global.to(beam_memory.device))
        if step + 1 < max_new_tokens:
            prediction, state = decoder.forward_one_step(
                selected, beam_memory, incremental_state=state
            )
            prediction = prediction[:, -1:, :]
    if histories is None:
        raise RuntimeError("beam search produced no histories")
    return histories[:, 0], beam_scores[:, 0], state


def speculative_decode(
    decoder: DecoderOwner,
    draft_tokens: torch.Tensor,
    memory: torch.Tensor,
    state: DecoderState,
    *,
    verifier_fn: Callable[[torch.Tensor, torch.Tensor], int] | None = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, DecoderState, int]:
    caches = [layer.self_attention.cache for layer in state.layers]
    start_lengths = [
        cache.get_seq_length() if cache is not None else None for cache in caches
    ]
    output, state = decoder.forward_multi_step(draft_tokens, memory, state, **kwargs)
    accepted = (
        int(verifier_fn(output, draft_tokens))
        if verifier_fn is not None
        else draft_tokens.size(1)
    )
    accepted = max(0, min(accepted, draft_tokens.size(1)))
    if accepted != draft_tokens.size(1):
        for start, layer in zip(start_lengths, state.layers, strict=True):
            cache = layer.self_attention.cache
            if start is not None and cache is not None:
                cache.crop(start + accepted)
    return output[:, :accepted], state, accepted


class GenerationEngine:
    def __init__(self, decoder: DecoderOwner, cache: DecoderCacheManager) -> None:
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


__all__ = ["GenerationEngine", "beam_search", "speculative_decode"]
