"""Beam-search decoding over typed decoder state."""

from __future__ import annotations

from collections.abc import Callable

import torch

from foreblocks.models.transformer.runtime.cache import DecoderCacheManager
from foreblocks.models.transformer.runtime.contracts import DecoderProtocol
from foreblocks.models.transformer.runtime.state import DecoderState


@torch.no_grad()
def beam_search(
    decoder: DecoderProtocol,
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


__all__ = ["beam_search"]
