"""Cache persistence and generation services for transformer decoders."""

from __future__ import annotations

from os import PathLike
from typing import Any, Protocol

import torch

from foreblocks.models.transformer.generation import GenerationConfig
from foreblocks.models.transformer.runtime.outputs import TransformerGenerationOutput
from foreblocks.modules.attention.cache.base import (
    cache_state_dict,
    load_cache_state_dict,
)
from foreblocks.modules.attention.cache.kv import StaticKVCache
from foreblocks.modules.attention.cache.paged import PagedKVCache


class DecoderProtocol(Protocol):
    output_size: int

    def parameters(self): ...
    def forward_one_step(self, tgt, memory, incremental_state=None, **kwargs): ...
    def forward_multi_step(self, tgt, memory, incremental_state, **kwargs): ...
    def prefill(self, tgt, memory, **kwargs): ...
    def decode(self, tgt, memory, incremental_state, **kwargs): ...


class DecoderCacheManager:
    def __init__(self, decoder: DecoderProtocol) -> None:
        self.decoder = decoder

    def reorder(
        self, state: dict[str, Any], beam_idx: torch.LongTensor
    ) -> dict[str, Any]:
        def select(value):
            if isinstance(value, (StaticKVCache, PagedKVCache)):
                return value.batch_select(beam_idx)
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.size(0) > int(beam_idx.max().item()):
                    return value.index_select(0, beam_idx.to(value.device))
                return value
            if isinstance(value, dict):
                return {key: select(item) for key, item in value.items()}
            if isinstance(value, list):
                return [select(item) for item in value]
            if isinstance(value, tuple):
                return tuple(select(item) for item in value)
            return value

        return select(state)

    def state_dict(self, state: dict[str, Any]) -> dict[str, Any]:
        return cache_state_dict(state)

    def load_state_dict(
        self, state: dict[str, Any], *, device: torch.device | str | None = None
    ) -> dict[str, Any]:
        if device is None:
            device = next(self.decoder.parameters()).device
        return load_cache_state_dict(state, device=device)

    def offload(self, state: dict[str, Any]) -> dict[str, Any]:
        return self.state_dict(state)

    def save(self, state: dict[str, Any], path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(state), path)

    def load(
        self, path: str | PathLike[str], *, device: torch.device | str | None = None
    ) -> dict[str, Any]:
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        return self.load_state_dict(snapshot, device=device)


class GenerationEngine:
    def __init__(self, decoder: DecoderProtocol, cache: DecoderCacheManager) -> None:
        self.decoder = decoder
        self.cache = cache

    def speculative_decode(
        self, draft_tokens, memory, incremental_state, *, verifier_fn=None, **kwargs
    ):
        start_lengths = []
        for layer_state in incremental_state.get("layers", []):
            self_state = (layer_state or {}).get("self_attn", {})
            cache = self_state.get("static_cache") or self_state.get("paged_cache")
            if cache is not None:
                start_lengths.append(cache.get_seq_length())
        output, state = self.decoder.forward_multi_step(
            draft_tokens, memory, incremental_state, **kwargs
        )
        accepted = (
            int(verifier_fn(output, draft_tokens))
            if verifier_fn is not None
            else draft_tokens.size(1)
        )
        accepted = max(0, min(accepted, draft_tokens.size(1)))
        if accepted != draft_tokens.size(1):
            for start, layer_state in zip(
                start_lengths, state.get("layers", []), strict=False
            ):
                self_state = (layer_state or {}).get("self_attn", {})
                cache = self_state.get("static_cache") or self_state.get("paged_cache")
                if cache is not None:
                    cache.crop(start + accepted)
        return output[:, :accepted], state, accepted

    def compile_prefill(self, **options):
        return torch.compile(self.decoder.prefill, **options)

    def compile_decode(self, **options):
        return torch.compile(self.decoder.decode, **options)

    @torch.no_grad()
    def generate(
        self,
        initial_tgt,
        memory,
        max_new_tokens=None,
        *,
        generation_config=None,
        incremental_state=None,
        feedback_fn=None,
        memory_mask=None,
        memory_key_padding_mask=None,
        return_dict=None,
    ):
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
                "output_size must match the decoder input width unless feedback_fn is provided"
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

    @torch.no_grad()
    def beam_search(self, initial_tgt, memory, max_new_tokens, num_beams, proposal_fn):
        if num_beams < 1 or max_new_tokens < 1:
            raise ValueError("num_beams and max_new_tokens must be positive")
        batch_size = initial_tgt.size(0)
        prediction, state = self.decoder.forward_one_step(initial_tgt, memory)
        prediction = prediction[:, -1:, :]
        beam_scores = prediction.new_zeros(batch_size, 1)
        histories = None
        beam_memory = memory
        for step in range(max_new_tokens):
            candidates, candidate_scores = proposal_fn(prediction, step)
            if candidates.ndim != 3 or candidate_scores.shape != candidates.shape[:2]:
                raise ValueError(
                    "proposal_fn must return [N,K,C] values and [N,K] scores"
                )
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
            selected = rows[batch_rows, parent, choice].reshape(
                batch_size * keep, 1, -1
            )
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
            state = self.cache.reorder(state, parent_global)
            beam_memory = beam_memory.index_select(
                0, parent_global.to(beam_memory.device)
            )
            if step + 1 < max_new_tokens:
                prediction, state = self.decoder.forward_one_step(
                    selected, beam_memory, incremental_state=state
                )
                prediction = prediction[:, -1:, :]
        return histories[:, 0], beam_scores[:, 0], state


__all__ = ["DecoderCacheManager", "GenerationEngine"]
