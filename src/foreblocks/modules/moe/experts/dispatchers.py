"""foreblocks.modules.moe.experts.dispatchers.

Token packing and scatter for Mixture-of-Experts routing.

Packs token-to-expert assignments by expert index without capacity pruning,
and provides scatter-back for accumulating expert outputs. Supports both token-choice
(every token picks top-K experts) and expert-choice (every expert picks top tokens)
routing modes with hard or soft capacity limits. Use when building custom MoE layers
that need vectorized packing without dropping tokens.

Core API:
- DroplessPackedDispatcher: token-choice dispatcher that preserves every assignment
- ExpertChoiceDispatcher: expert-choice dispatcher with per-expert token selection

"""

from __future__ import annotations

import math

import torch


class DroplessPackedDispatcher:
    """
    Packs every token-choice assignment by expert without capacity pruning.

    Capacity arguments remain accepted for API compatibility, but this dispatcher
    is deliberately dropless. Capacity-limited routing belongs in a separately
    named dispatcher so callers can rely on this class's contract.
    Returns packed_x / packed_w and (experts_seq, tokens_seq, offsets, dropped).
    """

    def __init__(self, num_experts: int, top_k: int, capacity_factor: float = 1.25):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)

        # Reusable buffers (grown as needed)
        self._buffer_size = 0
        self._sort_buffer: torch.Tensor | None = None
        self._offsets_buffer: torch.Tensor | None = None
        self._arange_buffer: torch.Tensor | None = None
        self._tokens_buffer: torch.Tensor | None = None

    def _ensure_buffers(self, size: int, device: torch.device):
        cur_device = self._sort_buffer.device if self._sort_buffer is not None else None
        if (
            self._sort_buffer is None
            or self._buffer_size < size
            or cur_device != device
        ):
            self._buffer_size = size
            self._sort_buffer = torch.empty(size, dtype=torch.long, device=device)
            self._offsets_buffer = torch.empty(
                self.num_experts + 1, dtype=torch.long, device=device
            )
            self._arange_buffer = torch.arange(size, dtype=torch.long, device=device)
            self._tokens_buffer = torch.empty(size, dtype=torch.long, device=device)

    def pack(
        self,
        x_flat: torch.Tensor,  # [T, D]
        topk_p: torch.Tensor,  # [T, K]
        topk_i: torch.Tensor,  # [T, K]
        capacity_factor: float | None = None,
        soft_capacity: bool = False,
        expert_usage: torch.Tensor | None = None,
        soft_capacity_min: float = 0.5,
        soft_capacity_max: float = 2.0,
        filter_zero_weights: bool = False,
    ):
        device = x_flat.device
        T, D = x_flat.shape
        K = topk_i.shape[1]
        S = T * K
        if S == 0:
            empty = x_flat.new_zeros((0, D))
            empty_long = x_flat.new_zeros((0,), dtype=torch.long)
            empty_offsets = x_flat.new_zeros((self.num_experts + 1,), dtype=torch.long)
            return empty, empty, empty_long, empty_long, empty_offsets, 0

        self._ensure_buffers(S, device)

        # Flatten routing decisions
        experts = topk_i.reshape(-1)  # [S]
        weights = topk_p.reshape(-1)  # [S]
        torch.div(
            self._arange_buffer[:S],  # type: ignore[index]
            K,
            rounding_mode="floor",
            out=self._tokens_buffer[:S],  # type: ignore[index]
        )
        tokens = self._tokens_buffer[:S]  # type: ignore[index]

        # Drop masked/zero-weight routes (used for adaptive K)
        if filter_zero_weights:
            keep = weights > 0
            experts = experts[keep]
            weights = weights[keep]
            tokens = tokens[keep]
            S = int(weights.numel())
            self._ensure_buffers(S, device)

        # Sort by expert (stable)
        torch.argsort(experts, stable=True, out=self._sort_buffer[:S])
        sort_idx = self._sort_buffer[:S]
        experts_sorted = experts[sort_idx]  # [S]
        weights_sorted = weights[sort_idx]  # [S]
        tokens_sorted = tokens[sort_idx]  # [S]

        # Offsets via bincount
        counts = torch.bincount(experts_sorted, minlength=self.num_experts)  # [E]
        self._offsets_buffer.zero_()
        torch.cumsum(counts, 0, out=self._offsets_buffer[1:])
        offsets = self._offsets_buffer.clone()  # [E+1]

        # Pack inputs and weights
        packed_x = x_flat.index_select(0, tokens_sorted)  # [S, D]
        packed_w = weights_sorted.unsqueeze(1)  # [S, 1]

        del capacity_factor, soft_capacity, expert_usage
        del soft_capacity_min, soft_capacity_max
        del filter_zero_weights
        return packed_x, packed_w, experts_sorted, tokens_sorted, offsets, 0

    @staticmethod
    def scatter_back(
        out_accum: torch.Tensor, packed_y: torch.Tensor, tokens_seq: torch.Tensor
    ):
        out_accum.index_add_(0, tokens_seq, packed_y)
        return out_accum


class ConfidenceCapacityDispatcher(DroplessPackedDispatcher):
    """Capacity-limited dispatcher that keeps the strongest routes per expert."""

    def pack(
        self,
        x_flat: torch.Tensor,
        topk_p: torch.Tensor,
        topk_i: torch.Tensor,
        capacity_factor: float | None = None,
        soft_capacity: bool = False,
        expert_usage: torch.Tensor | None = None,
        soft_capacity_min: float = 0.5,
        soft_capacity_max: float = 2.0,
        filter_zero_weights: bool = False,
    ):
        packed_x, packed_w, experts, tokens, offsets, _ = super().pack(
            x_flat, topk_p, topk_i, filter_zero_weights=filter_zero_weights
        )
        assignment_count = int(experts.numel())
        if assignment_count == 0:
            return packed_x, packed_w, experts, tokens, offsets, 0

        token_count = int(x_flat.size(0))
        routes_per_token = int(topk_i.size(1))
        factor = self.capacity_factor if capacity_factor is None else capacity_factor
        total_capacity = math.ceil(token_count * routes_per_token * float(factor))
        base_capacity = total_capacity / max(self.num_experts, 1)
        if soft_capacity and expert_usage is not None:
            usage = expert_usage.float().clamp_min(1e-6)
            scale = usage / usage.sum().clamp_min(1e-12) * self.num_experts
            scale = scale.clamp(min=soft_capacity_min, max=soft_capacity_max)
            capacities = (base_capacity * scale).ceil().long().clamp_min(1)
        else:
            capacities = torch.full(
                (self.num_experts,),
                max(1, math.ceil(base_capacity)),
                dtype=torch.long,
                device=x_flat.device,
            )

        kept_parts: list[torch.Tensor] = []
        for expert_idx in range(self.num_experts):
            start = int(offsets[expert_idx].item())
            end = int(offsets[expert_idx + 1].item())
            count = end - start
            capacity = min(count, int(capacities[expert_idx].item()))
            if capacity == 0:
                continue
            local_weights = packed_w[start:end, 0]
            strongest = torch.topk(
                local_weights, k=capacity, sorted=False
            ).indices.add(start)
            kept_parts.append(strongest)

        kept = torch.cat(kept_parts) if kept_parts else experts.new_empty((0,))
        experts = experts.index_select(0, kept)
        tokens = tokens.index_select(0, kept)
        packed_x = packed_x.index_select(0, kept)
        packed_w = packed_w.index_select(0, kept)
        counts = torch.bincount(experts, minlength=self.num_experts)
        offsets = torch.cumsum(
            torch.cat([counts.new_zeros(1), counts]), dim=0
        )
        return (
            packed_x,
            packed_w,
            experts,
            tokens,
            offsets,
            assignment_count - int(kept.numel()),
        )


class ExpertChoiceDispatcher:
    """
    Expert-choice dispatcher:
    each expert selects top tokens; token-wise probabilities are normalized
    over selected experts for weighted scatter.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        capacity_factor: float = 1.25,
        tokens_per_expert: int | None = None,
    ):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)
        self.tokens_per_expert = (
            int(tokens_per_expert) if tokens_per_expert is not None else None
        )

    def _resolve_tokens_per_expert(self, num_tokens: int) -> int:
        if num_tokens <= 0:
            return 0
        if self.tokens_per_expert is not None:
            return min(self.tokens_per_expert, num_tokens)
        # Keep default assignment volume close to token-choice mode.
        inferred = math.ceil(
            num_tokens * self.top_k * self.capacity_factor / max(self.num_experts, 1)
        )
        return max(1, min(inferred, num_tokens))

    def pack(self, x_flat: torch.Tensor, logits: torch.Tensor):
        device = x_flat.device
        T, D = x_flat.shape
        E = logits.size(-1)
        cap = self._resolve_tokens_per_expert(T)
        if T == 0 or E == 0 or cap == 0:
            empty = x_flat.new_zeros((0, D))
            empty_long = x_flat.new_zeros((0,), dtype=torch.long)
            empty_offsets = x_flat.new_zeros((E + 1,), dtype=torch.long)
            return empty, empty, empty_long, empty_long, empty_offsets, 0

        # Per-expert top tokens: logits [T, E] -> [E, T]
        expert_logits = logits.transpose(0, 1).contiguous()
        top_v, top_tok = torch.topk(expert_logits, k=cap, dim=-1, sorted=False)

        experts_seq = (
            torch.arange(E, device=device, dtype=torch.long)
            .unsqueeze(1)
            .expand(-1, cap)
            .reshape(-1)
        )
        tokens_seq = top_tok.reshape(-1)
        packed_x = x_flat.index_select(0, tokens_seq)

        # Token-wise normalize selected expert scores into combine weights.
        flat_scores = top_v.reshape(-1).float()
        flat_exp = torch.exp(flat_scores.clamp(min=-60.0, max=60.0))
        token_denom = flat_exp.new_zeros((T,))
        token_denom.index_add_(0, tokens_seq, flat_exp)
        flat_w = flat_exp / (token_denom.index_select(0, tokens_seq) + 1e-12)
        packed_w = flat_w.to(dtype=x_flat.dtype).unsqueeze(1)

        offsets = torch.arange(
            0, (E + 1) * cap, step=cap, device=device, dtype=torch.long
        )

        token_counts = torch.zeros((T,), device=device, dtype=torch.long)
        token_counts.index_add_(0, tokens_seq, torch.ones_like(tokens_seq))
        not_selected = int((token_counts == 0).sum().item())

        return packed_x, packed_w, experts_seq, tokens_seq, offsets, not_selected

    @staticmethod
    def scatter_back(
        out_accum: torch.Tensor, packed_y: torch.Tensor, tokens_seq: torch.Tensor
    ):
        out_accum.index_add_(0, tokens_seq, packed_y)
        return out_accum


__all__ = [
    "ConfidenceCapacityDispatcher",
    "DroplessPackedDispatcher",
    "ExpertChoiceDispatcher",
]
