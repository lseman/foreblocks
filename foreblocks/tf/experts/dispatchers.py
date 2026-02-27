from __future__ import annotations

import math
from typing import Optional, Tuple

import torch


class DroplessPackedDispatcher:
    """
    Packs token-choice assignments by expert, with optional capacity pruning.
    Returns packed_x / packed_w and (experts_seq, tokens_seq, offsets, dropped).
    """

    def __init__(self, num_experts: int, top_k: int, capacity_factor: float = 1.25):
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.capacity_factor = float(capacity_factor)

        # Reusable buffers (grown as needed)
        self._buffer_size = 0
        self._sort_buffer: Optional[torch.Tensor] = None
        self._offsets_buffer: Optional[torch.Tensor] = None
        self._arange_buffer: Optional[torch.Tensor] = None
        self._tokens_buffer: Optional[torch.Tensor] = None

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
        capacity_factor: Optional[float] = None,
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
        if (weights <= 0).any():
            keep = weights > 0
            if keep.sum().item() == 0:
                empty = x_flat.new_zeros((0, D))
                empty_long = x_flat.new_zeros((0,), dtype=torch.long)
                empty_offsets = x_flat.new_zeros(
                    (self.num_experts + 1,), dtype=torch.long
                )
                return empty, empty, empty_long, empty_long, empty_offsets, 0
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

        # Vectorized capacity prune
        total_capacity = math.ceil(T * K * (capacity_factor or self.capacity_factor))
        per_expert_cap = max(1, math.ceil(total_capacity / self.num_experts))

        idx_in_expert = self._arange_buffer[:S] - offsets[experts_sorted]  # type: ignore[index]
        kept_mask = idx_in_expert < per_expert_cap
        kept = kept_mask.nonzero(as_tuple=True)[0]
        dropped = int(S - kept.numel())

        if dropped > 0:
            experts_sorted = experts_sorted[kept]
            tokens_sorted = tokens_sorted[kept]
            packed_x = packed_x[kept]
            packed_w = packed_w[kept]
            counts = torch.bincount(experts_sorted, minlength=self.num_experts)
            offsets = torch.cumsum(
                torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts]),
                dim=0,
            )

        return packed_x, packed_w, experts_sorted, tokens_sorted, offsets, dropped

    @staticmethod
    def scatter_back(
        out_accum: torch.Tensor, packed_y: torch.Tensor, tokens_seq: torch.Tensor
    ):
        out_accum.index_add_(0, tokens_seq, packed_y)
        return out_accum


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
        tokens_per_expert: Optional[int] = None,
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
    "DroplessPackedDispatcher",
    "ExpertChoiceDispatcher",
]

