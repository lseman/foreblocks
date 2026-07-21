"""Attention kernel dispatch boundary."""

from __future__ import annotations

from typing import Protocol

import torch


class KernelDispatchOwner(Protocol):
    def _compute_attention_direct(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        q_start_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


class AttentionKernelDispatcher:
    def __init__(self, owner: KernelDispatchOwner) -> None:
        self.owner = owner

    def compute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
        need_weights: bool,
        q_start_pos: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.owner._compute_attention_direct(
            q,
            k,
            v,
            attn_mask,
            key_padding_mask,
            is_causal,
            need_weights,
            q_start_pos,
        )


__all__ = ["AttentionKernelDispatcher", "KernelDispatchOwner"]
