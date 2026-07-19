"""Centralized attention-mask normalization and cache-aware construction."""

from __future__ import annotations

import torch


def normalize_blocked_mask(
    mask: torch.Tensor,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_length: int,
) -> torch.Tensor:
    blocked = mask.bool()
    if blocked.dim() == 2:
        blocked = blocked.view(1, 1, query_length, key_length)
    elif blocked.dim() == 3:
        blocked = blocked.view(blocked.shape[0], 1, query_length, key_length)
    elif blocked.dim() != 4:
        raise ValueError("attention mask must be 2D, 3D, or 4D")
    if blocked.shape[-2:] != (query_length, key_length):
        raise ValueError(
            f"attention mask shape {tuple(blocked.shape[-2:])} must be "
            f"{(query_length, key_length)}"
        )
    if blocked.shape[0] not in (1, batch_size):
        raise ValueError("attention mask batch dimension is incompatible")
    if blocked.shape[1] not in (1, num_heads):
        raise ValueError("attention mask head dimension is incompatible")
    return blocked.expand(
        batch_size if blocked.shape[0] == 1 else blocked.shape[0],
        num_heads if blocked.shape[1] == 1 else blocked.shape[1],
        -1,
        -1,
    )


def build_attention_mask(
    *,
    query: torch.Tensor,
    key_length: int,
    attention_mask: torch.Tensor | None = None,
    padding_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    cache_position: torch.Tensor | None = None,
    key_lengths: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """Return a boolean mask with ``True = blocked`` semantics."""
    batch_size, num_heads, query_length, _ = query.shape
    blocked = None
    if attention_mask is not None:
        blocked = normalize_blocked_mask(
            attention_mask,
            batch_size=batch_size,
            num_heads=num_heads,
            query_length=query_length,
            key_length=key_length,
        )
    if padding_mask is not None:
        padding = padding_mask.to(device=query.device, dtype=torch.bool)
        if padding.shape != (batch_size, key_length):
            raise ValueError(
                f"padding mask shape {tuple(padding.shape)} must be "
                f"{(batch_size, key_length)}"
            )
        padding = padding.view(batch_size, 1, 1, key_length)
        blocked = padding if blocked is None else blocked | padding
    if key_lengths is not None:
        unused = torch.arange(key_length, device=query.device).view(1, 1, 1, -1)
        unused = unused >= key_lengths.to(query.device).view(batch_size, 1, 1, 1)
        blocked = unused if blocked is None else blocked | unused
    if is_causal:
        if cache_position is None:
            query_positions = torch.arange(query_length, device=query.device)
            query_positions = query_positions.view(1, 1, query_length, 1)
        else:
            positions = cache_position.to(device=query.device, dtype=torch.long)
            if positions.dim() == 1:
                positions = positions.unsqueeze(0).expand(batch_size, -1)
            if positions.shape != (batch_size, query_length):
                raise ValueError("cache_position must have shape [B, Tq] or [Tq]")
            query_positions = positions.view(batch_size, 1, query_length, 1)
        key_positions = torch.arange(key_length, device=query.device).view(1, 1, 1, -1)
        causal = key_positions > query_positions
        blocked = causal if blocked is None else blocked | causal
    if blocked is not None:
        blocked = blocked.expand(batch_size, num_heads, query_length, key_length)
    return blocked


def to_additive_mask(blocked: torch.Tensor | None, *, dtype: torch.dtype):
    if blocked is None:
        return None
    return torch.zeros(blocked.shape, device=blocked.device, dtype=dtype).masked_fill(
        blocked, float("-inf")
    )


__all__ = ["build_attention_mask", "normalize_blocked_mask", "to_additive_mask"]
