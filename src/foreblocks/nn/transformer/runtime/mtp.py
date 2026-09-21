"""Validation and alignment of multi-token prediction targets for decoder FFNs."""

from __future__ import annotations

import torch
from torch import nn


def build_decoder_mtp_targets(
    targets: torch.Tensor,
    *,
    batch_size: int,
    sequence_length: int,
    d_model: int,
    num_heads: int,
    input_adapter: nn.Module,
) -> torch.Tensor:
    """Build [B,T,H,D] targets, with head h predicting position t+h+1.

    Three-dimensional targets may use the input or model width. Already
    aligned four-dimensional targets are returned unchanged; extra horizons
    are allowed because individual layers may consume fewer heads. Missing
    future positions retain the decoder's existing zero-padding convention.
    """
    if targets.ndim not in (3, 4):
        raise ValueError(
            f"mtp_targets must be [B,T,F] or [B,T,H,D], got {tuple(targets.shape)}"
        )
    if targets.shape[:2] != (batch_size, sequence_length):
        raise ValueError(
            f"mtp_targets batch and length must match decoder "
            f"[B={batch_size},T={sequence_length}], got {tuple(targets.shape)}"
        )
    if targets.ndim == 4:
        if targets.size(-1) != d_model:
            raise ValueError(f"mtp_targets feature dim must match d_model {d_model}")
        if targets.size(2) < num_heads:
            raise ValueError(f"mtp_targets horizon must be at least {num_heads}")
        return targets

    base = targets
    if base.size(-1) != d_model:
        input_size = getattr(input_adapter, "in_features", None)
        if base.size(-1) != input_size:
            raise ValueError(
                f"mtp_targets last dim {base.size(-1)} must be decoder input "
                f"{input_size} or d_model {d_model}"
            )
        base = input_adapter(base)

    shifted = base.new_zeros(batch_size, sequence_length, max(0, num_heads), d_model)
    for head in range(min(num_heads, sequence_length - 1)):
        offset = head + 1
        shifted[:, :-offset, head, :] = base[:, offset:, :]
    return shifted


__all__ = ["build_decoder_mtp_targets"]
