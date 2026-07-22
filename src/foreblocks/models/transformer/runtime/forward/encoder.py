"""Input tokenization and mask preparation for transformer encoder execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from foreblocks.models.transformer.features.patching import (
    PatchInfo,
    PatchTokenizer,
    patchify_padding_mask,
)
from foreblocks.models.transformer.runtime.routing import patchify_gateskip_active_mask


class EncoderPreparationOwner(Protocol):
    input_size: int
    max_seq_len: int
    patch_encoder: bool
    patch_len: int
    patch_stride: int
    patch_pad_end: bool
    ct_patchtst: bool
    ct_patch_len: int
    ct_patch_stride: int
    ct_patch_pad_end: bool
    input_adapter: nn.Module
    patcher: PatchTokenizer

    def _ct_patchify(self, src: torch.Tensor) -> tuple[torch.Tensor, PatchInfo]: ...


@dataclass(frozen=True)
class PreparedEncoderInput:
    hidden_states: torch.Tensor
    padding_mask: torch.Tensor | None
    active_mask: torch.Tensor | None
    patch_info: PatchInfo | None


def prepare_encoder_input(
    owner: EncoderPreparationOwner,
    src: torch.Tensor,
    padding_mask: torch.Tensor | None,
    active_mask: torch.Tensor | None,
) -> PreparedEncoderInput:
    """Project or patch encoder input and transform its token-aligned masks."""
    if src.ndim != 3:
        raise ValueError(f"encoder expects src [B,T,C], got {tuple(src.shape)}")
    _, sequence_length, input_size = src.shape
    if owner.input_size != input_size:
        raise ValueError(f"Expected input size {owner.input_size}, got {input_size}")
    if (
        sequence_length > owner.max_seq_len
        and not owner.patch_encoder
        and not owner.ct_patchtst
    ):
        raise ValueError(
            f"Sequence length {sequence_length} exceeds max {owner.max_seq_len}"
        )

    patch_info: PatchInfo | None = None
    if owner.ct_patchtst:
        hidden_states, patch_info = owner._ct_patchify(src)
        patch_len, stride, pad_end = (
            owner.ct_patch_len,
            owner.ct_patch_stride,
            owner.ct_patch_pad_end,
        )
        label = "CT-patch"
    else:
        hidden_states = owner.input_adapter(src)
        if not owner.patch_encoder:
            return PreparedEncoderInput(
                hidden_states, padding_mask, active_mask, patch_info
            )
        hidden_states, patch_info = owner.patcher(hidden_states)
        patch_len, stride, pad_end = (
            owner.patch_len,
            owner.patch_stride,
            owner.patch_pad_end,
        )
        label = "patch"

    if hidden_states.shape[1] > owner.max_seq_len:
        raise ValueError(
            f"Encoder {label} token length {hidden_states.shape[1]} exceeds "
            f"max_seq_len={owner.max_seq_len}. Increase max_seq_len or adjust "
            f"{label.replace('-', '_')}_len/{label.replace('-', '_')}_stride."
        )
    padding_mask = patchify_padding_mask(
        padding_mask,
        T=sequence_length,
        patch_len=patch_len,
        stride=stride,
        pad_end=pad_end,
    )
    active_mask = patchify_gateskip_active_mask(
        active_mask,
        T=sequence_length,
        patch_len=patch_len,
        stride=stride,
        pad_end=pad_end,
    )
    return PreparedEncoderInput(hidden_states, padding_mask, active_mask, patch_info)


__all__ = ["EncoderPreparationOwner", "PreparedEncoderInput", "prepare_encoder_input"]
