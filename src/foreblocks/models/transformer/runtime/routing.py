"""Tensor routing helpers for GateSkip and Mixture-of-Depths execution."""

from __future__ import annotations

import torch

from foreblocks.models.transformer.features.patching import patchify_padding_mask


def gateskip_active_mask_from_padding(padding_mask):
    return None if padding_mask is None else ~padding_mask.to(dtype=torch.bool)


def patchify_gateskip_active_mask(active_mask, *, T, patch_len, stride, pad_end):
    if active_mask is None:
        return None
    patch_pad_mask = patchify_padding_mask(
        ~active_mask.to(dtype=torch.bool), T=T, patch_len=patch_len,
        stride=stride, pad_end=pad_end,
    )
    return None if patch_pad_mask is None else ~patch_pad_mask


def gather_sequence_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return x[:, :0]
    view = [indices.size(0), indices.size(1)] + [1] * (x.dim() - 2)
    shape = [indices.size(0), indices.size(1)] + list(x.shape[2:])
    return x.gather(1, indices.view(*view).expand(*shape))


def gather_padding_mask(mask, indices, slot_mask):
    if indices.numel() == 0:
        return slot_mask.new_ones(slot_mask.shape)
    if mask is None:
        return ~slot_mask
    return mask.to(dtype=torch.bool).gather(1, indices) | (~slot_mask)


def _batch_mask(mask, batch_size):
    if mask.size(0) == 1 and batch_size > 1:
        return mask.expand(batch_size, *mask.shape[1:])
    if mask.size(0) != batch_size:
        raise ValueError(
            f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
            f"batch size {batch_size}"
        )
    return mask


def gather_square_mask(mask, indices):
    if mask is None:
        return None
    if indices.numel() == 0:
        return mask.new_zeros((indices.size(0), 0, 0))
    batch_size, capacity = indices.shape
    if mask.dim() == 2:
        base = mask.unsqueeze(0).expand(batch_size, -1, -1)
    elif mask.dim() in (3, 4):
        base = _batch_mask(mask, batch_size)
    else:
        raise ValueError(f"Unsupported square attention mask shape {tuple(mask.shape)}")
    if mask.dim() < 4:
        rows = base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
        return rows.gather(2, indices.unsqueeze(1).expand(-1, capacity, -1))
    rows = base.gather(
        2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
    )
    return rows.gather(
        3, indices[:, None, None, :].expand(-1, base.size(1), capacity, -1)
    )


def gather_query_mask(mask, indices):
    if mask is None:
        return None
    if indices.numel() == 0:
        return mask.new_zeros((indices.size(0), 0, mask.size(-1)))
    batch_size = indices.size(0)
    if mask.dim() == 2:
        base = mask.unsqueeze(0).expand(batch_size, -1, -1)
    elif mask.dim() in (3, 4):
        base = _batch_mask(mask, batch_size)
    else:
        raise ValueError(f"Unsupported query attention mask shape {tuple(mask.shape)}")
    if mask.dim() < 4:
        return base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
    return base.gather(
        2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
    )


def scatter_mixture_of_depths_output(
    x_base, x_routed_in, x_routed_out, indices, slot_mask, router_logits,
    use_expert_choice=True,
):
    if indices.numel() == 0:
        return x_base
    out = x_base.clone()
    for batch_idx in range(out.size(0)):
        valid = slot_mask[batch_idx]
        if not bool(valid.any()):
            continue
        idx = indices[batch_idx, valid]
        if use_expert_choice:
            out[batch_idx, idx] = x_routed_out[batch_idx, valid]
        else:
            delta = x_routed_out[batch_idx, valid] - x_routed_in[batch_idx, valid]
            scale = torch.sigmoid(router_logits[batch_idx, valid]).unsqueeze(-1)
            out[batch_idx, idx] = x_base[batch_idx, idx] + scale * delta
    return out


_gateskip_active_mask_from_padding = gateskip_active_mask_from_padding
_patchify_gateskip_active_mask = patchify_gateskip_active_mask
_gather_sequence_tokens = gather_sequence_tokens
_gather_padding_mask = gather_padding_mask
_gather_square_mask = gather_square_mask
_gather_query_mask = gather_query_mask
_scatter_mixture_of_depths_output = scatter_mixture_of_depths_output

__all__ = [
    "gateskip_active_mask_from_padding", "gather_padding_mask", "gather_query_mask",
    "gather_sequence_tokens", "gather_square_mask", "patchify_gateskip_active_mask",
    "scatter_mixture_of_depths_output",
]
