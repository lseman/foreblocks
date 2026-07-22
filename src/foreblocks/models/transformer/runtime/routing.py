"""Tensor routing helpers for GateSkip and Mixture-of-Depths execution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
import torch.nn as nn

from foreblocks.models.transformer.features.patching import patchify_padding_mask


class RoutingOwner(Protocol):
    def _prepare_layer_routing(
        self,
        layer_idx: int,
        x: torch.Tensor,
        active_mask: torch.Tensor | None,
    ) -> tuple[
        nn.Module,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]: ...


def gateskip_active_mask_from_padding(
    padding_mask: torch.Tensor | None,
) -> torch.Tensor | None:
    return None if padding_mask is None else ~padding_mask.to(dtype=torch.bool)


def patchify_gateskip_active_mask(
    active_mask: torch.Tensor | None,
    *,
    T: int,
    patch_len: int,
    stride: int,
    pad_end: bool,
) -> torch.Tensor | None:
    if active_mask is None:
        return None
    patch_pad_mask = patchify_padding_mask(
        ~active_mask.to(dtype=torch.bool),
        T=T,
        patch_len=patch_len,
        stride=stride,
        pad_end=pad_end,
    )
    return None if patch_pad_mask is None else ~patch_pad_mask


def gather_sequence_tokens(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == 0:
        return x[:, :0]
    view = [indices.size(0), indices.size(1)] + [1] * (x.dim() - 2)
    shape = [indices.size(0), indices.size(1), *x.shape[2:]]
    return x.gather(1, indices.view(*view).expand(*shape))


def gather_padding_mask(
    mask: torch.Tensor | None,
    indices: torch.Tensor,
    slot_mask: torch.Tensor,
) -> torch.Tensor:
    if indices.numel() == 0:
        return slot_mask.new_ones(slot_mask.shape)
    if mask is None:
        return ~slot_mask
    return mask.to(dtype=torch.bool).gather(1, indices) | (~slot_mask)


def _canonicalize_attention_mask(
    mask: torch.Tensor, batch_size: int
) -> tuple[torch.Tensor, int]:
    if mask.dim() == 2:
        return mask.unsqueeze(0).expand(batch_size, -1, -1), 1
    if mask.dim() not in (3, 4):
        raise ValueError(f"Unsupported attention mask shape {tuple(mask.shape)}")
    if mask.size(0) == 1 and batch_size > 1:
        mask = mask.expand(batch_size, *mask.shape[1:])
    if mask.size(0) != batch_size:
        raise ValueError(
            f"Batch attention mask shape {tuple(mask.shape)} incompatible with "
            f"batch size {batch_size}"
        )
    return mask, mask.dim() - 2


def gather_square_mask(
    mask: torch.Tensor | None, indices: torch.Tensor
) -> torch.Tensor | None:
    if mask is None:
        return None
    if indices.numel() == 0:
        return mask.new_zeros((indices.size(0), 0, 0))
    batch_size, capacity = indices.shape
    base, query_dim = _canonicalize_attention_mask(mask, batch_size)
    if query_dim == 1:
        rows = base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
        return rows.gather(2, indices.unsqueeze(1).expand(-1, capacity, -1))
    rows = base.gather(
        2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
    )
    return rows.gather(
        3, indices[:, None, None, :].expand(-1, base.size(1), capacity, -1)
    )


def gather_query_mask(
    mask: torch.Tensor | None, indices: torch.Tensor
) -> torch.Tensor | None:
    if mask is None:
        return None
    if indices.numel() == 0:
        return mask.new_zeros((indices.size(0), 0, mask.size(-1)))
    batch_size = indices.size(0)
    base, query_dim = _canonicalize_attention_mask(mask, batch_size)
    if query_dim == 1:
        return base.gather(1, indices.unsqueeze(-1).expand(-1, -1, base.size(-1)))
    return base.gather(
        2, indices[:, None, :, None].expand(-1, base.size(1), -1, base.size(-1))
    )


def scatter_mixture_of_depths_output(
    x_base,
    x_routed_in,
    x_routed_out,
    indices,
    slot_mask,
    router_logits,
    use_expert_choice=True,
):
    if indices.numel() == 0:
        return x_base
    out = x_base.clone()
    batch_idx, slot_idx = slot_mask.nonzero(as_tuple=True)
    token_idx = indices[batch_idx, slot_idx]
    if use_expert_choice:
        values = x_routed_out[batch_idx, slot_idx]
    else:
        delta = x_routed_out[batch_idx, slot_idx] - x_routed_in[batch_idx, slot_idx]
        scale = torch.sigmoid(router_logits[batch_idx, slot_idx]).unsqueeze(-1)
        values = x_base[batch_idx, token_idx] + scale * delta
    out[batch_idx, token_idx] = values
    return out


def run_mod_layer(
    owner: RoutingOwner,
    layer_idx: int,
    x: torch.Tensor,
    active_mask: torch.Tensor | None,
    all_hidden_states: list[torch.Tensor] | None,
    router_states: list[object],
    gather_and_invoke: Callable[
        [nn.Module, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> tuple[torch.Tensor, bool]:
    layer, router_logits, _keep_mask, routed_indices, routed_slots = (
        owner._prepare_layer_routing(layer_idx, x, active_mask)
    )
    if routed_indices is None or routed_slots is None or not bool(routed_slots.any()):
        if all_hidden_states is not None:
            all_hidden_states.append(x)
        return x, False

    x_routed, x_routed_out = gather_and_invoke(layer, routed_indices, routed_slots)
    x = scatter_mixture_of_depths_output(
        x,
        x_routed,
        x_routed_out,
        routed_indices,
        routed_slots,
        router_logits,
        use_expert_choice=True,  # Standard Expert Choice: full replacement
    )
    if all_hidden_states is not None:
        all_hidden_states.append(x)
    ff_block = getattr(getattr(layer, "feed_forward", None), "block", None)
    router_state = getattr(ff_block, "last_routing_state", None)
    if router_state is not None:
        router_states.append(router_state)
    return x, True


__all__ = [
    "gateskip_active_mask_from_padding",
    "gather_padding_mask",
    "gather_query_mask",
    "gather_sequence_tokens",
    "gather_square_mask",
    "patchify_gateskip_active_mask",
    "run_mod_layer",
    "scatter_mixture_of_depths_output",
]
