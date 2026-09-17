"""Lower-triangular cumulative-sum (L-matrix) helpers for chunked SSD."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def segment_sum(x: torch.Tensor) -> torch.Tensor:
    cumsum = torch.cumsum(x, dim=-1)  # [..., C]
    cumsum_pad = F.pad(cumsum, (1, 0), value=0.0)  # [..., C+1]
    cumsum_pad = cumsum_pad[..., :-1]  # [..., C], cumsum_pad[j] = cumsum[j-1]
    diff = cumsum[..., :, None] - cumsum_pad[..., None, :]  # [..., C, C]
    tril = torch.tril(
        torch.ones(x.shape[-1], x.shape[-1], device=x.device, dtype=torch.bool)
    )
    diff = diff.masked_fill(~tril, float("-inf"))
    return torch.exp(diff)  # [..., C, C]


def _segment_sum_log(x: torch.Tensor) -> torch.Tensor:
    size = x.size(-1)
    expanded = x[..., None].expand(*x.shape, size)
    strict_lower = torch.tril(
        torch.ones(size, size, device=x.device, dtype=torch.bool),
        diagonal=-1,
    )
    expanded = expanded.masked_fill(~strict_lower, 0.0)
    segsum = torch.cumsum(expanded, dim=-2)
    lower = torch.tril(torch.ones(size, size, device=x.device, dtype=torch.bool))
    return segsum.masked_fill(~lower, float("-inf"))
