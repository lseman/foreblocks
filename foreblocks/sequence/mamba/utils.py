"""foreblocks.sequence.mamba.utils.

This module implements the utils pieces for its package.
It belongs to the Mamba and state-space operator kernels area of Foreblocks.
It exposes functions such as auto_dt_rank, inverse_softplus, fused_out_2d, conv_step.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def auto_dt_rank(d_model: int) -> int:
    return max(4, math.ceil(d_model / 16))


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    return x + torch.log(-torch.expm1(-x))


def fused_out_2d(
    y: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """RMSNormGated for 2-D tensors: rms_norm(y, w) * silu(z). Matches official Mamba2."""
    rms = torch.rsqrt((y * y).mean(-1, keepdim=True) + eps)
    return y * rms * norm_weight * F.silu(z)


def conv_step(
    u_raw: torch.Tensor,
    state_conv: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One step of a causal depthwise conv from a ring buffer."""
    full = torch.cat([state_conv, u_raw.unsqueeze(-1)], dim=-1)
    u = (full * weight.unsqueeze(0)).sum(-1)
    if bias is not None:
        u = u + bias
    return u, full[:, :, 1:]
