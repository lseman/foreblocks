from __future__ import annotations

import torch
import torch.nn.functional as F

from .causal_conv1d import causal_depthwise_conv1d
from .ssd import chunked_ssd_forward
from .triton_ops import dt_prep, fused_out


def _repeat_group_params(
    x: torch.Tensor,
    *,
    num_heads: int,
    n_groups: int,
    d_state: int,
) -> torch.Tensor:
    B, T, _ = x.shape
    return x.reshape(B, T, n_groups, d_state).repeat_interleave(
        num_heads // n_groups,
        dim=2,
    )


def mamba2_split_conv1d_scan_combined(
    projected: torch.Tensor,
    residual_inner: torch.Tensor,
    *,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    dt_proj_weight: torch.Tensor,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    Dskip: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    d_inner: int,
    conv_dim: int,
    dt_rank: int,
    num_heads: int,
    head_dim: int,
    n_groups: int,
    d_state: int,
    chunk_size: int,
    dt_limit: tuple[float, float],
    norm_eps: float,
    attention_mask: torch.Tensor | None = None,
    use_triton_ssd: bool = True,
) -> torch.Tensor:
    """Combined Mamba2 block path following FLA's split/conv/scan layout.

    This keeps the layer-level dataflow in one op boundary while reusing the
    local Triton kernels for causal conv, dt prep, SSD, and gated RMSNorm.
    """
    z, conv_input, dt_hidden = torch.split(
        projected,
        [d_inner, conv_dim, dt_rank],
        dim=-1,
    )
    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != projected.shape[:2]:
            raise ValueError("attention_mask must have shape [B, T]")
        mask = attention_mask.to(dtype=projected.dtype, device=projected.device)
        conv_input = conv_input * mask.unsqueeze(-1)

    conv_out = causal_depthwise_conv1d(
        conv_input.transpose(1, 2),
        conv_weight,
        conv_bias,
    ).transpose(1, 2)
    if attention_mask is not None:
        conv_out = conv_out * mask.unsqueeze(-1)

    u, Bflat, Cflat = torch.split(
        conv_out,
        [d_inner, n_groups * d_state, n_groups * d_state],
        dim=-1,
    )
    Bpar = _repeat_group_params(
        Bflat,
        num_heads=num_heads,
        n_groups=n_groups,
        d_state=d_state,
    )
    Cpar = _repeat_group_params(
        Cflat,
        num_heads=num_heads,
        n_groups=n_groups,
        d_state=d_state,
    )

    dt_raw = F.linear(dt_hidden, dt_proj_weight)
    dt = dt_prep(dt_raw, dt_bias, dt_min=dt_limit[0], dt_max=dt_limit[1])
    A = -torch.exp(A_log)

    y = chunked_ssd_forward(
        u=u.reshape(u.shape[0], u.shape[1], num_heads, head_dim),
        dt=dt,
        A=A,
        B=Bpar,
        C=Cpar,
        D=Dskip,
        chunk_size=chunk_size,
        use_triton=use_triton_ssd,
    ).reshape(u.shape[0], u.shape[1], d_inner)
    y = fused_out(y, z, residual_inner, norm_weight, eps=norm_eps)
    return F.linear(y, out_proj_weight, out_proj_bias)
