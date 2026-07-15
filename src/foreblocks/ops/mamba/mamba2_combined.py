"""foreblocks.ops.mamba.mamba2_combined.

Combined Mamba2 block: conv → dt prep → SSD scan → RMSNormGated → out_proj.

Keeps the layer-level dataflow in one op boundary while reusing local Triton
kernels for each substep. Supports attention masking, group repetition for
GQA, and Triton-backed SSD. Use when you need a single-call Mamba2 block
without managing intermediate tensors.

Core API:
- mamba2_split_conv1d_scan_combined: full Mamba2 block path in one call

"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from foreblocks.ops.mamba.causal_conv1d import causal_depthwise_conv1d
from foreblocks.ops.mamba.fused_dt import fused_dt
from foreblocks.ops.mamba.ssd import chunked_ssd_forward
from foreblocks.ops.mamba.triton_ops import fused_out


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
    dt_proj_weight: torch.Tensor | None,
    dt_bias: torch.Tensor,
    A_log: torch.Tensor,
    Dskip: torch.Tensor,
    norm_weight: torch.Tensor,
    out_proj_weight: torch.Tensor,
    out_proj_bias: torch.Tensor | None,
    d_inner: int,
    conv_dim: int,
    dt_rank: int | None = None,  # None → dt_rank = num_heads (direct projection)
    num_heads: int = 0,
    head_dim: int,
    n_groups: int,
    d_state: int,
    chunk_size: int,
    dt_limit: tuple[float, float] = (1e-4, 1.0),
    norm_eps: float = 1e-5,
    attention_mask: torch.Tensor | None = None,
    use_triton_ssd: bool = True,
    activation: str = "silu",
) -> torch.Tensor:
    """Combined Mamba2 block path following FLA's split/conv/scan layout.

    This keeps the layer-level dataflow in one op boundary while reusing the
    local Triton kernels for causal conv, dt prep, SSD, and gated RMSNorm.

    When dt_proj_weight is None, dt_rank defaults to num_heads (no low-rank bottleneck).
    """
    if dt_rank is None:
        dt_rank = num_heads
    z, conv_input, dt_hidden = torch.split(
        projected,
        [d_inner, conv_dim, dt_rank],
        dim=-1,
    )

    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != projected.shape[:2]:
            raise ValueError("attention_mask must have shape [B, T]")
        mask_t = attention_mask.to(dtype=projected.dtype, device=projected.device)
        conv_input = conv_input * mask_t.unsqueeze(-1)

    conv_out = causal_depthwise_conv1d(
        conv_input.transpose(1, 2),
        conv_weight,
        conv_bias,
    ).transpose(1, 2)

    if activation in ("silu", "swish"):
        conv_out = F.silu(conv_out)

    if attention_mask is not None:
        if attention_mask.ndim != 2 or attention_mask.shape != projected.shape[:2]:
            raise ValueError("attention_mask must have shape [B, T]")
        mask_t = attention_mask.to(dtype=projected.dtype, device=projected.device)
        conv_out = conv_out * mask_t.unsqueeze(-1)

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

    dt = fused_dt(
        dt_hidden, dt_proj_weight, dt_bias, dt_min=dt_limit[0], dt_max=dt_limit[1]
    )
    # dt_proj_weight=None → no projection, dt_hidden already [B, T, H]
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
        parallel="tiled",
    ).reshape(u.shape[0], u.shape[1], d_inner)
    # RMSNormGated: rms_norm(y, group_size) * silu(z) — matches official Mamba2
    group_size = d_inner // n_groups if n_groups > 1 else None
    y = fused_out(y, z, norm_weight, eps=norm_eps, group_size=group_size)
    return F.linear(y, out_proj_weight, out_proj_bias)
