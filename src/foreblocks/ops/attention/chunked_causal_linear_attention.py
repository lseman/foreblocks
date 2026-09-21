"""Chunk-parallel causal linear attention with optional accelerated dispatch."""

import torch

from foreblocks.kernels.attention.linear import (
    HAS_TRITON,
    can_use_fused_recurrent_linear_attn,
    fused_recurrent_causal_linear_attn,
)


def chunked_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    # The fused kernel is recurrent (sequential over T), so it only wins for
    # short sequences / decode; the chunk-parallel path wins for prefill.
    # Crossover measured at T~64-128 (fused 2x slower by T=512).
    if (
        q.shape[2] <= 64
        and not torch.is_grad_enabled()
        and can_use_fused_recurrent_linear_attn(q, k, v)
    ):
        return fused_recurrent_causal_linear_attn(q, k, v, eps=eps)

    B, H, T, F = q.shape
    Dh = v.shape[-1]
    C = chunk_size

    # Accumulate in at least fp32 for stability, but never downgrade fp64 input.
    acc_dtype = torch.promote_types(q.dtype, torch.float32)

    S = q.new_zeros(B, H, F, Dh, dtype=acc_dtype)
    z = q.new_zeros(B, H, F, dtype=acc_dtype)
    tri = torch.tril(torch.ones(C, C, device=q.device, dtype=acc_dtype))

    outs = []
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        qi = q[:, :, s:e].to(acc_dtype)
        ki = k[:, :, s:e].to(acc_dtype)
        vi = v[:, :, s:e].to(acc_dtype)

        num_inter = qi @ S
        den_inter = qi @ z.unsqueeze(-1)

        A = (qi @ ki.transpose(-1, -2)) * tri[:c, :c]
        num_intra = A @ vi
        den_intra = A.sum(-1, keepdim=True)

        out_i = (num_inter + num_intra) / (den_inter + den_intra + eps)
        outs.append(out_i.to(q.dtype))

        S = S + ki.transpose(-1, -2) @ vi
        z = z + ki.sum(2)

    return torch.cat(outs, dim=2)


__all__ = [
    "HAS_TRITON",
    "can_use_fused_recurrent_linear_attn",
    "chunked_causal_linear_attn",
    "fused_recurrent_causal_linear_attn",
]
