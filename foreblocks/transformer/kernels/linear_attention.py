# linear_attention.py
# -----------------------------------------------------------------------------
# Chunk-parallel causal linear attention (pure PyTorch, differentiable).
# -----------------------------------------------------------------------------

import torch

try:
    import triton  # noqa: F401

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


# ===================== Chunk-parallel causal linear attention =================
# Differentiable (autograd) chunked scan. Replaces the O(B·H·T·F·Dh) cumsum
# fallback for training and the sequential Triton scan for long-sequence
# inference. Memory is O(B·H·(T·Dh + F·Dh)); compute is T/C dense matmuls.
#
# Reproduces the inclusive-cumsum semantics exactly:
#   num[t,d] = Σ_f q[t,f] · Σ_{s≤t} k[s,f]·v[s,d]
#   den[t]   = Σ_f q[t,f] · Σ_{s≤t} k[s,f]
#   out      = num / (den + eps)


def chunked_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    chunk_size: int = 128,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Chunk-parallel causal linear attention.

    Args:
        q, k: [B, H, T, F]  (already feature-mapped, positive)
        v:    [B, H, T, Dh]
        chunk_size: chunk length C; T is split into ceil(T/C) chunks.
        eps:  denominator stabiliser.

    Returns:
        out: [B, H, T, Dh]

    Differentiable via autograd; works in both training and inference.
    """
    B, H, T, F = q.shape
    Dh = v.shape[-1]
    C = chunk_size

    # Accumulate in at least fp32 for stability, but never downgrade fp64 input.
    acc_dtype = torch.promote_types(q.dtype, torch.float32)

    S = q.new_zeros(B, H, F, Dh, dtype=acc_dtype)  # running Σ kᵀv
    z = q.new_zeros(B, H, F, dtype=acc_dtype)      # running Σ k
    tri = torch.tril(torch.ones(C, C, device=q.device, dtype=acc_dtype))

    outs = []
    for s in range(0, T, C):
        e = min(s + C, T)
        c = e - s
        qi = q[:, :, s:e].to(acc_dtype)
        ki = k[:, :, s:e].to(acc_dtype)
        vi = v[:, :, s:e].to(acc_dtype)

        # inter-chunk: contribution of all previous chunks via the state
        num_inter = qi @ S                          # [B,H,c,Dh]
        den_inter = qi @ z.unsqueeze(-1)            # [B,H,c,1]

        # intra-chunk: causal block within this chunk (inclusive diagonal)
        A = (qi @ ki.transpose(-1, -2)) * tri[:c, :c]   # [B,H,c,c]
        num_intra = A @ vi                          # [B,H,c,Dh]
        den_intra = A.sum(-1, keepdim=True)         # [B,H,c,1]

        out_i = (num_inter + num_intra) / (den_inter + den_intra + eps)
        outs.append(out_i.to(q.dtype))

        # advance state with this chunk
        S = S + ki.transpose(-1, -2) @ vi
        z = z + ki.sum(2)

    return torch.cat(outs, dim=2)


__all__ = [
    "HAS_TRITON",
    "chunked_causal_linear_attn",
]
