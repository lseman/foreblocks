"""foreblocks.ops.attention.chunked_causal_linear_attention.

Chunk-parallel causal linear attention with fused Triton inference path.

Implements causal linear attention (softmax-free, O(T) complexity) in two
modes: a chunk-parallel implementation for training and a fused Triton kernel
for fast inference. Both use [B, H, T, *] layout and support feature-mapped
positive query/key inputs. Use when you need linear attention as an efficient
alternative to softmax attention for long sequences.

Core API:
- chunked_causal_linear_attn: chunk-parallel causal linear attention (training + inference)
- fused_recurrent_causal_linear_attn: fused Triton recurrent kernel (inference only)
- can_use_fused_recurrent_linear_attn: guard for Triton kernel availability

"""

# chunked_causal_linear_attention.py
# -----------------------------------------------------------------------------
# Chunk-parallel causal linear attention plus a fused Triton inference path.
# -----------------------------------------------------------------------------

import os

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover
    HAS_TRITON = False


if HAS_TRITON:

    @triton.jit
    def _causal_linear_attn_fwd_kernel(
        Q,
        K,
        V,
        O,
        T: tl.constexpr,
        F: tl.constexpr,
        DV: tl.constexpr,
        s_qb: tl.constexpr,
        s_qh: tl.constexpr,
        s_qt: tl.constexpr,
        s_qf: tl.constexpr,
        s_vb: tl.constexpr,
        s_vh: tl.constexpr,
        s_vt: tl.constexpr,
        s_vd: tl.constexpr,
        eps: tl.constexpr,
        BF: tl.constexpr,
        BD: tl.constexpr,
    ):
        b = tl.program_id(0)
        h = tl.program_id(1)
        i_d = tl.program_id(2)
        offs_f = tl.arange(0, BF)
        offs_d = i_d * BD + tl.arange(0, BD)
        mask_f = offs_f < F
        mask_d = offs_d < DV

        q_base = Q + b * s_qb + h * s_qh
        k_base = K + b * s_qb + h * s_qh
        v_base = V + b * s_vb + h * s_vh
        o_base = O + b * s_vb + h * s_vh

        state = tl.zeros((BF, BD), tl.float32)
        z = tl.zeros((BF,), tl.float32)

        for t in range(0, T):
            q = tl.load(
                q_base + t * s_qt + offs_f * s_qf,
                mask=mask_f,
                other=0.0,
            ).to(tl.float32)
            k = tl.load(
                k_base + t * s_qt + offs_f * s_qf,
                mask=mask_f,
                other=0.0,
            ).to(tl.float32)
            v = tl.load(
                v_base + t * s_vt + offs_d * s_vd,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)

            z += k
            state += k[:, None] * v[None, :]
            numer = tl.sum(q[:, None] * state, axis=0)
            denom = tl.sum(q * z, axis=0) + eps
            out = numer / denom

            tl.store(
                o_base + t * s_vt + offs_d * s_vd,
                out,
                mask=mask_d,
            )


def can_use_fused_recurrent_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> bool:
    if not HAS_TRITON:
        return False
    if os.environ.get("FOREBLOCKS_DISABLE_TRITON_LINEAR_ATTN", "") == "1":
        return False
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return False
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    if q.shape[:3] != v.shape[:3]:
        return False
    _, _, _, f = q.shape
    dv = v.shape[-1]
    return f <= 128 and dv <= 128


@torch.compiler.disable
def fused_recurrent_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    if not can_use_fused_recurrent_linear_attn(q, k, v):
        raise RuntimeError("fused_recurrent_causal_linear_attn is not available")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, T, F = q.shape
    DV = v.shape[-1]
    BF = triton.next_power_of_2(F)
    BD = min(128, triton.next_power_of_2(DV))
    out = torch.empty_like(v)
    grid = (B, H, triton.cdiv(DV, BD))
    _causal_linear_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        T,
        F,
        DV,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        float(eps),
        BF,
        BD,
        num_warps=4,
    )
    return out


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
