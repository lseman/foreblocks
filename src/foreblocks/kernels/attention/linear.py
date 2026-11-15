"""Fused recurrent causal linear attention for short-sequence inference."""

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
        Out,
        InitialZ,
        InitialS,
        FinalZ,
        FinalS,
        H: tl.constexpr,
        HAS_INITIAL: tl.constexpr,
        SAVE_FINAL: tl.constexpr,
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
        o_base = Out + b * s_vb + h * s_vh

        state_base = (b * H + h) * F
        if HAS_INITIAL:
            state = tl.load(
                InitialS + state_base * DV + offs_f[:, None] * DV + offs_d[None, :],
                mask=mask_f[:, None] & mask_d[None, :],
                other=0.0,
            ).to(tl.float32)
            z = tl.load(InitialZ + state_base + offs_f, mask=mask_f, other=0.0).to(
                tl.float32
            )
        else:
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
        if SAVE_FINAL:
            tl.store(
                FinalS + state_base * DV + offs_f[:, None] * DV + offs_d[None, :],
                state,
                mask=mask_f[:, None] & mask_d[None, :],
            )
            if i_d == 0:
                tl.store(FinalZ + state_base + offs_f, z, mask=mask_f)


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
    return (
        q.device == k.device == v.device
        and 0 < f <= 128
        and 0 < dv <= 128
        and q.shape[2] > 0
    )


@torch.compiler.disable
def fused_recurrent_causal_linear_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    *,
    initial_state: tuple[torch.Tensor, torch.Tensor] | None = None,
    output_final_state: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Inference scan; state is (key sum [B,H,F], KV sum [B,H,F,Dv]).

    Returned state accumulates in fp32, never aliases the initial state, and
    can be passed to the next call. Training uses the differentiable torch scan.
    """
    if not can_use_fused_recurrent_linear_attn(q, k, v):
        raise RuntimeError("fused_recurrent_causal_linear_attn is not available")

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, H, T, F = q.shape
    DV = v.shape[-1]
    z0 = s0 = None
    if initial_state is not None:
        z0, s0 = initial_state
        if z0.shape != (B, H, F) or s0.shape != (B, H, F, DV):
            raise ValueError("initial linear attention state has incompatible shape")
        if any(
            t.device != q.device or t.dtype not in (q.dtype, torch.float32)
            for t in initial_state
        ):
            raise ValueError(
                "initial state must share the input device and use input dtype or fp32"
            )
        z0, s0 = z0.contiguous(), s0.contiguous()
    if torch.is_grad_enabled() and any(
        t.requires_grad
        for t in (q, k, v, *(() if initial_state is None else initial_state))
    ):
        raise RuntimeError("fused recurrent linear attention is inference-only")
    z_final = (
        torch.empty((B, H, F), device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    s_final = (
        torch.empty((B, H, F, DV), device=q.device, dtype=torch.float32)
        if output_final_state
        else None
    )
    BF = triton.next_power_of_2(F)
    BD = min(128, triton.next_power_of_2(DV))
    out = torch.empty_like(v)
    grid = (B, H, triton.cdiv(DV, BD))
    _causal_linear_attn_fwd_kernel[grid](
        q,
        k,
        v,
        out,
        z0,
        s0,
        z_final,
        s_final,
        H,
        initial_state is not None,
        output_final_state,
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
    return (out, (z_final, s_final)) if output_final_state else out
