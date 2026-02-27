from typing import Tuple

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _rope_apply_kernel(
        X,
        COS,
        SIN,
        OUT,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_xd,
        stride_cos_t,
        stride_cos_d,
        T: tl.constexpr,
        D: tl.constexpr,
        HALF_D: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_tb = tl.program_id(2)

        t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        d_offs = tl.arange(0, HALF_D)
        t_mask = t_offs < T

        cos = tl.load(
            COS + t_offs[:, None] * stride_cos_t + d_offs[None, :] * stride_cos_d,
            mask=t_mask[:, None],
            other=1.0,
        )
        sin = tl.load(
            SIN + t_offs[:, None] * stride_cos_t + d_offs[None, :] * stride_cos_d,
            mask=t_mask[:, None],
            other=0.0,
        )

        x_ptr = (
            X
            + pid_b * stride_xb
            + pid_h * stride_xh
            + t_offs[:, None] * stride_xt
            + d_offs[None, :] * stride_xd
        )
        xr_ptr = x_ptr + HALF_D * stride_xd

        x = tl.load(x_ptr, mask=t_mask[:, None], other=0.0)
        xr = tl.load(xr_ptr, mask=t_mask[:, None], other=0.0)

        out_x = x * cos - xr * sin
        out_xr = xr * cos + x * sin

        out_ptr = (
            OUT
            + pid_b * stride_xb
            + pid_h * stride_xh
            + t_offs[:, None] * stride_xt
            + d_offs[None, :] * stride_xd
        )
        out_r_ptr = out_ptr + HALF_D * stride_xd

        tl.store(out_ptr, out_x, mask=t_mask[:, None])
        tl.store(out_r_ptr, out_xr, mask=t_mask[:, None])


def _triton_apply_rope_single(
    x: torch.Tensor,  # [B, H, T, D]
    cos: torch.Tensor,  # [T, D//2]
    sin: torch.Tensor,  # [T, D//2]
    block_t: int,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError("Triton RoPE requires CUDA tensors")

    B, H, T, D = x.shape
    if D % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    out = torch.empty_like(x)
    half_d = D // 2
    grid = (B, H, triton.cdiv(T, block_t))

    _rope_apply_kernel[grid](
        x,
        cos,
        sin,
        out,
        stride_xb=x.stride(0),
        stride_xh=x.stride(1),
        stride_xt=x.stride(2),
        stride_xd=x.stride(3),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(1),
        T=T,
        D=D,
        HALF_D=half_d,
        BLOCK_T=block_t,
    )
    return out


def triton_apply_rope(
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, Hkv, T, D]
    cos: torch.Tensor,  # [T_max, D//2]
    sin: torch.Tensor,  # [T_max, D//2]
    seqlen_offset: int = 0,
    block_t: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("triton_apply_rope expects 2D cos/sin caches")

    T = q.size(2)
    cos_s = cos[seqlen_offset : seqlen_offset + T].contiguous()
    sin_s = sin[seqlen_offset : seqlen_offset + T].contiguous()

    out_q = _triton_apply_rope_single(q, cos_s, sin_s, block_t)
    out_k = _triton_apply_rope_single(k, cos_s, sin_s, block_t)
    return out_q, out_k
