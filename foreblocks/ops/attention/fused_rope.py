"""foreblocks.ops.attention.fused_rope.

This module implements the fused rope pieces for its package.
It belongs to the attention modules, variants, caches, and utilities area of Foreblocks.
It exposes functions such as triton_apply_rope, triton_apply_rope_bthd.
"""

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
        SEQLEN_OFFSETS,
        stride_xb,
        stride_xh,
        stride_xt,
        stride_xd,
        stride_ob,
        stride_oh,
        stride_ot,
        stride_od,
        stride_cos_t,
        stride_cos_d,
        T: tl.constexpr,
        D: tl.constexpr,
        R: tl.constexpr,
        TR: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_R: tl.constexpr,
        IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
        INTERLEAVED: tl.constexpr,
        CONJUGATE: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_tb = tl.program_id(2)

        t_offs = pid_tb * BLOCK_T + tl.arange(0, BLOCK_T)
        if IS_SEQLEN_OFFSETS_TENSOR:
            cache_t_offs = t_offs + tl.load(SEQLEN_OFFSETS + pid_b)
        else:
            cache_t_offs = t_offs + SEQLEN_OFFSETS
        t_mask = (t_offs < T) & (cache_t_offs >= 0) & (cache_t_offs < TR)

        x_base = X + pid_b * stride_xb + pid_h * stride_xh
        out_base = OUT + pid_b * stride_ob + pid_h * stride_oh

        if not INTERLEAVED:
            r_offs = tl.arange(0, BLOCK_R)
            mask = t_mask[:, None] & (r_offs < R)[None, :]
            cos = tl.load(
                COS
                + cache_t_offs[:, None] * stride_cos_t
                + r_offs[None, :] * stride_cos_d,
                mask=mask,
                other=1.0,
            ).to(tl.float32)
            sin = tl.load(
                SIN
                + cache_t_offs[:, None] * stride_cos_t
                + r_offs[None, :] * stride_cos_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if CONJUGATE:
                sin = -sin

            x0_ptr = x_base + t_offs[:, None] * stride_xt + r_offs[None, :] * stride_xd
            x1_ptr = x0_ptr + R * stride_xd
            x0 = tl.load(x0_ptr, mask=mask, other=0.0).to(tl.float32)
            x1 = tl.load(x1_ptr, mask=mask, other=0.0).to(tl.float32)

            y0 = x0 * cos - x1 * sin
            y1 = x0 * sin + x1 * cos

            out0_ptr = (
                out_base + t_offs[:, None] * stride_ot + r_offs[None, :] * stride_od
            )
            out1_ptr = out0_ptr + R * stride_od

            tl.store(out0_ptr, y0, mask=mask)
            tl.store(out1_ptr, y1, mask=mask)
        else:
            d_offs = tl.arange(0, BLOCK_R * 2)
            d_swap = d_offs + ((d_offs + 1) % 2) * 2 - 1
            r_offs = d_offs // 2
            mask = t_mask[:, None] & (r_offs < R)[None, :]
            cos = tl.load(
                COS
                + cache_t_offs[:, None] * stride_cos_t
                + r_offs[None, :] * stride_cos_d,
                mask=mask,
                other=1.0,
            ).to(tl.float32)
            sin = tl.load(
                SIN
                + cache_t_offs[:, None] * stride_cos_t
                + r_offs[None, :] * stride_cos_d,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            if CONJUGATE:
                sin = -sin

            x0 = tl.load(
                x_base + t_offs[:, None] * stride_xt + d_offs[None, :] * stride_xd,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                x_base + t_offs[:, None] * stride_xt + d_swap[None, :] * stride_xd,
                mask=mask,
                other=0.0,
            ).to(tl.float32)
            y = tl.where(
                d_offs[None, :] % 2 == 0,
                x0 * cos - x1 * sin,
                x0 * cos + x1 * sin,
            )
            tl.store(
                out_base + t_offs[:, None] * stride_ot + d_offs[None, :] * stride_od,
                y,
                mask=mask,
            )


def _next_power_of_2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _default_block_t(x: torch.Tensor, T: int) -> int:
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    return min(128, _next_power_of_2(triton.cdiv(T, max(sm_count, 1))))


def _triton_apply_rope_single_bhtd(
    x: torch.Tensor,  # [B, H, T, D]
    cos: torch.Tensor,  # [T, D//2]
    sin: torch.Tensor,  # [T, D//2]
    seqlen_offset: int | torch.Tensor = 0,
    block_t: int | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError("Triton RoPE requires CUDA tensors")

    B, H, T, D = x.shape
    if D % 2 != 0:
        raise ValueError("RoPE head_dim must be even")
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("Triton RoPE expects 2D cos/sin caches")
    if cos.dtype != sin.dtype or x.dtype != cos.dtype:
        raise ValueError("x, cos, and sin must have matching dtypes")

    TR, R = cos.shape
    ro_dim = R * 2
    if ro_dim > D:
        raise ValueError("RoPE rotary dimension cannot exceed head_dim")
    if isinstance(seqlen_offset, torch.Tensor):
        if seqlen_offset.shape != (B,) or seqlen_offset.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("tensor seqlen_offset must have shape [B] and int dtype")
    elif seqlen_offset + T > TR:
        raise ValueError("cos/sin cache is shorter than seqlen_offset + T")

    out = x if inplace else torch.empty_like(x)
    if ro_dim < D and not inplace:
        out[..., ro_dim:].copy_(x[..., ro_dim:])
    block_t = _default_block_t(x, T) if block_t is None else block_t
    block_r = _next_power_of_2(R)
    grid = (B, H, triton.cdiv(T, block_t))

    _rope_apply_kernel[grid](
        x,
        cos,
        sin,
        out,
        seqlen_offset,
        stride_xb=x.stride(0),
        stride_xh=x.stride(1),
        stride_xt=x.stride(2),
        stride_xd=x.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(1),
        stride_ot=out.stride(2),
        stride_od=out.stride(3),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(1),
        T=T,
        D=D,
        R=R,
        TR=TR,
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offset, torch.Tensor),
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )
    return out


def triton_apply_rope(
    q: torch.Tensor,  # [B, H, T, D]
    k: torch.Tensor,  # [B, Hkv, T, D]
    cos: torch.Tensor,  # [T_max, D//2]
    sin: torch.Tensor,  # [T_max, D//2]
    seqlen_offset: int | torch.Tensor = 0,
    block_t: int | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("triton_apply_rope expects 2D cos/sin caches")

    out_q = _triton_apply_rope_single_bhtd(
        q, cos, sin, seqlen_offset, block_t, interleaved, inplace, conjugate
    )
    out_k = _triton_apply_rope_single_bhtd(
        k, cos, sin, seqlen_offset, block_t, interleaved, inplace, conjugate
    )
    return out_q, out_k


def triton_apply_rope_bthd(
    x: torch.Tensor,  # [B, T, H, D]
    cos: torch.Tensor,  # [T, D//2]
    sin: torch.Tensor,  # [T, D//2]
    seqlen_offset: int | torch.Tensor = 0,
    block_t: int | None = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> torch.Tensor:
    """
    Apply RoPE to x in [B, T, H, D] layout using stride remapping — no transpose.

    The underlying kernel is launched with grid (B, H, T_blocks) and the
    time/head strides are swapped relative to the BHTD variant, avoiding
    any extra memory copies.
    """
    if not _TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available")
    if not x.is_cuda:
        raise RuntimeError("Triton RoPE requires CUDA tensors")
    if cos.dim() != 2 or sin.dim() != 2:
        raise ValueError("triton_apply_rope_bthd expects 2D cos/sin caches")

    B, T, H, D = x.shape
    if D % 2 != 0:
        raise ValueError("RoPE head_dim must be even")

    if cos.dtype != sin.dtype or x.dtype != cos.dtype:
        raise ValueError("x, cos, and sin must have matching dtypes")

    TR, R = cos.shape
    ro_dim = R * 2
    if ro_dim > D:
        raise ValueError("RoPE rotary dimension cannot exceed head_dim")
    if isinstance(seqlen_offset, torch.Tensor):
        if seqlen_offset.shape != (B,) or seqlen_offset.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("tensor seqlen_offset must have shape [B] and int dtype")
    elif seqlen_offset + T > TR:
        raise ValueError("cos/sin cache is shorter than seqlen_offset + T")

    out = x if inplace else torch.empty_like(x)
    if ro_dim < D and not inplace:
        out[..., ro_dim:].copy_(x[..., ro_dim:])
    block_t = _default_block_t(x, T) if block_t is None else block_t
    block_r = _next_power_of_2(R)
    grid = (B, H, triton.cdiv(T, block_t))

    _rope_apply_kernel[grid](
        x,
        cos,
        sin,
        out,
        seqlen_offset,
        stride_xb=x.stride(0),
        stride_xh=x.stride(2),  # head is dim-2 in BTHD
        stride_xt=x.stride(1),  # time is dim-1 in BTHD
        stride_xd=x.stride(3),
        stride_ob=out.stride(0),
        stride_oh=out.stride(2),
        stride_ot=out.stride(1),
        stride_od=out.stride(3),
        stride_cos_t=cos.stride(0),
        stride_cos_d=cos.stride(1),
        T=T,
        D=D,
        R=R,
        TR=TR,
        BLOCK_T=block_t,
        BLOCK_R=block_r,
        IS_SEQLEN_OFFSETS_TENSOR=isinstance(seqlen_offset, torch.Tensor),
        INTERLEAVED=interleaved,
        CONJUGATE=conjugate,
    )
    return out
