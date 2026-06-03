from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    ROTARY_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    ROTARY_TRITON_AVAILABLE = False


if ROTARY_TRITON_AVAILABLE:

    @triton.jit
    def _rotary_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        n_elements,
        H,
        T,
        D,
        INVERSE: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements

        d = offs % D
        t = (offs // D) % T
        half = D // 2
        pair_d = tl.where(d < half, d + half, d - half)
        pair_offs = offs + (pair_d - d)

        x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_pair = tl.load(x_ptr + pair_offs, mask=mask, other=0.0).to(tl.float32)
        cos = tl.load(cos_ptr + t * D + d, mask=mask, other=1.0).to(tl.float32)
        sin = tl.load(sin_ptr + t * D + d, mask=mask, other=0.0).to(tl.float32)
        sin_pair = tl.load(
            sin_ptr + t * D + pair_d,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        if INVERSE:
            out = tl.where(d < half, x * cos + x_pair * sin_pair, x * cos - x_pair * sin_pair)
        else:
            rotated = tl.where(d < half, -x_pair, x_pair)
            out = x * cos + rotated * sin
        tl.store(out_ptr + offs, out, mask=mask)


def rotary_apply_fallback(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos + rotated * sin


def _rotary_apply_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    inverse: bool = False,
) -> torch.Tensor:
    if not ROTARY_TRITON_AVAILABLE or not x.is_cuda:
        if inverse:
            return rotary_apply_fallback(x, cos, -sin)
        return rotary_apply_fallback(x, cos, sin)
    if x.ndim != 4:
        if inverse:
            return rotary_apply_fallback(x, cos, -sin)
        return rotary_apply_fallback(x, cos, sin)

    B, H, T, D = x.shape
    if D % 2 != 0:
        raise ValueError("rotary_apply expects an even head dimension")

    x_contig = x.contiguous()
    cos_contig = cos.contiguous().view(T, D)
    sin_contig = sin.contiguous().view(T, D)
    out = torch.empty_like(x_contig)

    n = x_contig.numel()
    block = 256
    grid = (triton.cdiv(n, block),)
    _rotary_kernel[grid](
        x_contig,
        cos_contig,
        sin_contig,
        out,
        n,
        H,
        T,
        D,
        INVERSE=inverse,
        BLOCK=block,
    )
    return out


class _RotaryApplyFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cos, sin):
        ctx.save_for_backward(cos, sin)
        return _rotary_apply_triton(x, cos, sin)

    @staticmethod
    def backward(ctx, grad_out):
        cos, sin = ctx.saved_tensors
        grad_x = _rotary_apply_triton(grad_out, cos, sin, inverse=True)
        return grad_x, None, None


def rotary_apply(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply half-rotary embedding over the last dimension.

    ``x`` is expected to have shape ``[B, H, T, D]`` for the Triton path.
    Other shapes fall back to regular PyTorch operations.
    """
    if ROTARY_TRITON_AVAILABLE and x.is_cuda and x.ndim == 4:
        return _RotaryApplyFn.apply(x, cos, sin)
    return rotary_apply_fallback(x, cos, sin)
