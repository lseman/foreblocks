from __future__ import annotations

import torch
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    CAUSAL_CONV1D_TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    CAUSAL_CONV1D_TRITON_AVAILABLE = False


if CAUSAL_CONV1D_TRITON_AVAILABLE:

    @triton.jit
    def _causal_depthwise_conv1d_kernel(
        x_ptr,
        weight_ptr,
        bias_ptr,
        out_ptr,
        total_elements,
        D,
        T,
        K: tl.constexpr,
        has_bias: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elements

        t = offs % T
        bd = offs // T
        d = bd % D

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        if has_bias:
            acc += tl.load(bias_ptr + d, mask=mask, other=0.0).to(tl.float32)

        for k in range(0, K):
            t_in = t - (K - 1 - k)
            valid = mask & (t_in >= 0)
            x_idx = bd * T + t_in
            w_idx = d * K + k
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(weight_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            acc += x_val * w_val

        tl.store(out_ptr + offs, acc, mask=mask)


def causal_depthwise_conv1d_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if x.ndim != 3:
        raise ValueError("x must have shape [B, D, T]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [D, K]")
    if x.size(1) != weight.size(0):
        raise ValueError("channel dimension mismatch between x and weight")
    if bias is not None and bias.shape != (weight.size(0),):
        raise ValueError("bias must have shape [D]")

    kernel_size = weight.size(1)
    x_pad = F.pad(x, (kernel_size - 1, 0))
    return F.conv1d(
        x_pad,
        weight.unsqueeze(1),
        bias=bias,
        groups=x.size(1),
    )


def causal_depthwise_conv1d_triton(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if not CAUSAL_CONV1D_TRITON_AVAILABLE:
        raise RuntimeError(
            "causal_depthwise_conv1d_triton called but Triton is not available"
        )
    if x.ndim != 3:
        raise ValueError("x must have shape [B, D, T]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [D, K]")

    B, D, T = x.shape
    K = weight.size(1)
    if weight.size(0) != D:
        raise ValueError("channel dimension mismatch between x and weight")
    if bias is not None and bias.shape != (D,):
        raise ValueError("bias must have shape [D]")

    x_contig = x.contiguous()
    weight_contig = weight.contiguous()
    out = torch.empty_like(x_contig)
    total = x.numel()
    BLOCK = 256
    grid = (triton.cdiv(total, BLOCK),)
    _causal_depthwise_conv1d_kernel[grid](
        x_contig,
        weight_contig,
        bias if bias is not None else x_contig,
        out,
        total,
        D,
        T,
        K,
        has_bias=bias is not None,
        BLOCK=BLOCK,
    )
    return out


class _CausalDepthwiseConv1dFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(
            x,
            weight,
            bias
            if bias is not None
            else torch.tensor([], device=x.device, dtype=x.dtype),
        )
        ctx.has_bias = bias is not None
        return causal_depthwise_conv1d_triton(x, weight, bias=bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, bias_or_empty = ctx.saved_tensors
        bias = bias_or_empty if ctx.has_bias else None

        with torch.enable_grad():
            x_ = x.detach().requires_grad_(x.requires_grad)
            weight_ = weight.detach().requires_grad_(weight.requires_grad)
            if bias is None:
                bias_ = None
                grad_inputs = (x_, weight_)
            else:
                bias_ = bias.detach().requires_grad_(bias.requires_grad)
                grad_inputs = (x_, weight_, bias_)

            out = causal_depthwise_conv1d_reference(x_, weight_, bias_)
            grads_required = torch.autograd.grad(
                outputs=out,
                inputs=grad_inputs,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )

        if bias is None:
            dx, dweight = grads_required
            dbias = None
        else:
            dx, dweight, dbias = grads_required
        return dx, dweight, dbias


def causal_depthwise_conv1d(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if (
        CAUSAL_CONV1D_TRITON_AVAILABLE
        and x.is_cuda
        and weight.is_cuda
        and (bias is None or bias.is_cuda)
    ):
        return _CausalDepthwiseConv1dFn.apply(x, weight, bias)
    return causal_depthwise_conv1d_reference(x, weight, bias=bias)
