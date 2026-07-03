"""foreblocks.ops.mamba.causal_conv1d.

This module implements the causal conv1d pieces for its package.
It belongs to the Mamba and state-space operator kernels area of Foreblocks.
It exposes functions such as causal_depthwise_conv1d_reference, causal_depthwise_conv1d_triton, causal_depthwise_conv1d_bwd_triton, causal_depthwise_conv1d.
"""

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

    @triton.jit
    def _causal_depthwise_conv1d_dx_kernel(
        grad_ptr,
        weight_ptr,
        dx_ptr,
        total_elements,
        D,
        T,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < total_elements

        t = offs % T
        bd = offs // T
        d = bd % D

        acc = tl.zeros([BLOCK], dtype=tl.float32)
        for k in range(0, K):
            t_out = t + (K - 1 - k)
            valid = mask & (t_out < T)
            grad_idx = bd * T + t_out
            w_idx = d * K + k
            grad_val = tl.load(grad_ptr + grad_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(weight_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)
            acc += grad_val * w_val

        tl.store(dx_ptr + offs, acc, mask=mask)

    @triton.jit
    def _causal_depthwise_conv1d_dwdb_kernel(
        x_ptr,
        grad_ptr,
        dweight_ptr,
        dbias_ptr,
        B,
        D,
        T,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        COMPUTE_DWEIGHT: tl.constexpr,
        COMPUTE_DBIAS: tl.constexpr,
        BLOCK_BT: tl.constexpr,
    ):
        pid_d = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        offs = tl.arange(0, BLOCK_BT)
        total_bt = B * T
        acc_w = tl.zeros([BLOCK_BT], dtype=tl.float32)
        acc_b = tl.zeros([BLOCK_BT], dtype=tl.float32)

        for base in range(0, total_bt, BLOCK_BT):
            bt = base + offs
            mask = bt < total_bt
            b = bt // T
            t = bt % T
            grad_idx = (b * D + pid_d) * T + t
            grad_val = tl.load(grad_ptr + grad_idx, mask=mask, other=0.0).to(tl.float32)

            if COMPUTE_DWEIGHT:
                t_in = t - (K - 1 - pid_k)
                valid = mask & (t_in >= 0)
                x_idx = (b * D + pid_d) * T + t_in
                x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
                acc_w += grad_val * x_val

            if HAS_BIAS and COMPUTE_DBIAS and pid_k == 0:
                acc_b += grad_val

        if COMPUTE_DWEIGHT:
            tl.store(dweight_ptr + pid_d * K + pid_k, tl.sum(acc_w, axis=0))
        if HAS_BIAS and COMPUTE_DBIAS and pid_k == 0:
            tl.store(dbias_ptr + pid_d, tl.sum(acc_b, axis=0))


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
    # larger block → more ILP; cap at 1024 to stay within register limits
    BLOCK = min(max(triton.next_power_of_2(T), 256), 1024)
    num_warps = min(BLOCK // 32, 8)
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
        num_warps=num_warps,
    )
    return out


def causal_depthwise_conv1d_bwd_triton(
    grad_out: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    needs_input_grad: tuple[bool, bool, bool] = (True, True, True),
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if not CAUSAL_CONV1D_TRITON_AVAILABLE:
        raise RuntimeError(
            "causal_depthwise_conv1d_bwd_triton called but Triton is not available"
        )
    if x.ndim != 3 or grad_out.ndim != 3:
        raise ValueError("x and grad_out must have shape [B, D, T]")
    if weight.ndim != 2:
        raise ValueError("weight must have shape [D, K]")
    if x.shape != grad_out.shape:
        raise ValueError("grad_out shape must match x")

    B, D, T = x.shape
    K = weight.size(1)
    if weight.size(0) != D:
        raise ValueError("channel dimension mismatch between x and weight")
    if bias is not None and bias.shape != (D,):
        raise ValueError("bias must have shape [D]")

    x_contig = x.contiguous()
    grad_contig = grad_out.contiguous()
    weight_contig = weight.contiguous()

    dx = torch.empty_like(x_contig) if needs_input_grad[0] else None
    dweight = torch.empty_like(weight_contig) if needs_input_grad[1] else None
    dbias = torch.empty_like(bias) if (bias is not None and needs_input_grad[2]) else None

    if dx is not None:
        block = min(max(triton.next_power_of_2(T), 256), 1024)
        num_warps = min(block // 32, 8)
        grid = (triton.cdiv(x_contig.numel(), block),)
        _causal_depthwise_conv1d_dx_kernel[grid](
            grad_contig,
            weight_contig,
            dx,
            x_contig.numel(),
            D,
            T,
            K,
            BLOCK=block,
            num_warps=num_warps,
        )

    if dweight is not None or dbias is not None:
        block_bt = min(max(triton.next_power_of_2(B * T), 16), 1024)
        grid = (D, K)
        _causal_depthwise_conv1d_dwdb_kernel[grid](
            x_contig,
            grad_contig,
            dweight if dweight is not None else weight_contig,
            dbias if dbias is not None else weight_contig,
            B,
            D,
            T,
            K,
            HAS_BIAS=bias is not None,
            COMPUTE_DWEIGHT=dweight is not None,
            COMPUTE_DBIAS=dbias is not None,
            BLOCK_BT=block_bt,
        )

    return dx, dweight, dbias


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
        return causal_depthwise_conv1d_bwd_triton(
            grad_out,
            x,
            weight,
            bias=bias,
            needs_input_grad=ctx.needs_input_grad,
        )


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
