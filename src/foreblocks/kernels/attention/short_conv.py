"""Streaming short depthwise convolution with fused inference activation/cache."""

import os

import torch
import torch.nn.functional as F

from foreblocks.kernels.mamba.causal_conv1d import (
    CAUSAL_CONV1D_TRITON_AVAILABLE,
    causal_depthwise_conv1d,
)

if CAUSAL_CONV1D_TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _streaming_short_conv_kernel(
        X,
        History,
        W,
        Bias,
        Y,
        NextHistory,
        TOTAL: tl.constexpr,
        T: tl.constexpr,
        D: tl.constexpr,
        K: tl.constexpr,
        HAS_HISTORY: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        SILU: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < TOTAL
        d = offsets % D
        t = (offsets // D) % T
        b = offsets // (T * D)
        acc = tl.zeros((BLOCK,), tl.float32)
        if HAS_BIAS:
            acc = tl.load(Bias + d, valid, other=0).to(tl.float32)
        for j in tl.static_range(K):
            source = t - (K - 1) + j
            value = tl.load(
                X + (b * T + source) * D + d, valid & (source >= 0), other=0
            ).to(tl.float32)
            if HAS_HISTORY:
                value += tl.load(
                    History + (b * D + d) * (K - 1) + source + (K - 1),
                    valid & (source < 0),
                    other=0,
                ).to(tl.float32)
            weight = tl.load(W + d * K + j, valid, other=0).to(tl.float32)
            acc += value * weight
        # Round the convolution like the eager conv output before activation.
        acc = acc.to(Y.dtype.element_ty).to(tl.float32)
        if SILU:
            acc = acc * tl.sigmoid(acc)
        tl.store(Y + offsets, acc, valid)
        for j in tl.static_range(K - 1):
            source = T - (K - 1) + j
            value = tl.load(
                X + (b * T + source) * D + d, valid & (t == 0) & (source >= 0), other=0
            )
            if HAS_HISTORY:
                value += tl.load(
                    History + (b * D + d) * (K - 1) + source + (K - 1),
                    valid & (t == 0) & (source < 0),
                    other=0,
                )
            tl.store(NextHistory + (b * D + d) * (K - 1) + j, value, valid & (t == 0))


def streaming_short_conv(x, weight, bias=None, history=None, *, activation=None):
    """Return (output [B,T,D], raw history [B,D,K-1]); never mutate inputs.

    Inference fuses history reads/writes, convolution, and optional SiLU.
    Training reuses the differentiable Mamba convolution on the joined history.
    """
    if x.ndim != 3 or weight.ndim != 2 or x.size(-1) != weight.size(0):
        raise ValueError("expected x [B,T,D] and weight [D,K]")
    if activation not in (None, "silu"):
        raise ValueError("short convolution activation must be None or silu")
    B, T, D = x.shape
    K = weight.size(1)
    if min(B, T, D, K) <= 0:
        raise ValueError("short convolution dimensions must be positive")
    if history is not None and (
        history.shape != (B, D, K - 1)
        or history.device != x.device
        or history.dtype != x.dtype
    ):
        raise ValueError("history must match input dtype/device and shape [B,D,K-1]")
    if bias is not None and bias.shape != (D,):
        raise ValueError("bias must have shape [D]")
    accelerated = (
        CAUSAL_CONV1D_TRITON_AVAILABLE
        and x.is_cuda
        and weight.device == x.device
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype == x.dtype
        and K <= 16
        and (bias is None or (bias.device == x.device and bias.dtype == x.dtype))
        and os.environ.get("FOREBLOCKS_DISABLE_TRITON_SHORT_CONV") != "1"
    )
    if accelerated and not torch.is_grad_enabled():
        x, weight = x.contiguous(), weight.contiguous()
        history = history.contiguous() if history is not None else None
        bias = bias.contiguous() if bias is not None else None
        output = torch.empty_like(x)
        next_history = x.new_empty(B, D, K - 1)
        _streaming_short_conv_kernel[(triton.cdiv(x.numel(), 256),)](
            x,
            history,
            weight,
            bias,
            output,
            next_history,
            x.numel(),
            T,
            D,
            K,
            history is not None,
            bias is not None,
            activation == "silu",
            256,
        )
        return output, next_history
    inputs = x.transpose(1, 2)
    if history is None:
        history = inputs.new_zeros(B, D, K - 1)
    joined = torch.cat((history, inputs), dim=-1)
    if accelerated:
        output = causal_depthwise_conv1d(joined, weight, bias)[..., K - 1 :]
    else:
        output = F.conv1d(joined, weight.unsqueeze(1), bias, groups=D)
    next_history = joined[..., -(K - 1) :].clone() if K > 1 else joined[..., :0]
    output = output.transpose(1, 2).contiguous()
    return (F.silu(output) if activation == "silu" else output), next_history
