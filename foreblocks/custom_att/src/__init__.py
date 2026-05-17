import math
import os

import torch

from .triton_bwd import can_use_triton_bwd, triton_flash_bwd
from .triton_fwd import can_use_triton_fwd, triton_flash_fwd


def _reference_forward(q, k, v, causal, softmax_scale):
    scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
    scores = torch.matmul(q, k.transpose(-1, -2)) * scale
    if causal:
        n = q.shape[-2]
        mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, v)
    lse = torch.logsumexp(scores.float(), dim=-1)
    return out, lse


def _reference_backward(q, k, v, grad_out, causal, softmax_scale):
    # Correct autograd fallback. This materializes attention probabilities; use for validation/research.
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        scale = softmax_scale or (1.0 / math.sqrt(q.shape[-1]))
        scores = torch.matmul(q_, k_.transpose(-1, -2)) * scale
        if causal:
            n = q.shape[-2]
            mask = torch.ones((n, n), device=q.device, dtype=torch.bool).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v_)
        dq, dk, dv = torch.autograd.grad(
            out, (q_, k_, v_), grad_out, retain_graph=False
        )
    return dq, dk, dv


def _triton_fwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_TRITON_FWD") != "1"


def _triton_bwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_TRITON_BWD") != "1"


class CustomAttFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = 0.0 if softmax_scale is None else float(softmax_scale)
        use_triton = _triton_fwd_enabled() and can_use_triton_fwd(q)
        if use_triton:
            out, lse = triton_flash_fwd(
                q,
                k,
                v,
                causal=bool(causal),
                softmax_scale=None if scale == 0.0 else scale,
            )
        else:
            out, lse = _reference_forward(
                q, k, v, bool(causal), None if scale == 0.0 else scale
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = scale if scale != 0.0 else None
        ctx.softmax_scale_arg = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        use_triton_bwd = _triton_bwd_enabled() and can_use_triton_bwd(q)
        if use_triton_bwd:
            dq, dk, dv = triton_flash_bwd(
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        else:
            dq, dk, dv = _reference_backward(
                q, k, v, grad_out, ctx.causal, ctx.softmax_scale
            )
        return dq, dk, dv, None, None


def flash_attn_func(q, k, v, causal=False, softmax_scale=None):
    """Exact attention with Triton-first forward/backward and torch fallbacks.

    Args:
        q,k,v: contiguous or strided tensors of shape [B,H,N,D]. D in {16,32,64,96,128,256}.
        causal: lower-triangular causal masking.
        softmax_scale: defaults to 1/sqrt(D).
    """
    return CustomAttFunction.apply(q, k, v, causal, softmax_scale)


def flash_attn_forward(q, k, v, causal=False, softmax_scale=None):
    scale = 0.0 if softmax_scale is None else float(softmax_scale)
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if _triton_fwd_enabled() and can_use_triton_fwd(q):
        return triton_flash_fwd(
            q,
            k,
            v,
            causal=bool(causal),
            softmax_scale=None if scale == 0.0 else scale,
        )
    return _reference_forward(q, k, v, bool(causal), None if scale == 0.0 else scale)


def flash_attn_forward_backend(q):
    """Return which forward backend ``flash_attn_func`` will use."""
    if _triton_fwd_enabled() and can_use_triton_fwd(q):
        return "triton"
    return "torch"


def flash_attn_uses_cuda_backward(q):
    """Retained for compatibility; custom_att no longer ships a CUDA extension."""
    return False


def flash_attn_backward_backend(q):
    """Return which backward backend ``flash_attn_func(...).backward`` will use."""
    if _triton_bwd_enabled() and can_use_triton_bwd(q):
        return "triton"
    return "torch"
