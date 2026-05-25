import math
import os

import torch
import torch.nn.functional as F

from .triton_bwd import can_use_triton_bwd, triton_flash_bwd
from .triton_fwd import can_use_triton_fwd, triton_flash_fwd

_sdpa = F.scaled_dot_product_attention


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


def _sdpa_forward(q, k, v, causal, softmax_scale):
    out = _sdpa(
        q,
        k,
        v,
        is_causal=bool(causal),
        scale=softmax_scale,
    )
    lse = torch.empty(0, device=q.device, dtype=torch.float32)
    return out, lse


def _aten_flash_forward(q, k, v, causal, softmax_scale):
    return torch.ops.aten._scaled_dot_product_flash_attention.default(
        q,
        k,
        v,
        0.0,
        bool(causal),
        False,
        scale=softmax_scale,
    )


def _aten_flash_backward(
    grad_out,
    q,
    k,
    v,
    out,
    lse,
    cum_seq_q,
    cum_seq_k,
    max_q,
    max_k,
    rng_state,
    unused,
    causal,
    softmax_scale,
):
    return torch.ops.aten._scaled_dot_product_flash_attention_backward.default(
        grad_out,
        q,
        k,
        v,
        out,
        lse,
        cum_seq_q,
        cum_seq_k,
        max_q,
        max_k,
        0.0,
        bool(causal),
        rng_state,
        unused,
        scale=softmax_scale,
    )


def _manual_backward(q, k, v, grad_out, causal, softmax_scale):
    # Correct fallback that materializes attention probabilities. Kept for
    # debugging and platforms where SDPA is unavailable or explicitly disabled.
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


def _sdpa_backward(q, k, v, grad_out, causal, softmax_scale):
    with torch.enable_grad():
        q_ = q.detach().requires_grad_(True)
        k_ = k.detach().requires_grad_(True)
        v_ = v.detach().requires_grad_(True)
        out = _sdpa(
            q_,
            k_,
            v_,
            is_causal=bool(causal),
            scale=softmax_scale,
        )
        dq, dk, dv = torch.autograd.grad(
            out, (q_, k_, v_), grad_out, retain_graph=False
        )
    return dq, dk, dv


def _reference_backward(q, k, v, grad_out, causal, softmax_scale):
    if os.environ.get("CUSTOM_ATT_DISABLE_SDPA_BWD") != "1":
        return _sdpa_backward(q, k, v, grad_out, causal, softmax_scale)
    return _manual_backward(q, k, v, grad_out, causal, softmax_scale)


def _triton_fwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_TRITON_FWD") != "1"


def _triton_bwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_TRITON_BWD") != "1"


def _sdpa_bwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_SDPA_BWD") != "1"


def _is_ada_or_newer(q):
    major, minor = torch.cuda.get_device_capability(q.device)
    return major > 8 or (major == 8 and minor >= 9)


def _can_use_aten_flash(q):
    return (
        q.is_cuda
        and hasattr(torch.ops.aten, "_scaled_dot_product_flash_attention")
        and q.dtype in (torch.float16, torch.bfloat16)
        and q.shape[-1] <= 256
    )


def _prefer_sdpa_backward(q, causal):
    if not (q.is_cuda and _sdpa_bwd_enabled()):
        return False
    if not can_use_triton_bwd(q):
        return True

    if not (
        _is_ada_or_newer(q)
        and q.dtype in (torch.float16, torch.bfloat16)
        and _can_use_aten_flash(q)
    ):
        return False

    # RTX 4090 measurements show native SDPA/ATen flash is faster or effectively
    # tied across the training shapes we sweep. Keep Triton available via
    # CUSTOM_ATT_DISABLE_SDPA_BWD=1 for kernel development comparisons.
    return True


class CustomAttFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = 0.0 if softmax_scale is None else float(softmax_scale)
        use_aten_flash = _prefer_sdpa_backward(q, bool(causal)) and _can_use_aten_flash(q)
        use_triton = (
            not use_aten_flash and _triton_fwd_enabled() and can_use_triton_fwd(q)
        )
        if use_triton:
            out, lse = triton_flash_fwd(
                q,
                k,
                v,
                causal=bool(causal),
                softmax_scale=None if scale == 0.0 else scale,
            )
            ctx.aten_flash_meta = None
            saved_tensors = (q, k, v, out, lse)
        elif use_aten_flash:
            (
                out,
                lse,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                rng_state,
                unused,
                _debug_mask,
            ) = _aten_flash_forward(
                q, k, v, bool(causal), None if scale == 0.0 else scale
            )
            ctx.aten_flash_meta = (cum_seq_q, cum_seq_k, max_q, max_k)
            saved_tensors = (q, k, v, out, lse, rng_state, unused)
        elif _prefer_sdpa_backward(q, bool(causal)):
            out, lse = _sdpa_forward(
                q, k, v, bool(causal), None if scale == 0.0 else scale
            )
            ctx.aten_flash_meta = None
            saved_tensors = (q, k, v, out, lse)
        else:
            out, lse = _reference_forward(
                q, k, v, bool(causal), None if scale == 0.0 else scale
            )
            ctx.aten_flash_meta = None
            saved_tensors = (q, k, v, out, lse)
        ctx.save_for_backward(*saved_tensors)
        ctx.causal = bool(causal)
        ctx.softmax_scale = scale if scale != 0.0 else None
        ctx.softmax_scale_arg = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        saved_tensors = ctx.saved_tensors
        q, k, v, out, lse = saved_tensors[:5]
        grad_out = grad_out.contiguous()
        use_sdpa_bwd = _prefer_sdpa_backward(q, ctx.causal)
        use_triton_bwd = (
            not use_sdpa_bwd and _triton_bwd_enabled() and can_use_triton_bwd(q)
        )
        use_aten_flash_bwd = (
            use_sdpa_bwd
            and ctx.aten_flash_meta is not None
            and len(saved_tensors) == 7
        )
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
        elif use_aten_flash_bwd:
            rng_state, unused = saved_tensors[5:]
            cum_seq_q, cum_seq_k, max_q, max_k = ctx.aten_flash_meta
            dq, dk, dv = _aten_flash_backward(
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                cum_seq_q,
                cum_seq_k,
                max_q,
                max_k,
                rng_state,
                unused,
                ctx.causal,
                ctx.softmax_scale,
            )
        elif use_sdpa_bwd:
            dq, dk, dv = _sdpa_backward(
                q, k, v, grad_out, ctx.causal, ctx.softmax_scale
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
    scale = 0.0 if softmax_scale is None else float(softmax_scale)
    if _prefer_sdpa_backward(q, bool(causal)):
        return _sdpa(
            q,
            k,
            v,
            is_causal=bool(causal),
            scale=None if scale == 0.0 else scale,
        )
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


def flash_attn_backward_backend(q, causal=False):
    """Return which backward backend ``flash_attn_func(...).backward`` will use."""
    if _prefer_sdpa_backward(q, bool(causal)):
        return "sdpa"
    if _triton_bwd_enabled() and can_use_triton_bwd(q):
        return "triton"
    if _sdpa_bwd_enabled() and q.is_cuda:
        return "sdpa"
    return "torch"
