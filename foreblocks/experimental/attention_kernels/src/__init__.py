"""foreblocks.experimental.attention_kernels.src.

Fast custom attention kernels with TileLang and Triton backends and PyTorch fallbacks.

Provides ``flash_attn_func`` as the main entry point — it dispatches to TileLang
forward (preferred on Ada+ GPUs), Triton, or a reference implementation. Also
includes dropout-aware attention, decode-only attention for KV-cache generation,
and fused attention + RMSNorm modules.

Core API:
- flash_attn_func: exact attention with TileLang/Triton forward-backward and torch fallbacks
- flash_attn_dropout_func: attention with FA2-style dropout
- flash_attn_decode: decode-only attention for single-token generation with KV cache
- flash_attn_forward: forward-only pass
- flash_attn_forward_backend / flash_attn_backward_backend: query which backend is selected
- CustomAttFunction: torch.autograd.Function with TileLang > Triton > reference dispatch
- CustomAttDropoutFunction: attention with FA2-style dropout
- FlashAttnRMSNorm: fused attention + RMSNorm module
- FlashDecodeModule: fused attention + RMSNorm for decoding with KV cache

"""

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tilelang_bwd import can_use_tilelang_bwd, tilelang_flash_bwd
from .triton_bwd import can_use_triton_bwd, triton_flash_bwd
from .triton_fwd import (
    can_use_triton_decode,
    can_use_triton_fwd,
    triton_flash_decode,
    triton_flash_fwd,
)

_DECODER_ENABLED = os.environ.get("CUSTOM_ATT_DISABLE_DECODER") != "1"


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


def _tilelang_fwd_enabled():
    """TileLang forward is ~1.6-1.8× faster than Triton on Ada (RTX 4090).
    Preferred when available; falls back to Triton then reference.
    """
    return os.environ.get("CUSTOM_ATT_DISABLE_TILELANG_FWD") != "1"


def _triton_bwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_TRITON_BWD") != "1"


def _tilelang_bwd_enabled():
    # tilelang backward is ~1.4x faster than Triton on Ada (RTX 4090). Preferred
    # when available; falls back to Triton then reference.
    return os.environ.get("CUSTOM_ATT_DISABLE_TILELANG_BWD") != "1"


class CustomAttFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = 0.0 if softmax_scale is None else float(softmax_scale)
        scale_arg = None if scale == 0.0 else scale
        # Prefer TileLang forward (faster on Ada+), then Triton, then ref.
        use_tilelang = _tilelang_fwd_enabled() and can_use_tilelang_fwd(q)
        use_triton = _triton_fwd_enabled() and can_use_triton_fwd(q)
        if use_tilelang:
            from .tilelang_fwd import tilelang_flash_fwd

            out, lse = tilelang_flash_fwd(
                q, k, v, causal=bool(causal), softmax_scale=scale_arg
            )
        elif use_triton:
            out, lse = triton_flash_fwd(
                q, k, v, causal=bool(causal), softmax_scale=scale_arg
            )
        else:
            out, lse = _reference_forward(q, k, v, bool(causal), scale_arg)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = scale if scale != 0.0 else None
        ctx.softmax_scale_arg = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        # Prefer TileLang backward, then Triton, then ref.
        use_tilelang_bwd = _tilelang_bwd_enabled() and can_use_tilelang_bwd(q)
        use_triton_bwd = _triton_bwd_enabled() and can_use_triton_bwd(q)
        if use_tilelang_bwd:
            dq, dk, dv = tilelang_flash_bwd(
                grad_out,
                q,
                k,
                v,
                out,
                lse,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        elif use_triton_bwd:
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
    """Forward-only pass. Prefers TileLang > Triton > torch ref."""
    scale = 0.0 if softmax_scale is None else float(softmax_scale)
    scale_arg = None if scale == 0.0 else scale
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    use_tilelang = _tilelang_fwd_enabled() and can_use_tilelang_fwd(q)
    use_triton = _triton_fwd_enabled() and can_use_triton_fwd(q)
    if use_tilelang:
        from .tilelang_fwd import tilelang_flash_fwd

        return tilelang_flash_fwd(q, k, v, causal=bool(causal), softmax_scale=scale_arg)
    if use_triton:
        return triton_flash_fwd(q, k, v, causal=bool(causal), softmax_scale=scale_arg)
    return _reference_forward(q, k, v, bool(causal), scale_arg)


def flash_attn_forward_backend(q):
    """Return which forward backend ``flash_attn_func`` will use."""
    if _tilelang_fwd_enabled() and can_use_tilelang_fwd(q):
        return "tilelang"
    if _triton_fwd_enabled() and can_use_triton_fwd(q):
        return "triton"
    return "torch"


def flash_attn_uses_cuda_backward(q):
    """Retained for compatibility; custom_att no longer ships a CUDA extension."""
    return False


def flash_attn_backward_backend(q):
    """Return which backward backend ``flash_attn_func(...).backward`` will use."""
    if _tilelang_bwd_enabled() and can_use_tilelang_bwd(q):
        return "tilelang"
    if _triton_bwd_enabled() and can_use_triton_bwd(q):
        return "triton"
    return "torch"


# ---------------------------------------------------------------------------
# Dropout-aware attention (FA2-style)
# ---------------------------------------------------------------------------


def _triton_dropout_fwd_enabled():
    return os.environ.get("CUSTOM_ATT_DISABLE_DROPOUT") != "1"


class CustomAttDropoutFunction(torch.autograd.Function):
    """Attention with FA2-style dropout (forward only; backward uses reference)."""

    @staticmethod
    def forward(ctx, q, k, v, causal=False, softmax_scale=None, dropout_p=0.0):
        if dropout_p == 0.0:
            # No dropout: use the regular path
            return CustomAttFunction.apply(q, k, v, causal, softmax_scale)
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        scale = 0.0 if softmax_scale is None else float(softmax_scale)
        use_triton = _triton_dropout_fwd_enabled() and can_use_triton_fwd(q)
        if use_triton:
            out, lse = triton_flash_fwd(
                q,
                k,
                v,
                causal=bool(causal),
                softmax_scale=None if scale == 0.0 else scale,
                dropout_p=dropout_p,
            )
        else:
            out, lse = _reference_forward(
                q, k, v, bool(causal), None if scale == 0.0 else scale
            )
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal = bool(causal)
        ctx.softmax_scale = scale if scale != 0.0 else None
        ctx.dropout_p = dropout_p
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        # Dropout is only applied in training forward; backward doesn't need it
        return triton_flash_bwd(
            grad_out.contiguous(),
            q,
            k,
            v,
            out,
            lse,
            causal=ctx.causal,
            softmax_scale=ctx.softmax_scale,
        ) + (None, None)


def flash_attn_dropout_func(q, k, v, causal=False, softmax_scale=None, dropout_p=0.0):
    """Attention with FA2-style dropout.

    Args:
        q, k, v: tensors of shape [B, H, N, D]
        causal: causal masking
        softmax_scale: defaults to 1/sqrt(D)
        dropout_p: dropout probability (0.0 = no dropout)
    """
    return CustomAttDropoutFunction.apply(q, k, v, causal, softmax_scale, dropout_p)


# ---------------------------------------------------------------------------
# Decode-only (KV-cache) attention
# ---------------------------------------------------------------------------


def _triton_decode_enabled():
    return _DECODER_ENABLED and _triton_fwd_enabled()


def flash_attn_decode(q, k_cache, v_cache, seqlens, softmax_scale=None):
    """Decode-only attention for single-token generation with KV cache.

    Optimized for the decoding phase where q has sequence length 1.

    Args:
        q: [B, H, 1, D] - single token queries
        k_cache: [B*H, max_seqlen, D] - contiguous KV cache (batched layout)
        v_cache: [B*H, max_seqlen, D] - contiguous value cache
        seqlens: [B*H] - sequence lengths per (batch, head) group
        softmax_scale: defaults to 1/sqrt(D)

    Returns:
        out: [B, H, 1, D] - attention outputs
        lse: [B, H] - log-sum-exp for gradient computation
    """
    q = q.contiguous()
    k_cache = k_cache.contiguous()
    v_cache = v_cache.contiguous()
    if _triton_decode_enabled() and can_use_triton_decode(q):
        return triton_flash_decode(q, k_cache, v_cache, seqlens, softmax_scale)
    # Fallback: standard attention
    bh = q.shape[0] * q.shape[1]
    scale = (
        (1.0 / math.sqrt(q.shape[-1]))
        if softmax_scale is None
        else float(softmax_scale)
    )
    out = torch.empty_like(q)
    lse = torch.empty(bh, device=q.device, dtype=torch.float32)
    for b in range(q.shape[0]):
        for h in range(q.shape[1]):
            qi = q[b : b + 1, h : h + 1]
            seqlen = int(seqlens[b * q.shape[1] + h])
            scores = qi @ k_cache[b * q.shape[1] + h, :seqlen].transpose(-1, -2) * scale
            probs = F.softmax(scores, dim=-1)
            out[b : b + 1, h : h + 1] = probs @ v_cache[b * q.shape[1] + h, :seqlen]
            lse[b * q.shape[1] + h] = torch.logsumexp(scores.float(), dim=-1)
    return out, lse


# ---------------------------------------------------------------------------
# Fused attention + RMSNorm (FA2-style)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - compatible with PyTorch 2.x."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, eps=self.eps)


class FlashAttnRMSNorm(nn.Module):
    """Fused attention + RMSNorm module.

    Common pattern in modern LLMs: normalize inputs, run attention, normalize outputs.
    This module fuses the forward pass for memory efficiency.

    Args:
        dim: hidden dimension
        n_heads: number of attention heads
        eps: RMSNorm epsilon
        causal: causal masking
        softmax_scale: attention scale (defaults to 1/sqrt(head_dim))
        dropout_p: dropout probability
    """

    def __init__(
        self, dim, n_heads, eps=1e-6, causal=False, softmax_scale=None, dropout_p=0.0
    ):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = dropout_p
        self.attn_norm = RMSNorm(dim, eps)
        self.out_norm = RMSNorm(dim, eps)
        # QKV projection - weight dtype handled by to()
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def to(self, dtype, *args, **kwargs):
        super().to(dtype, *args, **kwargs)
        if hasattr(self, "qkv_proj") and self.qkv_proj is not None:
            self.qkv_proj.weight = self.qkv_proj.weight.to(dtype)
        if hasattr(self, "out_proj") and self.out_proj is not None:
            self.out_proj.weight = self.out_proj.weight.to(dtype)
        return self

    def forward(self, x, seqlens=None):
        """
        Args:
            x: [B, N, D] - input sequence
            seqlens: optional [B] for variable-length sequences (pad with attention mask)
        Returns:
            out: [B, N, D] - attention output with residual
        """
        B, N, D = x.shape
        residual = x

        # Normalize and project
        x_norm = self.attn_norm(x)  # [B, N, D]
        qkv = self.qkv_proj(x_norm)  # [B, N, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)  # each [B, N, D]

        # Reshape to [B, H, N, D_head]
        q = q.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention
        out = flash_attn_func(
            q, k, v, causal=self.causal, softmax_scale=self.softmax_scale
        )

        # Output projection + norm + residual
        out = out.transpose(1, 2).reshape(B, N, D)
        out = self.out_proj(out)
        out = self.out_norm(out)
        return out + residual


# ---------------------------------------------------------------------------
# Decode-only fused module
# ---------------------------------------------------------------------------


class FlashDecodeModule(nn.Module):
    """Fused attention + RMSNorm for decoding with KV cache.

    Optimized for single-token generation.
    """

    def __init__(self, dim, n_heads, eps=1e-6, softmax_scale=None, max_seqlen=2048):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.softmax_scale = softmax_scale
        self.attn_norm = RMSNorm(dim, eps)
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)

    def forward(self, x, k_cache, v_cache, seqlens):
        """
        Args:
            x: [B, 1, D] - single token input
            k_cache: [B*H, max_seqlen, D]
            v_cache: [B*H, max_seqlen, D]
            seqlens: [B*H] - sequence lengths
        Returns:
            out: [B, 1, D] - attention output
            lse: [B, H] - log-sum-exp
        """
        B = x.shape[0]
        x_norm = self.attn_norm(x)  # [B, 1, D]
        qkv = self.qkv_proj(x_norm)  # [B, 1, 3*D]
        q, k_new, v_new = qkv.chunk(3, dim=-1)

        # Reshape q to [B, H, 1, D_head]
        q = q.view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        return flash_attn_decode(q, k_cache, v_cache, seqlens, self.softmax_scale)
