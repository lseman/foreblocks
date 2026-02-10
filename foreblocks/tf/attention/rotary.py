# Copyright (c) 2025, Tri Dao
# Optimized version with improved performance (batched + cu_seqlens support)

import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
from einops import rearrange, repeat
from torch import Tensor


# ---------------------------------------------------------------------------
# Basic rotary helpers
# ---------------------------------------------------------------------------

def rotate_half(x: Tensor, interleaved: bool = False) -> Tensor:
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(
            torch.stack((-x2, x1), dim=-1),
            "... d two -> ... (d two)",
            two=2,
        )


def apply_rotary_emb_torch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
) -> Tensor:
    """
    x:   (batch_size, seqlen, nheads, headdim)
    cos: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    sin: same as cos
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    cos_full = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )
    sin_full = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )

    x_rot = x[..., :ro_dim]
    rotated = x_rot * cos_full + rotate_half(x_rot, interleaved) * sin_full
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# Core rotary application used by custom autograd Functions
# ---------------------------------------------------------------------------

def apply_rotary(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
) -> Tensor:
    """
    Arguments:
        x: (batch_size, seqlen, nheads, headdim) if cu_seqlens is None
           else (total_seqlen, nheads, headdim)
        cos, sin: (seqlen_rotary, rotary_dim / 2)
                  or (batch_size, seqlen_rotary, rotary_dim / 2)
        interleaved: if True, rotate pairs of even/odd dims (GPT-J style)
                     instead of [0:D/2] vs [D/2:D] (GPT-NeoX style).
        inplace:     if True, modify x in-place.
        seqlen_offsets: int or (batch_size,), used for KV cache offset.
        cu_seqlens:  (batch + 1,) or None
        max_seqlen:  int, upper bound for rotary cache
        conjugate:   if True, use inverse rotation (for backward).

    Returns:
        out: same shape as x
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]

    if cu_seqlens is not None:
        return _apply_rotary_cu_seqlens(
            x,
            cos,
            sin,
            cu_seqlens,
            seqlen_offsets,
            max_seqlen,
            ro_dim,
            interleaved,
            inplace,
            conjugate,
        )
    else:
        return _apply_rotary_batch(
            x,
            cos,
            sin,
            seqlen_offsets,
            max_seqlen,
            ro_dim,
            interleaved,
            inplace,
            conjugate,
        )


def _prepare_cos_sin_batch(
    cos: Tensor,
    sin: Tensor,
    batch: int,
    seqlen: int,
    ro_dim: int,
    seqlen_offsets: Optional[Tensor],
    max_seqlen: Optional[int],
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    """
    Normalize cos/sin to shape (batch, seqlen, rotary_dim / 2)
    for the standard batched case (no cu_seqlens).
    """
    if cos.dim() == 2:
        # cos/sin: (seqlen_rotary, rotary_dim/2)
        if max_seqlen is None:
            if seqlen_offsets is None:
                max_seqlen = seqlen
            else:
                max_seqlen = seqlen + int(seqlen_offsets.max().item())
        cos = cos[:max_seqlen]
        sin = sin[:max_seqlen]

        if seqlen_offsets is not None:
            # (batch, seqlen) offsets → per-position indices
            positions = torch.arange(seqlen, device=device)  # (seqlen,)
            indices = seqlen_offsets.unsqueeze(1) + positions.unsqueeze(0)  # (batch, seqlen)
            cos = cos[indices]  # (batch, seqlen, d/2)
            sin = sin[indices]
        else:
            cos = cos[:seqlen].unsqueeze(0).expand(batch, -1, -1)
            sin = sin[:seqlen].unsqueeze(0).expand(batch, -1, -1)

    elif cos.dim() == 3:
        # cos/sin: (batch, seqlen_rotary, rotary_dim/2)
        assert cos.shape[0] == batch and sin.shape[0] == batch
        assert cos.shape[2] * 2 == ro_dim

        if seqlen_offsets is not None:
            positions = torch.arange(seqlen, device=device)
            indices = seqlen_offsets.unsqueeze(1) + positions.unsqueeze(0)
            cos = torch.gather(
                cos,
                1,
                indices.unsqueeze(-1).expand(-1, -1, cos.shape[-1]),
            )
            sin = torch.gather(
                sin,
                1,
                indices.unsqueeze(-1).expand(-1, -1, sin.shape[-1]),
            )
        else:
            cos = cos[:, :seqlen]
            sin = sin[:, :seqlen]
    else:
        raise ValueError(
            "cos and sin must be (seqlen_rotary, rotary_dim/2) or "
            "(batch, seqlen_rotary, rotary_dim/2)"
        )

    return cos, sin


def _apply_rotary_batch(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    seqlen_offsets: Union[int, Tensor],
    max_seqlen: Optional[int],
    ro_dim: int,
    interleaved: bool,
    inplace: bool,
    conjugate: bool,
) -> Tensor:
    """Optimized rotary embedding for standard batched input."""
    batch, seqlen, nheads, headdim = x.shape
    device = x.device

    # Fast path: no offset, cos/sin 2D, same for all batch
    if (
        isinstance(seqlen_offsets, int)
        and seqlen_offsets == 0
        and cos.dim() == 2
    ):
        if conjugate:
            sin = -sin

        cos_ = cos[:seqlen].unsqueeze(0)  # (1, seqlen, d/2)
        sin_ = sin[:seqlen].unsqueeze(0)

        cos_full = repeat(
            cos_,
            "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
        )
        sin_full = repeat(
            sin_,
            "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
        )

        x_rot = x[..., :ro_dim]
        rotated = x_rot * cos_full + rotate_half(x_rot, interleaved) * sin_full

        if inplace:
            x[..., :ro_dim] = rotated
            return x
        if ro_dim == headdim:
            return rotated
        return torch.cat([rotated, x[..., ro_dim:]], dim=-1)

    # General path with offsets and/or 3D cos/sin
    if isinstance(seqlen_offsets, int):
        if seqlen_offsets != 0:
            raise ValueError(
                "seqlen_offsets must be 0 or a tensor when cu_seqlens is None"
            )
        seqlen_offsets = None
    else:
        assert seqlen_offsets.dim() == 1 and seqlen_offsets.shape[0] == batch

    cos, sin = _prepare_cos_sin_batch(
        cos,
        sin,
        batch,
        seqlen,
        ro_dim,
        seqlen_offsets,
        max_seqlen,
        device,
    )

    if conjugate:
        sin = -sin

    cos_full = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )
    sin_full = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )

    x_rot = x[..., :ro_dim]
    rotated = x_rot * cos_full + rotate_half(x_rot, interleaved) * sin_full

    if inplace:
        x[..., :ro_dim] = rotated
        return x
    if ro_dim == headdim:
        return rotated
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)


def _apply_rotary_cu_seqlens(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    cu_seqlens: Tensor,
    seqlen_offsets: Union[int, Tensor],
    max_seqlen: Optional[int],
    ro_dim: int,
    interleaved: bool,
    inplace: bool,
    conjugate: bool,
) -> Tensor:
    """Rotary embedding for variable-length sequences (packed with cu_seqlens)."""
    total_seqlen, nheads, headdim = x.shape
    assert cu_seqlens.dim() == 1 and cu_seqlens.dtype in (torch.int32, torch.int64)
    batch = cu_seqlens.shape[0] - 1
    device = x.device

    if isinstance(seqlen_offsets, int):
        if seqlen_offsets != 0:
            raise ValueError(
                "If cu_seqlens is provided, seqlen_offsets must be a tensor or 0"
            )
        seqlen_offsets = torch.zeros(
            batch,
            dtype=torch.int64,
            device=cu_seqlens.device,
        )
    else:
        assert seqlen_offsets.dim() == 1 and seqlen_offsets.shape[0] == batch

    seq_lens = cu_seqlens[1:] - cu_seqlens[:-1]  # (batch,)
    # Build flat indices for each (batch,pos)
    positions = torch.cat(
        [
            torch.arange(seq_lens[b].item(), device=device)
            + seqlen_offsets[b].item()
            for b in range(batch)
        ],
        dim=0,
    )  # (total_seqlen,)

    if cos.dim() == 2:
        if max_seqlen is None:
            max_seqlen = total_seqlen + int(seqlen_offsets.max().item())
        cos = cos[:max_seqlen]
        sin = sin[:max_seqlen]

        cos = cos[positions]  # (total_seqlen, d/2)
        sin = sin[positions]
    elif cos.dim() == 3:
        assert cos.shape[0] == batch and sin.shape[0] == batch
        assert cos.shape[2] * 2 == ro_dim
        max_needed = total_seqlen + int(seqlen_offsets.max().item())
        assert cos.shape[1] >= max_needed and sin.shape[1] >= max_needed

        batch_indices = torch.arange(batch, device=device).repeat_interleave(seq_lens)
        cos = cos[batch_indices, positions]  # (total_seqlen, d/2)
        sin = sin[batch_indices, positions]
    else:
        raise ValueError(
            "cos and sin must be (seqlen_rotary, rotary_dim/2) or "
            "(batch, seqlen_rotary, rotary_dim/2)"
        )

    if conjugate:
        sin = -sin

    cos_full = repeat(
        cos,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )
    sin_full = repeat(
        sin,
        "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)",
    )

    x_rot = x[..., :ro_dim]
    rotated = x_rot * cos_full + rotate_half(x_rot, interleaved) * sin_full

    if inplace:
        x[..., :ro_dim] = rotated
        return x
    if ro_dim == headdim:
        return rotated
    return torch.cat([rotated, x[..., ro_dim:]], dim=-1)


# ---------------------------------------------------------------------------
# Autograd wrappers for rotary on x and qkv
# ---------------------------------------------------------------------------

class ApplyRotaryEmb(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleaved: bool = False,
        inplace: bool = False,
        seqlen_offsets: Union[int, Tensor] = 0,
        cu_seqlens: Optional[Tensor] = None,
        max_seqlen: Optional[int] = None,
    ):
        out = apply_rotary(
            x,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            interleaved=interleaved,
            inplace=inplace,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cu_seqlens)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cu_seqlens, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.inplace = inplace
        ctx.max_seqlen = max_seqlen
        return out if not inplace else x

    @staticmethod
    def backward(ctx, do: Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cu_seqlens, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cu_seqlens = ctx.saved_tensors

        dx = apply_rotary(
            do,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            cu_seqlens=cu_seqlens,
            max_seqlen=ctx.max_seqlen,
            interleaved=ctx.interleaved,
            inplace=ctx.inplace,
            conjugate=True,
        )
        return dx, None, None, None, None, None, None, None


def apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    inplace: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    cu_seqlens: Optional[Tensor] = None,
    max_seqlen: Optional[int] = None,
) -> Tensor:
    """
    Public entry point: same signature as original Tri Dao function.
    """
    return ApplyRotaryEmb.apply(
        x,
        cos,
        sin,
        interleaved,
        inplace,
        seqlen_offsets,
        cu_seqlens,
        max_seqlen,
    )


# For backward compatibility
apply_rotary_emb_func = apply_rotary_emb


def _apply_rotary_emb_qkv(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    cos_k: Optional[Tensor] = None,
    sin_k: Optional[Tensor] = None,
    interleaved: bool = False,
    inplace: bool = False,
    conjugate: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
) -> Tensor:
    apply_rotary_fn = partial(
        apply_rotary,
        interleaved=interleaved,
        inplace=inplace,
        conjugate=conjugate,
        seqlen_offsets=seqlen_offsets,
    )

    if cos_k is None and sin_k is None and qkv.is_contiguous():
        # Single-kernel path for contiguous qkv
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            qk = qkv[:, :, :2].reshape(batch, seqlen, -1, headdim)
            qk = apply_rotary_fn(qk, cos, sin)
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            qk = qkv[:, :, : num_heads_q + num_heads_k]
            qk = apply_rotary_fn(qk, cos, sin)
        if not inplace:
            if qkv.dim() == 5:
                qkv = torch.cat(
                    [
                        rearrange(qk, "b s (t h) d -> b s t h d", t=2),
                        qkv[:, :, 2:],
                    ],
                    dim=2,
                )
            else:
                qkv = torch.cat(
                    [qk, qkv[:, :, num_heads_q + num_heads_k :]],
                    dim=2,
                )
    else:
        cos_k = cos if cos_k is None else cos_k
        sin_k = sin if sin_k is None else sin_k
        if qkv.dim() == 5:
            batch, seqlen, three, nheads, headdim = qkv.shape
            assert three == 3
            q, k = qkv[:, :, 0], qkv[:, :, 1]
        else:
            assert qkv.dim() == 4
            assert num_heads_q is not None
            num_heads_k = (qkv.shape[2] - num_heads_q) // 2
            assert qkv.shape[2] == num_heads_q + 2 * num_heads_k
            q = qkv[:, :, :num_heads_q]
            k = qkv[:, :, num_heads_q : num_heads_q + num_heads_k]

        q = apply_rotary_fn(q, cos, sin)
        k = apply_rotary_fn(k, cos_k, sin_k)

        if not inplace:
            if qkv.dim() == 5:
                qkv = torch.stack([q, k, qkv[:, :, 2]], dim=2)
            else:
                qkv = torch.cat(
                    [q, k, qkv[:, :, num_heads_q + num_heads_k :]],
                    dim=2,
                )
    return qkv


class ApplyRotaryEmbQKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        qkv: Tensor,
        cos: Tensor,
        sin: Tensor,
        cos_k: Optional[Tensor] = None,
        sin_k: Optional[Tensor] = None,
        interleaved: bool = False,
        seqlen_offsets: Union[int, Tensor] = 0,
        num_heads_q: Optional[int] = None,
    ):
        qkv = _apply_rotary_emb_qkv(
            qkv,
            cos,
            sin,
            cos_k,
            sin_k,
            interleaved=interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offsets,
            num_heads_q=num_heads_q,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin, cos_k, sin_k)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, cos_k, sin_k, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        ctx.num_heads_q = num_heads_q
        return qkv

    @staticmethod
    def backward(ctx, dqkv: Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, cos_k, sin_k, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin, cos_k, sin_k = ctx.saved_tensors

        dqkv = _apply_rotary_emb_qkv(
            dqkv,
            cos,
            sin,
            cos_k,
            sin_k,
            interleaved=ctx.interleaved,
            inplace=True,
            seqlen_offsets=seqlen_offsets,
            num_heads_q=ctx.num_heads_q,
            conjugate=True,
        )
        return dqkv, None, None, None, None, None, None, None


def apply_rotary_emb_qkv_(
    qkv: Tensor,
    cos: Tensor,
    sin: Tensor,
    cos_k: Optional[Tensor] = None,
    sin_k: Optional[Tensor] = None,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
    num_heads_q: Optional[int] = None,
) -> Tensor:
    """
    Arguments:
        qkv: (batch, seqlen, 3, nheads, headdim)
             or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
        cos_k, sin_k: optional (seqlen, rotary_dim / 2) for XPos-style scaling.
    Returns:
        qkv with rotary applied in-place to Q and K.
    """
    return ApplyRotaryEmbQKV_.apply(
        qkv,
        cos,
        sin,
        cos_k,
        sin_k,
        interleaved,
        seqlen_offsets,
        num_heads_q,
    )


class ApplyRotaryEmbKV_(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        kv: Tensor,
        cos: Tensor,
        sin: Tensor,
        interleaved: bool = False,
        seqlen_offsets: Union[int, Tensor] = 0,
    ):
        batch, seqlen, two, nheads, headdim = kv.shape
        assert two == 2
        k = kv[:, :, 0]
        apply_rotary(
            k,
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=interleaved,
            inplace=True,
        )
        if isinstance(seqlen_offsets, int):
            ctx.save_for_backward(cos, sin)
            ctx.seqlen_offsets = seqlen_offsets
        else:
            ctx.save_for_backward(cos, sin, seqlen_offsets)
            ctx.seqlen_offsets = None
        ctx.interleaved = interleaved
        return kv

    @staticmethod
    def backward(ctx, dkv: Tensor):
        seqlen_offsets = ctx.seqlen_offsets
        if seqlen_offsets is None:
            cos, sin, seqlen_offsets = ctx.saved_tensors
        else:
            cos, sin = ctx.saved_tensors
        apply_rotary(
            dkv[:, :, 0],
            cos,
            sin,
            seqlen_offsets=seqlen_offsets,
            interleaved=ctx.interleaved,
            inplace=True,
            conjugate=True,
        )
        return dkv, None, None, None, None


def apply_rotary_emb_kv_(
    kv: Tensor,
    cos: Tensor,
    sin: Tensor,
    interleaved: bool = False,
    seqlen_offsets: Union[int, Tensor] = 0,
) -> Tensor:
    """
    Arguments:
        kv:  (batch_size, seqlen, 2, nheads, headdim)
        cos, sin: (seqlen, rotary_dim / 2)
    Returns:
        kv: same shape, with rotary applied in-place to K.
    """
    return ApplyRotaryEmbKV_.apply(kv, cos, sin, interleaved, seqlen_offsets)


# ---------------------------------------------------------------------------
# RotaryEmbedding module with caching (RoPE / XPos)
# ---------------------------------------------------------------------------

class RotaryEmbedding(torch.nn.Module):
    """
    Rotary position embeddings (RoFormer / GPT-NeoX) with optional XPos scaling.

    If scale_base is not None, this implements XPos (Sun et al.).
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        interleaved: bool = False,
        scale_base: Optional[float] = None,
        device=None,
    ):
        super().__init__()
        self.dim = dim
        self.base = float(base)

        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        self.interleaved = interleaved
        self.scale_base = scale_base
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale, persistent=False)

        self._seq_len_cached = 0
        self._cos_cached: Optional[Tensor] = None
        self._sin_cached: Optional[Tensor] = None
        self._cos_k_cached: Optional[Tensor] = None
        self._sin_k_cached: Optional[Tensor] = None

    def _compute_inv_freq(self, device=None) -> Tensor:
        return 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim)
        )

    def _update_cos_sin_cache(
        self,
        seqlen: int,
        device=None,
        dtype=None,
    ) -> None:
        # Recompute if:
        #  - seqlen grows
        #  - device changes
        #  - dtype changes
        #  - we switch from inference -> training (need grad)
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached is None
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.requires_grad)
        ):
            self._seq_len_cached = seqlen
            t = torch.arange(seqlen, device=device, dtype=torch.float32)

            if self.inv_freq.dtype != torch.float32:
                inv_freq = self._compute_inv_freq(device=device)
            else:
                inv_freq = self.inv_freq

            freqs = torch.outer(t, inv_freq)  # (seqlen, dim/2)

            if self.scale is None:
                cos = torch.cos(freqs)
                sin = torch.sin(freqs)
                self._cos_cached = cos.to(dtype)
                self._sin_cached = sin.to(dtype)
                self._cos_k_cached = None
                self._sin_k_cached = None
            else:
                power = (
                    torch.arange(
                        seqlen,
                        dtype=self.scale.dtype,
                        device=self.scale.device,
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(power.device) ** rearrange(power, "s -> s 1")

                cos = torch.cos(freqs)
                sin = torch.sin(freqs)
                self._cos_cached = (cos * scale).to(dtype)
                self._sin_cached = (sin * scale).to(dtype)
                self._cos_k_cached = (cos / scale).to(dtype)
                self._sin_k_cached = (sin / scale).to(dtype)

    def forward(
        self,
        qkv: Tensor,
        kv: Optional[Tensor] = None,
        seqlen_offset: Union[int, Tensor] = 0,
        max_seqlen: Optional[int] = None,
        num_heads_q: Optional[int] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """
        qkv:
          - If kv is None:
              (batch, seqlen, 3, nheads, headdim)
              or (batch, seqlen, num_heads_q + 2 * num_heads_k, headdim)
            If MQA/GQA layout, num_heads_q must be provided.
          - If kv is not None:
              qkv = q: (batch, seqlen, nheads, headdim)
              kv:      (batch, seqlen, 2, nheads, headdim)

        seqlen_offset: int or (batch,)
        max_seqlen:    used to grow cache when working with KV cache.
        """
        seqlen = qkv.shape[1]
        if max_seqlen is not None:
            self._update_cos_sin_cache(
                max_seqlen,
                device=qkv.device,
                dtype=qkv.dtype,
            )
        elif isinstance(seqlen_offset, int):
            self._update_cos_sin_cache(
                seqlen + seqlen_offset,
                device=qkv.device,
                dtype=qkv.dtype,
            )
        else:
            # tensor offsets
            self._update_cos_sin_cache(
                seqlen + int(seqlen_offset.max().item()),
                device=qkv.device,
                dtype=qkv.dtype,
            )

        cos_qk = self._cos_cached
        sin_qk = self._sin_cached
        cos_k = self._cos_k_cached if self.scale is not None else None
        sin_k = self._sin_k_cached if self.scale is not None else None

        if kv is None:
            # qkv includes Q and K
            return apply_rotary_emb_qkv_(
                qkv,
                cos_qk,
                sin_qk,
                cos_k,
                sin_k,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
                num_heads_q=num_heads_q,
            )
        else:
            # q and kv given separately
            q = apply_rotary_emb_func(
                qkv,
                cos_qk,
                sin_qk,
                interleaved=self.interleaved,
                inplace=True,
                seqlen_offsets=seqlen_offset,
            )
            kv = apply_rotary_emb_kv_(
                kv,
                cos_qk if self.scale is None else cos_k,
                sin_qk if self.scale is None else sin_k,
                interleaved=self.interleaved,
                seqlen_offsets=seqlen_offset,
            )
            return q, kv

