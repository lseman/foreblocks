"""foreblocks.models.transformer.patching.

This module defines patch tokenization and detokenization utilities.
It belongs to the modular transformer layers and helpers area of Foreblocks.
It exposes classes such as PatchInfo, PatchTokenizer, PatchDetokenizer.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    _TRITON_AVAILABLE = False


if _TRITON_AVAILABLE:

    @triton.jit
    def _patch_detok_fwd_kernel(
        patches_ptr,  # [B, Np, P, D] contiguous (stride_D = 1)
        out_ptr,  # [B, T_pad, D] contiguous
        Np,
        P,
        D,
        T_pad,
        S,
        stride_pb,
        stride_pnp,
        stride_pp,
        stride_ob,
        stride_ot,
        BLOCK_D: tl.constexpr,
    ):
        """For each (b, t): gather contributing patches, average, store."""
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)
        pid_d = tl.program_id(2)

        t = pid_t
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        # np range contributing to position t:
        #   np*S + p == t  with  0 <= p < P  =>  np in [np_lo, np_hi]
        # np_lo = ceil((t - P + 1) / S) clamped to 0
        # Using C-style truncation: (t - P + S) // S == ceil((t-P+1)/S) for t >= 0
        np_lo = tl.maximum(0, (t - P + S) // S)
        np_hi = tl.minimum(Np - 1, t // S)

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for np_idx in range(np_lo, np_hi + 1):
            p_idx = t - np_idx * S
            ptr = (
                patches_ptr
                + pid_b * stride_pb
                + np_idx * stride_pnp
                + p_idx * stride_pp
                + d_offs
            )
            acc += tl.load(ptr, mask=d_mask, other=0.0).to(tl.float32)

        count = tl.maximum(1, np_hi - np_lo + 1).to(tl.float32)
        acc = acc / count

        tl.store(out_ptr + pid_b * stride_ob + t * stride_ot + d_offs, acc, mask=d_mask)

    @triton.jit
    def _patch_detok_bwd_kernel(
        grad_out_ptr,  # [B, T_pad, D] contiguous
        grad_patches_ptr,  # [B, Np, P, D] contiguous
        Np,
        P,
        D,
        S,
        stride_go_b,
        stride_go_t,
        stride_gp_b,
        stride_gp_np,
        stride_gp_p,
        BLOCK_D: tl.constexpr,
    ):
        """For each (b, np, p): grad_patches = grad_out[b, np*S+p, :] / count(np*S+p)."""
        pid_b = tl.program_id(0)
        pid_np = tl.program_id(1)
        pid_d = tl.program_id(2)

        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        for p_idx in range(P):
            t = pid_np * S + p_idx

            # Recompute count for position t (same formula as forward)
            np_lo = tl.maximum(0, (t - P + S) // S)
            np_hi = tl.minimum(Np - 1, t // S)
            count = tl.maximum(1, np_hi - np_lo + 1).to(tl.float32)

            go_ptr = grad_out_ptr + pid_b * stride_go_b + t * stride_go_t + d_offs
            dout = tl.load(go_ptr, mask=d_mask, other=0.0).to(tl.float32)

            gp_ptr = (
                grad_patches_ptr
                + pid_b * stride_gp_b
                + pid_np * stride_gp_np
                + p_idx * stride_gp_p
                + d_offs
            )
            tl.store(gp_ptr, dout / count, mask=d_mask)

    class _PatchDetokFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, patches, Np, P, D, T_pad, S, BLOCK_D):  # type: ignore[override]
            B = patches.shape[0]
            patches = patches.contiguous()
            out = torch.empty(B, T_pad, D, dtype=patches.dtype, device=patches.device)

            grid = (B, T_pad, triton.cdiv(D, BLOCK_D))
            _patch_detok_fwd_kernel[grid](
                patches,
                out,
                Np,
                P,
                D,
                T_pad,
                S,
                patches.stride(0),
                patches.stride(1),
                patches.stride(2),
                out.stride(0),
                out.stride(1),
                BLOCK_D=BLOCK_D,
            )
            ctx.Np = Np
            ctx.P = P
            ctx.D = D
            ctx.S = S
            ctx.BLOCK_D = BLOCK_D
            ctx.T_pad = T_pad
            return out

        @staticmethod
        def backward(ctx, grad_out):  # type: ignore[override]
            Np, P, D, S, BLOCK_D, T_pad = (
                ctx.Np, ctx.P, ctx.D, ctx.S, ctx.BLOCK_D, ctx.T_pad,
            )
            B = grad_out.shape[0]

            # grad_out has shape [B, T_orig, D] because the detokenizer slices
            # the forward output to [:, :T_orig, :].  Zero-pad to [B, T_pad, D]
            # so the kernel can safely read all positions 0..T_pad-1.
            if grad_out.shape[1] < T_pad:
                grad_out = F.pad(grad_out, (0, 0, 0, T_pad - grad_out.shape[1]))
            grad_out = grad_out.contiguous()
            grad_patches = torch.empty(
                B, Np, P, D, dtype=grad_out.dtype, device=grad_out.device
            )

            grid = (B, Np, triton.cdiv(D, BLOCK_D))
            _patch_detok_bwd_kernel[grid](
                grad_out,
                grad_patches,
                Np,
                P,
                D,
                S,
                grad_out.stride(0),
                grad_out.stride(1),
                grad_patches.stride(0),
                grad_patches.stride(1),
                grad_patches.stride(2),
                BLOCK_D=BLOCK_D,
            )
            # Non-tensor args → None
            return grad_patches, None, None, None, None, None, None


# ──────────────────────────────────────────────────────────────────────────────
# PatchTST-style patching helpers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class PatchInfo:
    T_orig: int
    T_pad: int
    n_patches: int
    patch_len: int
    stride: int


def _compute_patch_pad(T: int, P: int, S: int) -> int:
    if T <= 0:
        return 0
    if T < P:
        return P - T
    n_patches = math.ceil((T - P) / S) + 1
    T_pad = (n_patches - 1) * S + P
    return max(0, T_pad - T)


def patchify_padding_mask(
    kpm: torch.Tensor | None,  # [B,T] bool
    T: int,
    patch_len: int,
    stride: int,
    pad_end: bool = True,
) -> torch.Tensor | None:
    """
    Convert timestep key-padding-mask [B,T] to patch-token mask [B,Np].
    Heuristic: a patch token is "padded" if ALL timesteps inside that patch are padded.
    This is conservative and works well for right-padding masks.

    If kpm is None, returns None.
    """
    if kpm is None:
        return None
    if kpm.dim() != 2 or kpm.shape[1] != T:
        raise ValueError(f"Expected kpm [B,T={T}], got {tuple(kpm.shape)}")

    P, S = int(patch_len), int(stride)
    pad = _compute_patch_pad(T, P, S) if pad_end else 0

    if pad > 0:
        kpm = F.pad(kpm, (0, pad), value=True)  # padded timesteps treated as masked
    # unfold: [B, Np, P]
    patches = kpm.unfold(dimension=1, size=P, step=S).contiguous()
    # patch masked if all elements are masked
    pkpm = patches.all(dim=-1)  # [B, Np]
    return pkpm


class PatchTokenizer(nn.Module):
    """
    Patchify + embed:
      x: [B, T, D] -> tokens: [B, Np, D]
    """

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        pad_end: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.pad_end = bool(pad_end)
        self.proj = nn.Linear(self.patch_len * self.d_model, self.d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, PatchInfo]:
        if x.dim() != 3:
            raise ValueError(f"PatchTokenizer expects [B,T,D], got {tuple(x.shape)}")
        B, T, D = x.shape
        if D != self.d_model:
            raise ValueError(
                f"d_model mismatch: x has D={D}, tokenizer d_model={self.d_model}"
            )

        pad = _compute_patch_pad(T, self.patch_len, self.stride) if self.pad_end else 0
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.shape[1]

        patches = x.unfold(
            dimension=1, size=self.patch_len, step=self.stride
        ).contiguous()  # [B,Np,P,D]
        Np = patches.shape[1]
        flat = patches.reshape(B, Np, self.patch_len * D)
        tokens = self.proj(flat)  # [B,Np,D]

        info = PatchInfo(
            T_orig=T,
            T_pad=T_pad,
            n_patches=Np,
            patch_len=self.patch_len,
            stride=self.stride,
        )
        return tokens, info


class PatchDetokenizer(nn.Module):
    """
    Optional unpatch (only needed if you patch the decoder and want per-timestep output):
      tokens: [B,Np,D] -> x: [B,T_orig,D]
    Uses overlap-add folding with count normalization.
    """

    def __init__(self, d_model: int, patch_len: int, stride: int, bias: bool = True):
        super().__init__()
        self.d_model = int(d_model)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.expand = nn.Linear(self.d_model, self.patch_len * self.d_model, bias=bias)

    def forward(self, tokens: torch.Tensor, info: PatchInfo) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(
                f"PatchDetokenizer expects [B,Np,D], got {tuple(tokens.shape)}"
            )
        B, Np, D = tokens.shape
        if D != self.d_model:
            raise ValueError(
                f"d_model mismatch: tokens D={D}, detok d_model={self.d_model}"
            )
        if Np != info.n_patches:
            raise ValueError(
                f"Patch count mismatch: tokens Np={Np} vs info.n_patches={info.n_patches}"
            )

        P, S = info.patch_len, info.stride
        T_pad = info.T_pad
        device = tokens.device
        dtype = tokens.dtype

        patches = self.expand(tokens).reshape(B, Np, P, D)  # [B,Np,P,D]

        if (
            _TRITON_AVAILABLE
            and patches.is_cuda
            and patches.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and not torch.jit.is_scripting()
        ):
            BLOCK_D = min(triton.next_power_of_2(D), 64)
            out = _PatchDetokFunction.apply(patches, Np, P, D, T_pad, S, BLOCK_D)
        else:
            positions = (
                torch.arange(Np, device=device).unsqueeze(1) * S
            ) + torch.arange(P, device=device).unsqueeze(0)
            pos = positions.reshape(-1)  # [Np*P]

            patch_flat = patches.reshape(B, Np * P, D)  # [B,Np*P,D]

            out = torch.zeros(B, T_pad, D, device=device, dtype=dtype)
            out = out.index_add(1, pos, patch_flat)

            ones = torch.ones_like(pos, dtype=dtype, device=device)
            counts = torch.zeros(T_pad, device=device, dtype=dtype).index_add(
                0, pos, ones
            )
            out = out / counts.clamp_min(1.0).view(1, T_pad, 1)

        return out[:, : info.T_orig, :]
