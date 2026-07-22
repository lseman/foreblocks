"""foreblocks.models.transformer.features.patching.

PatchTST-style patch tokenization and detokenization with Triton acceleration.

Provides PatchTokenizer for splitting time series into overlapping patch tokens
and PatchDetokenizer for overlap-add reconstruction. CUDA kernels accelerate
patch materialization and detokenization while preserving PyTorch fallbacks.

Core API:
- PatchTokenizer: patchify + embed [B,T,D] → [B,Np,D]
- PatchDetokenizer: unpatch with overlap-add [B,Np,D] → [B,T,D]
- PatchInfo: metadata for patch/detoken operations
- patchify_padding_mask: convert timestep mask to patch-token mask

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

# Above this input size, PyTorch's unfold copy is faster on representative
# transformer shapes. Keep the custom path focused on its measured crossover.
_TRITON_PATCHIFY_MAX_INPUT_NUMEL = 524_288


if _TRITON_AVAILABLE:

    @triton.jit
    def _patchify_fwd_kernel(
        x_ptr,  # [B, T, D], arbitrary strides
        patches_ptr,  # [B, Np, D, P], contiguous (matches Tensor.unfold)
        Np,
        P,
        D,
        S,
        stride_xb,
        stride_xt,
        stride_xd,
        stride_pb,
        stride_pnp,
        BLOCK: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        patch_idx = tl.program_id(1)
        tile_idx = tl.program_id(2)

        flat_offs = tile_idx * BLOCK + tl.arange(0, BLOCK)
        flat_mask = flat_offs < D * P
        d_offs = flat_offs // P
        patch_offsets = flat_offs - d_offs * P
        timesteps = patch_idx * S + patch_offsets

        values = tl.load(
            x_ptr + pid_b * stride_xb + timesteps * stride_xt + d_offs * stride_xd,
            mask=flat_mask,
            other=0.0,
        )
        tl.store(
            patches_ptr + pid_b * stride_pb + patch_idx * stride_pnp + flat_offs,
            values,
            mask=flat_mask,
        )

    @triton.jit
    def _patchify_bwd_kernel(
        grad_patches_ptr,  # [B, Np, D, P], contiguous
        grad_x_ptr,  # [B, T, D], contiguous
        Np,
        P,
        D,
        S,
        stride_gpb,
        stride_gpnp,
        stride_gpd,
        stride_gxb,
        stride_gxt,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        timestep = tl.program_id(1)
        pid_d = tl.program_id(2)
        d_offs = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        patch_lo = tl.maximum(0, (timestep - P + S) // S)
        patch_hi = tl.minimum(Np - 1, timestep // S)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for patch_idx in range(patch_lo, patch_hi + 1):
            patch_offset = timestep - patch_idx * S
            ptr = (
                grad_patches_ptr
                + pid_b * stride_gpb
                + patch_idx * stride_gpnp
                + d_offs * stride_gpd
                + patch_offset
            )
            acc += tl.load(ptr, mask=d_mask, other=0.0).to(tl.float32)

        tl.store(
            grad_x_ptr + pid_b * stride_gxb + timestep * stride_gxt + d_offs,
            acc,
            mask=d_mask,
        )

    class _PatchifyFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, Np, P, D, S, BLOCK_D):  # type: ignore[override]
            B, T, _ = x.shape
            patches = torch.empty(B, Np, D, P, dtype=x.dtype, device=x.device)
            block = 256
            grid = (B, Np, triton.cdiv(D * P, block))
            _patchify_fwd_kernel[grid](
                x,
                patches,
                Np,
                P,
                D,
                S,
                x.stride(0),
                x.stride(1),
                x.stride(2),
                patches.stride(0),
                patches.stride(1),
                BLOCK=block,
            )
            ctx.input_shape = (B, T, D)
            ctx.Np = Np
            ctx.P = P
            ctx.S = S
            ctx.BLOCK_D = BLOCK_D
            return patches

        @staticmethod
        def backward(ctx, grad_patches):  # type: ignore[override]
            B, T, D = ctx.input_shape
            grad_patches = grad_patches.contiguous()
            grad_x = torch.empty(
                B, T, D, dtype=grad_patches.dtype, device=grad_patches.device
            )
            grid = (B, T, triton.cdiv(D, ctx.BLOCK_D))
            _patchify_bwd_kernel[grid](
                grad_patches,
                grad_x,
                ctx.Np,
                ctx.P,
                D,
                ctx.S,
                grad_patches.stride(0),
                grad_patches.stride(1),
                grad_patches.stride(2),
                grad_x.stride(0),
                grad_x.stride(1),
                BLOCK_D=ctx.BLOCK_D,
            )
            return grad_x, None, None, None, None, None

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
                ctx.Np,
                ctx.P,
                ctx.D,
                ctx.S,
                ctx.BLOCK_D,
                ctx.T_pad,
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


def _can_use_triton_patchify(x: torch.Tensor) -> bool:
    return (
        _TRITON_AVAILABLE
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and x.numel() > 0
        and x.numel() <= _TRITON_PATCHIFY_MAX_INPUT_NUMEL
        and not torch.jit.is_scripting()
    )


def _materialize_patches(
    x: torch.Tensor, *, n_patches: int, patch_len: int, stride: int
) -> torch.Tensor:
    if _can_use_triton_patchify(x):
        feature_size = x.shape[-1]
        block_d = min(triton.next_power_of_2(feature_size), 64)
        return _PatchifyFunction.apply(
            x, n_patches, patch_len, feature_size, stride, block_d
        )
    return x.unfold(dimension=1, size=patch_len, step=stride).contiguous()


def patchify_padding_mask(
    kpm: torch.Tensor | None,  # [B,T] bool
    T: int,
    patch_len: int,
    stride: int,
    pad_end: bool = True,
) -> torch.Tensor | None:
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
        if self.d_model != D:
            raise ValueError(
                f"d_model mismatch: x has D={D}, tokenizer d_model={self.d_model}"
            )

        pad = _compute_patch_pad(T, self.patch_len, self.stride) if self.pad_end else 0
        if pad > 0:
            x = F.pad(x, (0, 0, 0, pad))
        T_pad = x.shape[1]

        Np = (T_pad - self.patch_len) // self.stride + 1
        patches = _materialize_patches(
            x,
            n_patches=Np,
            patch_len=self.patch_len,
            stride=self.stride,
        )
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
        if self.d_model != D:
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
