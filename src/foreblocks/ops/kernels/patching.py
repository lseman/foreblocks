"""foreblocks.ops.kernels.patching.

Triton kernels for patch tokenization and detokenization.

Provides Triton-accelerated patch materialization and detokenization while
preserving PyTorch fallbacks in the features layer.

Core API:
- _TRITON_AVAILABLE, _TRITON_PATCHIFY_MAX_INPUT_NUMEL
- _can_use_triton_patchify
- _materialize_patches

"""

import torch
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


_PatchifyFunction = None
_PatchDetokFunction = None
_patchify_fwd_kernel = None
_patchify_bwd_kernel = None
_patch_detok_fwd_kernel = None
_patch_detok_bwd_kernel = None

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


__all__ = [
    "_TRITON_AVAILABLE",
    "_TRITON_PATCHIFY_MAX_INPUT_NUMEL",
    "_can_use_triton_patchify",
    "_materialize_patches",
]

if _TRITON_AVAILABLE:
    __all__.extend([
        "_patchify_fwd_kernel",
        "_patchify_bwd_kernel",
        "_PatchifyFunction",
        "_patch_detok_fwd_kernel",
        "_patch_detok_bwd_kernel",
        "_PatchDetokFunction",
    ])
