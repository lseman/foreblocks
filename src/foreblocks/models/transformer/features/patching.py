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

from foreblocks.ops.kernels.patching import (
    _TRITON_AVAILABLE,
    _can_use_triton_patchify,
    _materialize_patches,
)

# Import Triton detoken function and triton module if available
if _TRITON_AVAILABLE:
    from foreblocks.ops.kernels.patching import _PatchDetokFunction
    import triton
else:
    _PatchDetokFunction = None  # type: ignore[assignment]
    triton = None  # type: ignore[assignment]


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
            and _PatchDetokFunction is not None
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


__all__ = [
    "PatchDetokenizer",
    "PatchInfo",
    "PatchTokenizer",
    "patchify_padding_mask",
    "_compute_patch_pad",
    "_TRITON_AVAILABLE",
    "_can_use_triton_patchify",
    "_materialize_patches",
]
