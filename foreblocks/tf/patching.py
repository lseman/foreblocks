from dataclasses import dataclass
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    kpm: Optional[torch.Tensor],  # [B,T] bool
    T: int,
    patch_len: int,
    stride: int,
    pad_end: bool = True,
) -> Optional[torch.Tensor]:
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

    B = kpm.shape[0]
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

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, PatchInfo]:
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

        positions = (torch.arange(Np, device=device).unsqueeze(1) * S) + torch.arange(
            P, device=device
        ).unsqueeze(0)
        pos = positions.reshape(-1)  # [Np*P]

        patch_flat = patches.reshape(B, Np * P, D)  # [B,Np*P,D]

        out = torch.zeros(B, T_pad, D, device=device, dtype=dtype)
        out = out.index_add(1, pos, patch_flat)

        ones = torch.ones_like(pos, dtype=dtype, device=device)
        counts = torch.zeros(T_pad, device=device, dtype=dtype).index_add(0, pos, ones)
        out = out / counts.clamp_min(1.0).view(1, T_pad, 1)

        return out[:, : info.T_orig, :]
