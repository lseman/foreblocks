# heads_all.py
# ──────────────────────────────────────────────────────────────────────────────
# Time-series heads: decomposition, normalization, conv, FFT Top-K, Time2Vec,
# differencing, learnable Fourier seasonal, DAIN, patch embedding,
# Haar wavelet Top-K, TimeAttention, Chronos-2 embeddings, DropoutTS.
#
# FIXES included:
#   (1) DecompositionHead: safer MA init + groups validation
#   (2) PatchEmbed: robust upsample via F.interpolate(size=T) (no length drift)
#   (3) TimeAttention: rewritten (correct channel-independent attention, no d_model collapse)
#   (4) Chronos2EmbedHead: projector dtype/device fixed + explicit warmup() to instantiate params
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui_aux.node_spec import node


# ──────────────────────────────────────────────────────────────────────────────
# Decomposition (Autoformer-style moving average via depthwise Conv1d)
# Splits x into (seasonal, trend). Seasonal can go to encoder; add trend back.
# ──────────────────────────────────────────────────────────────────────────────


@node(
    type_id="decomposition_head",
    name="DecompositionHead",
    category="Preprocessing",
    outputs=["decomposition_head"],
    color="bg-gradient-to-r from-green-400 to-blue-500",
)
class DecompositionHead(nn.Module):
    """
    Series decomposition head for trend-seasonal separation.
    Inspired by Autoformer: learnable moving average (depthwise Conv1d) per feature.
    Forward: (seasonal, trend)  with shape [B,T,F].
    """

    def __init__(
        self,
        kernel_size: int = 25,
        feature_dim: int = None,  # Input features F (required)
        hidden_dim: Optional[int] = None,  # Optional projection on seasonal
        groups: Optional[int] = None,
    ):
        super().__init__()
        if feature_dim is None:
            raise ValueError("feature_dim must be provided (e.g., 32)")

        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim) if hidden_dim is not None else self.feature_dim
        self.kernel_size = int(kernel_size)

        if groups is None:
            groups = self.feature_dim  # per-channel filtering
        groups = int(groups)
        if self.feature_dim % groups != 0:
            raise ValueError(
                f"groups must divide feature_dim (feature_dim={self.feature_dim}, groups={groups})"
            )

        padding = self.kernel_size // 2
        self.decomp = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # Initialize as exact moving average (constant)
        with torch.no_grad():
            self.decomp.weight.fill_(1.0 / float(self.kernel_size))

        self.post_proj = (
            nn.Linear(self.feature_dim, self.hidden_dim)
            if self.hidden_dim != self.feature_dim
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [B,T,F]
        F_in = x.size(-1)
        if F_in != self.feature_dim:
            raise RuntimeError(
                f"Input feature dim {F_in} != expected {self.feature_dim}."
            )

        xt = x.transpose(1, 2)  # [B,F,T] for Conv1d
        trend = self.decomp(xt)  # [B,F,T]
        seasonal = xt - trend  # [B,F,T]
        seasonal = seasonal.transpose(1, 2)  # [B,T,F]
        trend = trend.transpose(1, 2)  # [B,T,F]

        seasonal = self.post_proj(seasonal)  # optional projection
        return seasonal, trend


class DecompositionBlock(BaseHead):
    """BaseHead wrapper for DecompositionHead."""

    def __init__(
        self,
        kernel_size: int = 25,
        feature_dim: int = 1,
        hidden_dim: Optional[int] = None,
    ):
        decomp_module = DecompositionHead(
            kernel_size=kernel_size, feature_dim=feature_dim, hidden_dim=hidden_dim
        )
        super().__init__(module=decomp_module, name="decomposition")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# RevIN (Reversible Instance Normalization)
# ──────────────────────────────────────────────────────────────────────────────


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (per-variable, over time).
    Forward: returns (x_norm, ctx) where ctx has {mu, sigma}.
    invert(x_hat, ctx) -> original scale.
    """

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = int(num_features)
        self.affine = bool(affine)
        self.eps = float(eps)
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, self.num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, self.num_features))
        else:
            self.register_buffer("gamma", torch.ones(1, 1, self.num_features))
            self.register_buffer("beta", torch.zeros(1, 1, self.num_features))

    @torch.no_grad()
    def _stats(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=1, keepdim=True)  # [B,1,F]
        var = x.var(dim=1, unbiased=False, keepdim=True)  # [B,1,F]
        sigma = torch.sqrt(var + self.eps)
        return mu, sigma

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        mu, sigma = self._stats(x)
        x_hat = (x - mu) / sigma
        x_hat = x_hat * self.gamma + self.beta
        ctx = {"mu": mu, "sigma": sigma}
        return x_hat, ctx

    def invert(self, x_hat: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = (x_hat - self.beta) / (self.gamma + 1e-12)
        return x * ctx["sigma"] + ctx["mu"]


@node(
    type_id="revin_head",
    name="RevINHead",
    category="Preprocessing",
    outputs=["revin_head"],
    color="bg-gradient-to-r from-yellow-400 to-red-500",
)
class RevINHead(BaseHead):
    """BaseHead wrapper for RevIN. Forward -> (x_norm, ctx)."""

    def __init__(self, feature_dim: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(
            module=RevIN(feature_dim, affine=affine, eps=eps), name="revin"
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Conv
# ──────────────────────────────────────────────────────────────────────────────


class MultiScaleConv(nn.Module):
    """
    Depthwise separable multi-kernel convs per variable, fused by 1x1 conv.
    Input/Output: [B,T,F]. Returns residual x + y.
    """

    def __init__(
        self,
        feature_dim: int,
        kernels: List[int] = [3, 5, 7, 11],
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.branches = nn.ModuleList()
        for k in kernels:
            k = int(k)
            pad = (k // 2) * int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.feature_dim,
                        self.feature_dim,
                        kernel_size=k,
                        padding=pad,
                        dilation=int(dilation),
                        groups=self.feature_dim,
                        bias=False,
                    ),
                    nn.Conv1d(self.feature_dim, self.feature_dim, kernel_size=1, bias=True),
                    nn.GELU(),
                )
            )
        self.fuse = nn.Conv1d(
            self.feature_dim * len(kernels), self.feature_dim, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)  # [B,F,T]
        outs = [b(xt) for b in self.branches]
        y = torch.cat(outs, dim=1)  # [B,F*K,T]
        y = self.fuse(y)  # [B,F,T]
        y = self.dropout(y).transpose(1, 2)
        return x + y


@node(
    type_id="msconv_head",
    name="MultiScaleConvHead",
    category="Preprocessing",
    outputs=["msconv_head"],
    color="bg-gradient-to-r from-pink-400 to-yellow-500",
)
class MultiScaleConvHead(BaseHead):
    """BaseHead wrapper for MultiScaleConv. Forward -> [B,T,F]."""

    def __init__(
        self,
        feature_dim: int,
        kernels=[3, 5, 7, 11],
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            module=MultiScaleConv(feature_dim, kernels, dilation, dropout),
            name="msconv",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# FFT Top-K carry
# ──────────────────────────────────────────────────────────────────────────────


class FFTTopK(nn.Module):
    """
    Keep top-K magnitudes in frequency domain as seasonal carry; residual is main.
    Input/Output: [B,T,F]. Returns (main, seasonal).
    """

    def __init__(self, topk: int = 8):
        super().__init__()
        self.topk = int(topk)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        Xf = fft.rfft(x, dim=1)  # [B, Tr, F]
        mag = Xf.abs()

        k = min(self.topk, mag.size(1))
        if k <= 0:
            seasonal = torch.zeros_like(x)
            return x, seasonal

        topk_idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_idx, value=True)

        Xf_seasonal = torch.where(mask, Xf, torch.zeros_like(Xf))
        Xf_residual = Xf - Xf_seasonal

        seasonal = fft.irfft(Xf_seasonal, n=T, dim=1)
        main = fft.irfft(Xf_residual, n=T, dim=1)
        return main, seasonal


class FFTTopKHead(BaseHead):
    """BaseHead wrapper for FFTTopK. Forward -> (main, seasonal)."""

    def __init__(self, topk: int = 8):
        super().__init__(module=FFTTopK(topk=topk), name="fft_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time2Vec
# ──────────────────────────────────────────────────────────────────────────────


class Time2Vec(nn.Module):
    """
    Time2Vec-style periodic features per timestep, projected back to feature_dim.
    Input/Output: [B,T,F]. Keeps dim invariant.
    """

    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.k = int(k)
        self.periodic = bool(periodic)
        self.freq = nn.Parameter(torch.randn(1, 1, self.k))
        self.phase = nn.Parameter(torch.zeros(1, 1, self.k))
        in_dim = self.feature_dim + 1 + (self.k if self.periodic else 0)
        self.proj = nn.Linear(in_dim, self.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1)
        parts = [x, t]
        if self.periodic and self.k > 0:
            z = torch.sin(2 * math.pi * (t * self.freq + self.phase))  # [1,T,k]
            parts.append(z.expand(B, -1, -1))
        h = torch.cat(parts, dim=-1)
        return self.proj(h)


class Time2VecHead(BaseHead):
    """BaseHead wrapper for Time2Vec. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__(module=Time2Vec(feature_dim, k, periodic), name="time2vec")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Differencing (reversible)
# ──────────────────────────────────────────────────────────────────────────────


class Differencing(nn.Module):
    """
    First-order differencing along time with length preservation.
    Forward: (delta, ctx) where delta[:,0,:]=0 and ctx['x0']=first step.
    """

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        delta = x.clone()
        delta[:, 1:, :] = x[:, 1:, :] - x[:, :-1, :]
        delta[:, :1, :] = 0.0
        ctx = {"x0": x[:, :1, :]}
        return delta, ctx

    def invert(self, y_hat: torch.Tensor, ctx: Dict[str, torch.Tensor]) -> torch.Tensor:
        x0 = ctx["x0"]
        rec = torch.cumsum(y_hat, dim=1)
        rec[:, :1, :] = 0.0
        return rec + x0


class DifferencingHead(BaseHead):
    """BaseHead wrapper for Differencing. Forward -> (delta, ctx)."""

    def __init__(self):
        super().__init__(module=Differencing(), name="diff")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Fourier Seasonal
# ──────────────────────────────────────────────────────────────────────────────


class LearnableFourierSeasonal(nn.Module):
    """
    Learnable per-channel seasonal component via Fourier bases (sin/cos up to K).
    seasonal = B @ W, main = x - seasonal. Shapes [B,T,F].
    """

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.K = int(K)
        in_dim = 2 * self.K
        if share_weights:
            self.W = nn.Parameter(torch.randn(1, in_dim))
        else:
            self.W = nn.Parameter(torch.randn(self.feature_dim, in_dim))
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def _bases(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(-1)  # [T,1]
        ks = torch.arange(1, self.K + 1, device=device, dtype=dtype).view(1, self.K)
        ang = 2 * math.pi * t * ks / float(T)
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        return torch.cat([sin, cos], dim=-1)  # [T,2K]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}")
        Bx = self._bases(T, x.device, x.dtype)  # [T,2K]
        W = self.W.expand(F_, -1) if self.W.size(0) == 1 else self.W  # [F,2K]
        seasonal = (Bx @ W.t()).unsqueeze(0).expand(B, -1, -1)  # [B,T,F]
        main = x - seasonal
        return main, seasonal


class LearnableFourierSeasonalHead(BaseHead):
    """BaseHead wrapper for LearnableFourierSeasonal. Forward -> (main, seasonal)."""

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__(
            module=LearnableFourierSeasonal(feature_dim, K, share_weights),
            name="lfourier",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# DAIN
# ──────────────────────────────────────────────────────────────────────────────


class DAIN(nn.Module):
    """
    Deep Adaptive Input Normalization (DAIN).
    Applies adaptive shift, scale, and gating based on time summaries.
    Input/Output: [B,T,F]
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.W_a = nn.Parameter(torch.eye(self.feature_dim))
        self.W_b = nn.Parameter(torch.eye(self.feature_dim))
        self.W_c = nn.Parameter(torch.randn(self.feature_dim, self.feature_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(self.feature_dim))
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}.")

        a = x.mean(dim=1)  # [B,F]
        alpha = a @ self.W_a.t()
        shifted = x - alpha.unsqueeze(1)

        b = torch.sqrt(torch.mean(shifted**2, dim=1) + self.eps)
        beta = b @ self.W_b.t()
        scaled = shifted / (beta.unsqueeze(1) + self.eps)

        c = scaled.mean(dim=1)
        gate_input = (c @ self.W_c.t()) + self.d
        gamma = torch.sigmoid(gate_input)
        return scaled * gamma.unsqueeze(1)


class DAINHead(BaseHead):
    """BaseHead wrapper for DAIN. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int):
        super().__init__(module=DAIN(feature_dim=feature_dim), name="dain")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding (FIXED upsample)
# ──────────────────────────────────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """
    Local patching via depthwise Conv1d, stride=patch_size (non-overlap), then upsample.
    Shape preserved: [B,T,F] -> [B,T,F] (residual added).
    """

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.patch_size = int(patch_size)
        self.patch_proj = nn.Conv1d(
            in_channels=self.feature_dim,
            out_channels=self.feature_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            groups=self.feature_dim,
            bias=True,
            padding=0,
        )
        self.dropout = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != feature_dim={self.feature_dim}.")
        if T % self.patch_size != 0:
            raise ValueError(f"T={T} must be divisible by patch_size={self.patch_size}.")

        xt = x.transpose(1, 2)  # [B,F,T]
        patches = self.patch_proj(xt)  # [B,F,T/patch]
        patches = self.dropout(patches)
        # Robust resize back to exactly T (no scale-factor drift)
        embedded = F.interpolate(patches, size=T, mode="linear", align_corners=False)  # [B,F,T]
        return embedded.transpose(1, 2) + x


@node(
    type_id="patchemb_head",
    name="PatchEmbedHead",
    category="Preprocessing",
    outputs=["patchemb_head"],
    color="bg-gradient-to-r from-blue-400 to-purple-500",
)
class PatchEmbedHead(BaseHead):
    """BaseHead wrapper for PatchEmbed. Forward -> [B,T,F]."""

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__(module=PatchEmbed(feature_dim, patch_size, dropout), name="patchemb")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Haar Wavelet Top-K
# ──────────────────────────────────────────────────────────────────────────────


class HaarWaveletTopK(nn.Module):
    """
    1-level Haar wavelet analysis with Top-K keep on detail coefficients.
    Returns (main, detail_sparse) with shape [B,T,F].
    """

    def __init__(self, topk: int = 8):
        super().__init__()
        self.topk = int(topk)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        pad_added = False
        if T % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1, 0, 0), mode="replicate")
            B, T, F_ = x.shape
            pad_added = True

        x_even = x[:, 0::2, :]
        x_odd = x[:, 1::2, :]
        x_low = (x_even + x_odd) / math.sqrt(2.0)   # [B,T/2,F]
        x_high = (x_even - x_odd) / math.sqrt(2.0)  # [B,T/2,F]

        k = min(self.topk, x_high.size(1))
        if k > 0:
            mag = x_high.abs()
            idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
            mask = torch.zeros_like(mag, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            xh_sparse = torch.where(mask, x_high, torch.zeros_like(x_high))
        else:
            xh_sparse = torch.zeros_like(x_high)

        out_len = T
        main = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)
        detail_sparse = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)

        # main = inverse with detail=0
        main[:, 0::2, :] = (x_low + 0.0) / math.sqrt(2.0)
        main[:, 1::2, :] = (x_low - 0.0) / math.sqrt(2.0)

        # sparse detail = inverse with low=0
        detail_sparse[:, 0::2, :] = (0.0 + xh_sparse) / math.sqrt(2.0)
        detail_sparse[:, 1::2, :] = (0.0 - xh_sparse) / math.sqrt(2.0)

        if pad_added:
            main = main[:, :-1, :]
            detail_sparse = detail_sparse[:, :-1, :]

        return main, detail_sparse


class HaarWaveletTopKHead(BaseHead):
    """BaseHead wrapper for HaarWaveletTopK. Forward -> (main, detail_sparse)."""

    def __init__(self, topk: int = 8):
        super().__init__(module=HaarWaveletTopK(topk=topk), name="haar_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time Attention (FIXED)
# Channel-independent self-attention over time for each feature stream.
# Input/Output: [B,T,F]
# ──────────────────────────────────────────────────────────────────────────────


class _RoPE1D(nn.Module):
    """Vectorized 1D rotary positional embedding over last dim (head_dim must be even)."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        head_dim = int(head_dim)
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [N, T, H] where H=head_dim
        """
        N, T, H = x.shape
        t = torch.arange(T, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, H/2]
        sin = freqs.sin().to(dtype=x.dtype, device=x.device)  # [T, H/2]
        cos = freqs.cos().to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        return torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)  # interleave


class TimeAttention(nn.Module):
    """
    Per-feature Transformer block over time:
      - treats each feature f as its own sequence (length T)
      - attention is computed over time independently per feature stream
      - maps back to [B,T,F] via learned scalar projection + residual

    This is "channel-independent" in the sense that features do not attend to each other;
    each feature attends over its own history.

    NOTE: This head is shape-preserving but increases compute by F (batch becomes B*F).
    """

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        causal: bool = False,
        window: Optional[int] = None,  # if set, mask attention to |i-j| <= window
        use_rope: bool = True,
    ):
        super().__init__()
        feature_dim = int(feature_dim)
        d_model = int(d_model)
        n_heads = int(n_heads)

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = bool(causal)
        self.window = int(window) if window is not None else None
        self.use_rope = bool(use_rope)

        # Lift each feature stream (scalar per timestep) -> d_model using grouped 1x1 conv:
        # Input:  [B, F, T]
        # Output: [B, F*d_model, T]  with groups=F means each feature has its own Linear(1->d_model)
        self.lift = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim * d_model,
            kernel_size=1,
            groups=feature_dim,
            bias=True,
        )

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=float(dropout), batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = int(ffn_mult) * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, d_model),
            nn.Dropout(float(dropout)),
        )

        self.drop = nn.Dropout(float(dropout))

        # Project back to scalar per timestep for each feature stream
        self.to_scalar = nn.Linear(d_model, 1, bias=True)

        head_dim = d_model // n_heads
        self.rope = _RoPE1D(head_dim) if self.use_rope else None

    def _attn_mask(self, T: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if (not self.causal) and (self.window is None):
            return None

        # float mask: 0 allowed, -inf disallowed
        mask = torch.zeros((T, T), device=device, dtype=dtype)

        if self.causal:
            mask = mask + torch.triu(torch.full((T, T), float("-inf"), device=device, dtype=dtype), diagonal=1)

        if self.window is not None:
            idx = torch.arange(T, device=device)
            dist = idx.unsqueeze(1) - idx.unsqueeze(0)
            win_mask = dist.abs() > self.window
            mask = torch.where(win_mask, torch.full_like(mask, float("-inf")), mask)

        return mask

    def _apply_rope_qk(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rope is None:
            return q, k
        # q,k: [N, T, d_model] -> [N, T, n_heads, head_dim]
        N, T, _ = q.shape
        head_dim = self.d_model // self.n_heads
        qh = q.view(N, T, self.n_heads, head_dim)
        kh = k.view(N, T, self.n_heads, head_dim)

        # apply RoPE per head (vectorized by merging N*n_heads)
        qh2 = qh.permute(0, 2, 1, 3).contiguous().view(N * self.n_heads, T, head_dim)
        kh2 = kh.permute(0, 2, 1, 3).contiguous().view(N * self.n_heads, T, head_dim)
        qh2 = self.rope(qh2)
        kh2 = self.rope(kh2)
        qh = qh2.view(N, self.n_heads, T, head_dim).permute(0, 2, 1, 3).contiguous()
        kh = kh2.view(N, self.n_heads, T, head_dim).permute(0, 2, 1, 3).contiguous()

        q = qh.view(N, T, self.d_model)
        k = kh.view(N, T, self.d_model)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F] -> y: [B,T,F]
        """
        if x.dim() != 3:
            raise ValueError(f"TimeAttention expects [B,T,F], got {tuple(x.shape)}")

        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input F={F_} != configured feature_dim={self.feature_dim}")

        # Build per-feature streams: [B,F,T] -> lift -> [B, F*d_model, T]
        xt = x.permute(0, 2, 1).contiguous()                 # [B,F,T]
        lifted = self.lift(xt)                               # [B, F*d_model, T]
        lifted = lifted.view(B, self.feature_dim, self.d_model, T).permute(0, 1, 3, 2).contiguous()
        # lifted: [B, F, T, d_model] -> merge B*F
        h = lifted.view(B * self.feature_dim, T, self.d_model)

        h1 = self.norm1(h)
        q = k = v = h1
        q, k = self._apply_rope_qk(q, k)

        attn_mask = self._attn_mask(T, device=x.device, dtype=torch.float32)  # keep float32 for stability
        y, _ = self.attn(q, k, v, attn_mask=attn_mask)

        h2 = h + self.drop(y)
        h3 = self.norm2(h2)
        h4 = h2 + self.ffn(h3)

        # back to scalar per timestep per feature
        s = self.to_scalar(h4).squeeze(-1)                   # [B*F, T]
        s = s.view(B, self.feature_dim, T).permute(0, 2, 1).contiguous()  # [B,T,F]

        return x + s  # residual


@node(
    type_id="timeattn_head",
    name="TimeAttentionHead",
    category="Preprocessing",
    outputs=["timeattn_head"],
    color="bg-gradient-to-r from-indigo-400 to-cyan-500",
)
class TimeAttentionHead(BaseHead):
    """BaseHead wrapper for TimeAttention. Forward -> [B,T,F]."""

    def __init__(
        self,
        feature_dim: int,
        d_model: int = 128,
        n_heads: int = 4,
        dropout: float = 0.0,
        ffn_mult: int = 4,
        causal: bool = False,
        window: Optional[int] = None,
        use_rope: bool = True,
    ):
        super().__init__(
            module=TimeAttention(
                feature_dim=feature_dim,
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                ffn_mult=ffn_mult,
                causal=causal,
                window=window,
                use_rope=use_rope,
            ),
            name="timeattn",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Chronos-2 Embeddings Head
# ──────────────────────────────────────────────────────────────────────────────


@node(
    type_id="chronos2_embed_head",
    name="Chronos2EmbedHead",
    category="Feature",
    outputs=["chronos2_embeddings"],
    color="bg-gradient-to-r from-sky-500 to-emerald-500",
)
class Chronos2EmbedHead(BaseHead):
    """
    Extract Chronos-2 embeddings and return a 3D sequence for downstream encoders.

    IMPORTANT (optimizer safety):
      This head lazily creates a projector (D -> F) on first forward when needed.
      If you build the optimizer BEFORE the first forward, the projector params
      will NOT be in the optimizer. Fix by calling:

          head.warmup(sample_x)

      once before creating the optimizer (or run one model forward pass).

    Forward:
      x: [B,T,F] -> returns [B,T,F_out] or [B,T+1,F] depending on attach.
    """

    def __init__(
        self,
        pipeline,
        channel: int = 0,
        reduction: str = "mean",
        hook_layer: str = "encoder",
        pred_len: int = 1,
        attach: str = "feature",  # 'feature' | 'time_token' | 'replace'
        proj_to_input_dim: bool = True,
        offload_cpu: bool = True,
        make_dates=None,
    ):
        module = _Chronos2EmbedderModule(
            pipeline=pipeline,
            channel=channel,
            reduction=reduction,
            hook_layer=hook_layer,
            pred_len=pred_len,
            attach=attach,
            proj_to_input_dim=proj_to_input_dim,
            offload_cpu=offload_cpu,
            make_dates=make_dates,
        )
        super().__init__(module=module, name="chronos2_embed")

    @torch.no_grad()
    def warmup(self, x: torch.Tensor) -> None:
        """Run once to instantiate any lazy projector parameters (call before optimizer)."""
        _ = self.module(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class _Chronos2EmbedderModule(nn.Module):
    def __init__(
        self,
        pipeline,
        channel: int,
        reduction: str,
        hook_layer: str,
        pred_len: int,
        attach: str,
        proj_to_input_dim: bool,
        offload_cpu: bool,
        make_dates,
    ):
        super().__init__()
        if reduction not in {"mean", "last"}:
            raise ValueError("reduction must be 'mean' or 'last'")
        if hook_layer not in {"encoder", "input_patch"}:
            raise ValueError("hook_layer must be 'encoder' or 'input_patch'")
        if attach not in {"feature", "time_token", "replace"}:
            raise ValueError("attach must be 'feature', 'time_token', or 'replace'")

        self.pipeline = pipeline
        self.channel = int(channel)
        self.reduction = reduction
        self.hook_layer = hook_layer
        self.pred_len = max(1, int(pred_len))
        self.attach = attach
        self.proj_to_input_dim = bool(proj_to_input_dim)
        self.offload_cpu = bool(offload_cpu)
        self.make_dates = make_dates

        # Resolve model + hook target
        self._model = getattr(pipeline, "model", None) or getattr(pipeline, "inner_model", None)
        if self._model is None:
            raise RuntimeError("Chronos2EmbedHead: could not locate Chronos2Model.")
        if self.hook_layer == "encoder":
            if not hasattr(self._model, "encoder") or not hasattr(self._model.encoder, "final_layer_norm"):
                raise RuntimeError("Chronos2EmbedHead: encoder.final_layer_norm not found.")
            self._hook_module = self._model.encoder.final_layer_norm
        else:
            if not hasattr(self._model, "input_patch_embedding"):
                raise RuntimeError("Chronos2EmbedHead: input_patch_embedding not found.")
            self._hook_module = self._model.input_patch_embedding

        # Lazy init for projector (Chronos D -> input F)
        self._proj: Optional[nn.Linear] = None  # created on first forward once we know D and F

    @torch.no_grad()
    def _get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        import numpy as np
        import pandas as pd

        B, T, F = x.shape

        # Build minimal DF
        if self.make_dates is None:
            def _mk_dates(n):  # daily
                return pd.date_range("2000-01-01", periods=n, freq="D")
            make_dates = _mk_dates
        else:
            make_dates = self.make_dates

        x_cpu = x.detach().to("cpu")
        vals = x_cpu[:, :, self.channel].numpy().astype(np.float32)  # [B,T]

        ids, stamps, tgts = [], [], []
        for b in range(B):
            ids.extend([f"series_{b}"] * T)
            stamps.extend(make_dates(T))
            tgts.extend(vals[b].tolist())

        context_df = (
            pd.DataFrame({"id": ids, "timestamp": stamps, "target": tgts})
            .sort_values(["id", "timestamp"])
        )

        collected: List[torch.Tensor] = []

        def _hook(_, __, out):
            collected.append(out.detach().to("cpu") if self.offload_cpu else out.detach())

        h = self._hook_module.register_forward_hook(_hook)
        try:
            _ = self.pipeline.predict_df(
                context_df,
                future_df=None,
                prediction_length=self.pred_len,
                quantile_levels=[0.5],
                id_column="id",
                timestamp_column="timestamp",
                target="target",
            )
        finally:
            h.remove()

        if not collected:
            raise RuntimeError("Chronos2EmbedHead: no activations captured.")

        enc = torch.cat(collected, dim=0)  # typically [B, P, D]
        if enc.size(0) != B:
            enc = enc[:B]

        if enc.dim() == 2:
            # If hook returns [B, D], treat as already pooled
            emb = enc
        else:
            # pool across patches/positions
            emb = enc.mean(dim=1) if self.reduction == "mean" else enc[:, -1]  # [B, D]

        return emb  # cpu or gpu depending on offload

    def _ensure_proj(self, D: int, Fout: int, device: torch.device, dtype: torch.dtype) -> nn.Linear:
        if (self._proj is None) or (self._proj.in_features != D) or (self._proj.out_features != Fout):
            self._proj = nn.Linear(D, Fout, bias=True).to(device=device, dtype=dtype)
        else:
            self._proj.to(device=device, dtype=dtype)
        return self._proj

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F] -> returns sequence with embeddings integrated:
          - attach='feature'    -> [B,T,F+D]
          - attach='time_token' -> [B,T+1,F]  (D→F projected if proj_to_input_dim)
          - attach='replace'    -> [B,T,D] (or [B,T,F] if projected)
        """
        if x.dim() != 3:
            raise ValueError(f"x must be [B,T,F], got {tuple(x.shape)}")
        B, T, Fin = x.shape

        emb = self._get_embeddings(x)  # [B, D]
        D = emb.size(-1)
        device, dtype = x.device, x.dtype
        emb = emb.to(device=device, dtype=dtype)

        proj_emb = None
        if (self.attach in {"time_token", "replace"}) and self.proj_to_input_dim:
            proj = self._ensure_proj(D, Fin, device=device, dtype=dtype)
            proj_emb = proj(emb)  # [B, Fin]

        if self.attach == "feature":
            rep = emb.unsqueeze(1).repeat(1, T, 1)  # [B,T,D]
            return torch.cat([x, rep], dim=-1)      # [B,T,Fin+D]

        if self.attach == "time_token":
            if self.proj_to_input_dim:
                tok = proj_emb.unsqueeze(1)        # [B,1,Fin]
            else:
                if D == Fin:
                    tok = emb.unsqueeze(1)
                elif D > Fin:
                    tok = emb[:, :Fin].unsqueeze(1)
                else:
                    pad = torch.zeros(B, Fin - D, device=device, dtype=dtype)
                    tok = torch.cat([emb, pad], dim=-1).unsqueeze(1)
            return torch.cat([x, tok], dim=1)       # [B,T+1,Fin]

        # attach == "replace"
        base = proj_emb if (self.proj_to_input_dim and proj_emb is not None) else emb
        return base.unsqueeze(1).repeat(1, T, 1)    # [B,T,Fin or D]


# ──────────────────────────────────────────────────────────────────────────────
# DropoutTS (training-time temporal dropout / span masking)
# ──────────────────────────────────────────────────────────────────────────────


@node(
    type_id="dropoutts_head",
    name="DropoutTSHead",
    category="Preprocessing",
    outputs=["dropoutts_head"],
    color="bg-gradient-to-r from-zinc-500 to-slate-700",
)
class DropoutTSHead(BaseHead):
    """
    Training-only DropoutTS head for time series.

    Modes:
      - "timestep": drop individual timesteps (token dropout)
      - "span":     drop contiguous spans along time
      - "feature":  drop whole features (channels)
      - "mixed":    timestep + feature (and optional spans)
    """

    def __init__(
        self,
        p_time: float = 0.1,
        p_feat: float = 0.0,
        mode: str = "span",
        span_len: int = 8,
        n_spans: int = 1,
        fill: str = "zero",
        scale_keep: bool = False,
    ):
        module = _DropoutTS(
            p_time=p_time,
            p_feat=p_feat,
            mode=mode,
            span_len=span_len,
            n_spans=n_spans,
            fill=fill,
            scale_keep=scale_keep,
        )
        super().__init__(module=module, name="dropoutts")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)


class _DropoutTS(nn.Module):
    def __init__(
        self,
        p_time: float,
        p_feat: float,
        mode: str,
        span_len: int,
        n_spans: int,
        fill: str,
        scale_keep: bool,
    ):
        super().__init__()
        if not (0.0 <= p_time <= 1.0):
            raise ValueError("p_time must be in [0,1]")
        if not (0.0 <= p_feat <= 1.0):
            raise ValueError("p_feat must be in [0,1]")
        if mode not in {"timestep", "span", "feature", "mixed"}:
            raise ValueError("mode must be one of: timestep, span, feature, mixed")
        if fill not in {"zero", "mean"}:
            raise ValueError("fill must be 'zero' or 'mean'")
        self.p_time = float(p_time)
        self.p_feat = float(p_feat)
        self.mode = mode
        self.span_len = int(max(1, span_len))
        self.n_spans = int(max(1, n_spans))
        self.fill = fill
        self.scale_keep = bool(scale_keep)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or (self.p_time == 0.0 and self.p_feat == 0.0):
            return x
        if x.dim() != 3:
            raise ValueError(f"DropoutTS expects [B,T,F], got {tuple(x.shape)}")

        B, T, F_ = x.shape
        device = x.device

        drop = torch.zeros(B, T, F_, dtype=torch.bool, device=device)

        if self.mode in {"timestep", "mixed"} and self.p_time > 0.0:
            dt = torch.rand(B, T, 1, device=device) < self.p_time
            drop |= dt.expand(B, T, F_)

        if self.mode in {"feature", "mixed"} and self.p_feat > 0.0:
            df = torch.rand(B, 1, F_, device=device) < self.p_feat
            drop |= df.expand(B, T, F_)

        if self.mode in {"span", "mixed"} and self.p_time > 0.0:
            L = min(self.span_len, T)
            for _ in range(self.n_spans):
                starts = torch.randint(0, max(1, T - L + 1), (B,), device=device)
                idx = starts.view(B, 1) + torch.arange(L, device=device).view(1, L)  # [B,L]
                idx = idx.clamp(max=T - 1)
                drop.scatter_(1, idx.unsqueeze(-1).expand(B, L, F_), True)

        if self.fill == "zero":
            y = x.masked_fill(drop, 0.0)
        else:
            mu = x.mean(dim=1, keepdim=True)  # [B,1,F]
            y = torch.where(drop, mu.expand_as(x), x)

        if self.scale_keep:
            keep_time = (1.0 - self.p_time) if self.mode in {"timestep", "mixed"} else 1.0
            keep_feat = (1.0 - self.p_feat) if self.mode in {"feature", "mixed"} else 1.0
            keep = max(1e-6, keep_time * keep_feat)
            y = y / keep

        return y
