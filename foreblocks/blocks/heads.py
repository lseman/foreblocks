# heads_all.py
# ──────────────────────────────────────────────────────────────────────────────
# Time-series heads: decomposition, normalization, conv, FFT Top-K, Time2Vec,
# differencing, learnable Fourier seasonal, DAIN, patch embedding, and
# Haar wavelet Top-K. Each wrapped (when appropriate) with BaseHead.
# ──────────────────────────────────────────────────────────────────────────────

import math
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core import BaseHead
from foreblocks.node_spec import node


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

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim or feature_dim
        self.kernel_size = kernel_size

        if groups is None:
            groups = feature_dim  # per-channel filtering

        padding = kernel_size // 2
        self.decomp = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=groups,
            bias=False,
        )
        # Initialize as (approx) moving average
        nn.init.uniform_(self.decomp.weight, a=1.0 / kernel_size, b=1.0 / kernel_size)

        self.post_proj = (
            nn.Linear(feature_dim, self.hidden_dim)
            if self.hidden_dim != feature_dim
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
    """
    BaseHead wrapper for DecompositionHead.
    """

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
# Stabilizes scale/shift per series; provides context to invert predictions.
# ──────────────────────────────────────────────────────────────────────────────


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (per-variable, over time).
    Forward: returns (x_norm, ctx) where ctx has {mu, sigma}.
    invert(x_hat, ctx) -> original scale.
    """

    def __init__(self, num_features: int, affine: bool = True, eps: float = 1e-5):
        super().__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))
        else:
            self.register_buffer("gamma", torch.ones(1, 1, num_features))
            self.register_buffer("beta", torch.zeros(1, 1, num_features))

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
    """
    BaseHead wrapper for RevIN. Forward -> (x_norm, ctx)
    """

    def __init__(self, feature_dim: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(
            module=RevIN(feature_dim, affine=affine, eps=eps), name="revin"
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Conv (depthwise separable convs with multiple kernels + residual)
# Captures local patterns at several temporal scales. Shape-preserving.
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
        self.feature_dim = feature_dim
        self.branches = nn.ModuleList()
        for k in kernels:
            pad = (k // 2) * dilation
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        feature_dim,
                        feature_dim,
                        kernel_size=k,
                        padding=pad,
                        dilation=dilation,
                        groups=feature_dim,
                        bias=False,
                    ),
                    nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True),
                    nn.GELU(),
                )
            )
        self.fuse = nn.Conv1d(
            feature_dim * len(kernels), feature_dim, kernel_size=1, bias=True
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
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
    """
    BaseHead wrapper for MultiScaleConv. Forward -> [B,T,F]
    """

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
# Preserves top-K frequency magnitudes as "seasonal"; main is residual.
# ──────────────────────────────────────────────────────────────────────────────


class FFTTopK(nn.Module):
    """
    Keep top-K magnitudes in frequency domain as seasonal carry; residual is main.
    Input/Output: [B,T,F]. Returns (main, seasonal).
    """

    def __init__(self, topk: int = 8, center: bool = False):
        super().__init__()
        self.topk = topk
        self.center = center

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        Xf = fft.rfft(x, dim=1)  # [B, T_r, F]
        mag = Xf.abs()

        k = min(self.topk, mag.size(1))
        topk_idx = torch.topk(
            mag, k=k, dim=1, largest=True, sorted=False
        ).indices  # [B,k,F]
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_idx, value=True)

        Xf_seasonal = torch.where(mask, Xf, torch.zeros_like(Xf))
        Xf_residual = Xf - Xf_seasonal

        seasonal = fft.irfft(Xf_seasonal, n=T, dim=1)  # [B,T,F]
        main = fft.irfft(Xf_residual, n=T, dim=1)  # [B,T,F]
        return main, seasonal


class FFTTopKHead(BaseHead):
    """
    BaseHead wrapper for FFTTopK. Forward -> (main, seasonal)
    """

    def __init__(self, topk: int = 8):
        super().__init__(module=FFTTopK(topk=topk), name="fft_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time2Vec-style temporal encoding
# Adds linear + periodic components, projects back to F (shape-preserving).
# ──────────────────────────────────────────────────────────────────────────────


class Time2Vec(nn.Module):
    """
    Time2Vec-style periodic features per timestep, projected back to feature_dim.
    Input/Output: [B,T,F]. Keeps dim invariant.
    """

    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__()
        self.feature_dim = feature_dim
        self.k = k
        self.periodic = periodic
        self.freq = nn.Parameter(torch.randn(1, 1, k))
        self.phase = nn.Parameter(torch.zeros(1, 1, k))
        self.proj = nn.Linear(feature_dim + (k if periodic else 0) + 1, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        t = torch.linspace(0, 1, T, device=x.device, dtype=x.dtype).view(1, T, 1)
        parts = [x, t]
        if self.periodic and self.k > 0:
            z = torch.sin(2 * math.pi * (t * self.freq + self.phase))  # [1,T,k]
            z = z.expand(B, -1, -1)  # [B,T,k]
            parts.append(z)
        h = torch.cat(parts, dim=-1)  # [B,T,F + 1 + k]
        return self.proj(h)


class Time2VecHead(BaseHead):
    """
    BaseHead wrapper for Time2Vec. Forward -> [B,T,F]
    """

    def __init__(self, feature_dim: int, k: int = 8, periodic: bool = True):
        super().__init__(module=Time2Vec(feature_dim, k, periodic), name="time2vec")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# First-order Differencing (reversible)
# Improves stationarity; returns (diffs, ctx) and invert(ctx) for reconstruction.
# ──────────────────────────────────────────────────────────────────────────────


class Differencing(nn.Module):
    """
    First-order differencing along time with length preservation.
    Forward: (delta, ctx) where delta[:,0,:]=0 and ctx['x0']=first step.
    """

    def __init__(self):
        super().__init__()

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
    """
    BaseHead wrapper for Differencing. Forward -> (delta, ctx)
    """

    def __init__(self):
        super().__init__(module=Differencing(), name="diff")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Learnable Fourier Seasonal (global periodic bases with learnable weights)
# Returns (main, seasonal). Low-cost global seasonality; K controls richness.
# ──────────────────────────────────────────────────────────────────────────────


class LearnableFourierSeasonal(nn.Module):
    """
    Learnable per-channel seasonal component via Fourier bases (sin/cos up to K).
    seasonal = B @ W, main = x - seasonal. Shapes [B,T,F].
    """

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__()
        self.feature_dim = feature_dim
        self.K = K
        in_dim = 2 * K
        if share_weights:
            self.W = nn.Parameter(torch.randn(1, in_dim))  # shared across features
        else:
            self.W = nn.Parameter(torch.randn(feature_dim, in_dim))  # per-feature
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def _bases(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(-1)  # [T,1]
        ks = torch.arange(1, self.K + 1, device=device, dtype=dtype).view(1, self.K)
        ang = 2 * math.pi * t * ks / float(T)  # [T,K]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        return torch.cat([sin, cos], dim=-1)  # [T,2K]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        Bx = self._bases(T, x.device, x.dtype)  # [T,2K]
        W = self.W.expand(F_, -1) if self.W.size(0) == 1 else self.W  # [F,2K]
        seasonal = Bx @ W.t()  # [T,F]
        seasonal = seasonal.unsqueeze(0).expand(B, -1, -1)  # [B,T,F]
        main = x - seasonal
        return main, seasonal


class LearnableFourierSeasonalHead(BaseHead):
    """
    BaseHead wrapper for LearnableFourierSeasonal. Forward -> (main, seasonal)
    """

    def __init__(self, feature_dim: int, K: int = 8, share_weights: bool = False):
        super().__init__(
            module=LearnableFourierSeasonal(feature_dim, K, share_weights),
            name="lfourier",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# DAIN (Deep Adaptive Input Normalization)
# Adaptive shift/scale/gate from summaries; robustifies inputs.
# ──────────────────────────────────────────────────────────────────────────────


class DAIN(nn.Module):
    """
    Deep Adaptive Input Normalization (DAIN).
    Applies adaptive shift, scale, and gating based on time summaries.
    Input/Output: [B,T,F]
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.W_a = nn.Parameter(torch.eye(feature_dim))
        self.W_b = nn.Parameter(torch.eye(feature_dim))
        self.W_c = nn.Parameter(torch.randn(feature_dim, feature_dim) * 0.01)
        self.d = nn.Parameter(torch.zeros(feature_dim))
        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(
                f"Input feature dim {F_} != expected {self.feature_dim}."
            )

        a = x.mean(dim=1)  # [B,F]
        alpha = torch.matmul(a, self.W_a.t())
        shifted = x - alpha.unsqueeze(1)

        b = torch.sqrt(torch.mean(shifted**2, dim=1) + self.eps)
        beta = torch.matmul(b, self.W_b.t())
        scaled = shifted / (beta.unsqueeze(1) + self.eps)

        c = scaled.mean(dim=1)
        gate_input = torch.matmul(c, self.W_c.t()) + self.d
        gamma = torch.sigmoid(gate_input)
        out = scaled * gamma.unsqueeze(1)
        return out


class DAINHead(BaseHead):
    """
    BaseHead wrapper for DAIN. Forward -> [B,T,F]
    """

    def __init__(self, feature_dim: int):
        super().__init__(module=DAIN(feature_dim=feature_dim), name="dain")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Patch Embedding (local mixing)
# Patchwise depthwise projection + upsample back; residual add. T must be ÷ patch.
# ──────────────────────────────────────────────────────────────────────────────


class PatchEmbed(nn.Module):
    """
    Local patching via depthwise Conv1d, stride=patch_size (non-overlap), then upsample.
    Shape preserved: [B,T,F] -> [B,T,F] (residual added).
    """

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_size = patch_size
        self.patch_proj = nn.Conv1d(
            in_channels=feature_dim,
            out_channels=feature_dim,
            kernel_size=patch_size,
            stride=patch_size,
            groups=feature_dim,
            bias=True,
            padding=0,
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.upsample = nn.Upsample(
            scale_factor=patch_size, mode="linear", align_corners=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(
                f"Input feature dim {F_} != expected {self.feature_dim}."
            )
        if T % self.patch_size != 0:
            raise ValueError(
                f"T={T} must be divisible by patch_size={self.patch_size}."
            )

        xt = x.transpose(1, 2)  # [B,F,T]
        patches = self.patch_proj(xt)  # [B,F,T/patch]
        patches = self.dropout(patches)
        embedded = self.upsample(patches)  # [B,F,T]
        embedded = embedded.transpose(1, 2) + x
        return embedded


@node(
    type_id="patchemb_head",
    name="PatchEmbedHead",
    category="Preprocessing",
    outputs=["patchemb_head"],
    color="bg-gradient-to-r from-blue-400 to-purple-500",
)
class PatchEmbedHead(BaseHead):
    """
    BaseHead wrapper for PatchEmbed. Forward -> [B,T,F]
    """

    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__(
            module=PatchEmbed(feature_dim, patch_size, dropout), name="patchemb"
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Haar Wavelet Top-K (new)
# 1-level Haar analysis; keep Top-K detail coeffs; main = x − detail_sparse.
# Highlights bursts/spikes; auto-pads if T is odd for clean split.
# ──────────────────────────────────────────────────────────────────────────────


class HaarWaveletTopK(nn.Module):
    """
    1-level Haar wavelet analysis with Top-K keep on detail coefficients.
    Returns (main, detail_sparse) with shape [B,T,F].
    """

    def __init__(self, topk: int = 8):
        super().__init__()
        self.topk = topk

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        pad_added = False
        if T % 2 != 0:
            # pad time to even for clean (even, odd) pairing
            x = F.pad(x, (0, 0, 0, 1, 0, 0), mode="replicate")  # pad +1 on time
            B, T, F_ = x.shape
            pad_added = True

        x_even = x[:, 0::2, :]
        x_odd = x[:, 1::2, :]
        x_low = (x_even + x_odd) / math.sqrt(2.0)  # approx  [B,T/2,F]
        x_high = (x_even - x_odd) / math.sqrt(2.0)  # detail  [B,T/2,F]

        # Top-K on |detail| along time per (B,F)
        k = min(self.topk, x_high.size(1))
        if k > 0:
            mag = x_high.abs()  # [B,T/2,F]
            idx = torch.topk(
                mag, k=k, dim=1, largest=True, sorted=False
            ).indices  # [B,k,F]
            mask = torch.zeros_like(mag, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            xh_sparse = torch.where(mask, x_high, torch.zeros_like(x_high))
        else:
            xh_sparse = torch.zeros_like(x_high)

        # Inverse 1-level Haar to original length
        out_len = T
        main = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)
        detail_sparse = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)

        main[:, 0::2, :] = (x_low + torch.zeros_like(xh_sparse)) / math.sqrt(2.0)
        main[:, 1::2, :] = (x_low - torch.zeros_like(xh_sparse)) / math.sqrt(2.0)
        detail_sparse[:, 0::2, :] = (torch.zeros_like(x_low) + xh_sparse) / math.sqrt(
            2.0
        )
        detail_sparse[:, 1::2, :] = (torch.zeros_like(x_low) - xh_sparse) / math.sqrt(
            2.0
        )

        if pad_added:
            main = main[:, :-1, :]
            detail_sparse = detail_sparse[:, :-1, :]

        return main, detail_sparse


class HaarWaveletTopKHead(BaseHead):
    """
    BaseHead wrapper for HaarWaveletTopK. Forward -> (main, detail_sparse)
    """

    def __init__(self, topk: int = 8):
        super().__init__(module=HaarWaveletTopK(topk=topk), name="haar_topk")

    def forward(self, x: torch.Tensor):
        return self.module(x)


# ──────────────────────────────────────────────────────────────────────────────
# Time Attention (self-attention along time per variable)
# Channel-independent: for each feature f, attend over its T timesteps.
# Supports: causal mask, local window attention, and RoPE.
# Input/Output: [B,T,F]
# ──────────────────────────────────────────────────────────────────────────────


class _RoPE1D(nn.Module):
    """Simple 1D rotary positional embedding for time steps."""

    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE.")
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B*F, T, H]   (applies to q and k)
        returns same shape with rotary applied across last dim
        """
        bft, T, H = x.shape
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        freqs = torch.einsum("t,d->td", t, self.inv_freq)  # [T, H/2]
        sin, cos = freqs.sin(), freqs.cos()  # [T, H/2]

        x_odd = x[..., 1::2]
        x_even = x[..., 0::2]
        x_rot_even = x_even * cos - x_odd * sin
        x_rot_odd = x_even * sin + x_odd * cos
        x_rot = torch.stack((x_rot_even, x_rot_odd), dim=-1).flatten(-2)  # interleave
        return x_rot


class TimeAttention(nn.Module):
    """
    Channel-independent Transformer block over time.
      - Projects x:[B,T,F] -> per-feature sequences of dim d_model
      - Self-attention along time (per feature), optional RoPE
      - Optional local window &/or causal mask
      - Residual + FFN, shape-preserving back to F
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
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        self.feature_dim = feature_dim
        self.d_model = d_model
        self.n_heads = n_heads
        self.causal = causal
        self.window = window
        self.use_rope = use_rope

        # Per-time linear maps on feature axis
        self.in_proj = nn.Linear(feature_dim, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, feature_dim, bias=True)

        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        hidden = ffn_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        self.drop = nn.Dropout(dropout)
        self.rope = _RoPE1D(d_model // n_heads) if use_rope else None

    def _build_mask(self, T: int, device, dtype) -> Optional[torch.Tensor]:
        """Returns an attention mask of shape [T,T] with -inf where disallowed, else 0."""
        if not self.causal and self.window is None:
            return None

        mask = torch.zeros(T, T, device=device, dtype=dtype)
        if self.causal:
            # prevent attending to future: j > i => -inf
            i = torch.arange(T, device=device)
            causal_mask = (j := i.unsqueeze(1)) < (i.unsqueeze(0))
            # We want allow i >= j; disallow j>i -> set -inf above diagonal
            mask = mask.masked_fill(
                ~causal_mask
                & (
                    torch.ones_like(mask, dtype=torch.bool)
                    ^ torch.tril(torch.ones_like(mask, dtype=torch.bool))
                ),
                0.0,
            )
            mask = torch.triu(
                torch.full((T, T), float("-inf"), device=device), diagonal=1
            )

        if self.window is not None:
            dist = torch.arange(T, device=device).unsqueeze(1) - torch.arange(
                T, device=device
            ).unsqueeze(0)
            win_mask = dist.abs() > self.window  # True = disallowed
            mask = torch.where(win_mask, torch.full_like(mask, float("-inf")), mask)

        return mask

    def _apply_rope(
        self, q: torch.Tensor, k: torch.Tensor, b: int, f: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q,k: [B*F, T, d_model] -> reshape to [B*F, T, n_heads, head_dim], apply RoPE to last head_dim.
        """
        if self.rope is None:
            return q, k
        head_dim = self.d_model // self.n_heads
        q = q.view(b * f, -1, self.n_heads, head_dim)
        k = k.view(b * f, -1, self.n_heads, head_dim)
        # Apply RoPE per head
        q = torch.stack([self.rope(q[..., h, :]) for h in range(self.n_heads)], dim=2)
        k = torch.stack([self.rope(k[..., h, :]) for h in range(self.n_heads)], dim=2)
        q = q.view(b * f, -1, self.d_model)
        k = k.view(b * f, -1, self.d_model)
        return q, k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F] -> y: [B,T,F]
        """
        B, T, F = x.shape
        if F != self.feature_dim:
            raise RuntimeError(
                f"Input F={F} != configured feature_dim={self.feature_dim}"
            )

        # Project to model dim (time-major features)
        h = self.in_proj(x)  # [B,T,d_model]

        # We perform channel-independent attention by treating features as "batch items".
        # Reshape to [B*F, T, d_model] by first reordering dims:
        # Current h is [B,T,dm] for each feature together; we need per-feature sequences.
        # Easiest: expand feature dimension via 1x1 conv on original x then split per-feature.
        # Instead, we project per-feature by applying linear over last axis already; to make it
        # channel-independent, we chunk along F using broadcasting:
        # Trick: we gather per-feature streams by repeating and masking.
        #
        # Simpler robust approach:
        # - First permute to [B,T,F] (already), then view as (B*F, T, 1) and linearly lift to d_model
        #   using a feature-specific embedding via a learned table. But we already used in_proj above.
        #
        # We'll re-project per feature explicitly for clarity:
        xbf = x.permute(0, 2, 1).contiguous()  # [B,F,T]
        xbf = xbf.view(B * F, T, 1)  # each feature stream
        # Feature embedding table to d_model:
        # Implement as a tiny linear with shared weights per feature stream via expand:
        # To keep parameters minimal & deterministic, reuse the same in_proj applied on
        # one-hot feature indices: we approximate by gathering from original projection 'h'.
        # A cleaner solution: compute per-feature streams from 'h' by slicing along F with a 1x1:
        # We'll redo the projection in a channel-separable way:

        # Channel-separable projection: W_in: [F, d_model]
        # Build on-the-fly parameter from in_proj: in_proj.weight: [d_model, F]
        W_in = self.in_proj.weight.t().contiguous()  # [F, d_model]
        b_in = self.in_proj.bias  # [d_model]
        hcf = (xbf.squeeze(-1) @ W_in) + b_in  # [B*F, T, d_model]

        # LayerNorm + (optionally) RoPE on q,k
        h1 = self.norm1(hcf)  # [B*F, T, d_model]
        q = k = v = h1
        if self.use_rope:
            q, k = self._apply_rope(q, k, B, F)

        attn_mask = self._build_mask(T, h.device, torch.float32)  # float mask for MHA
        # MultiheadAttention (batch_first=True) expects [N, T, E]
        # Here N = B*F
        y, _ = self.attn(q, k, v, attn_mask=attn_mask)

        h2 = hcf + self.drop(y)  # residual
        h3 = self.norm2(h2)
        h4 = h2 + self.ffn(h3)  # second residual

        # Back to [B,T,F]
        ybf = h4.view(B, F, T, self.d_model).mean(
            -1
        )  # collapse model dim -> scalar per (B,F,T)
        y_out = ybf.permute(0, 2, 1).contiguous()  # [B,T,F]

        # Final output projection + residual to keep dim & ease optimization
        y_proj = self.out_proj(self.in_proj(x))  # reuse learned scales
        return x + y_out + y_proj  # robust residual blend


@node(
    type_id="timeattn_head",
    name="TimeAttentionHead",
    category="Preprocessing",
    outputs=["timeattn_head"],
    color="bg-gradient-to-r from-indigo-400 to-cyan-500",
)
class TimeAttentionHead(BaseHead):
    """
    BaseHead wrapper for TimeAttention. Forward -> [B,T,F]

    Example:
        head = TimeAttentionHead(feature_dim=F, d_model=128, n_heads=4,
                                 causal=False, window=64, use_rope=True)
        y = head(x)  # [B,T,F]
    """

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
# Extract per-series embeddings from Chronos-2 via a forward hook.
# Inputs:  x [B,T,F]  (we feed one selected feature/channel to Chronos-2)
# Output:  emb [B,D]  (D is Chronos-2 hidden size, e.g., 768)
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

    Args:
        pipeline: Chronos2Pipeline, loaded externally.
        channel: which feature index from x[:, :, F] to feed as Chronos 'target'.
        reduction: 'mean' | 'last' pooling across Chronos patches.
        hook_layer: 'encoder' | 'input_patch' (where to tap embeddings).
        pred_len: tiny forecast to trigger encoder.
        attach:
            - 'feature'   -> cat([x, repeat(emb, T)], dim=-1) => [B,T,F+D]
            - 'time_token'-> cat([x, proj(emb).unsqueeze(1)], dim=1) => [B,T+1,F]
            - 'replace'   -> repeat(emb, T) => [B,T,D] (or project back to F)
        proj_to_input_dim: if True, project Chronos D → F via Linear for 'time_token'/'replace'.
        offload_cpu: move captured activations to CPU.
        make_dates: optional callable T->pd.DatetimeIndex.

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
        proj_to_input_dim: bool = True,  # project Chronos D -> F for 'time_token'/'replace'
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
        self._model = getattr(pipeline, "model", None) or getattr(
            pipeline, "inner_model", None
        )
        if self._model is None:
            raise RuntimeError("Chronos2EmbedHead: could not locate Chronos2Model.")
        if self.hook_layer == "encoder":
            if not hasattr(self._model, "encoder") or not hasattr(
                self._model.encoder, "final_layer_norm"
            ):
                raise RuntimeError(
                    "Chronos2EmbedHead: encoder.final_layer_norm not found."
                )
            self._hook_module = self._model.encoder.final_layer_norm
        else:
            if not hasattr(self._model, "input_patch_embedding"):
                raise RuntimeError(
                    "Chronos2EmbedHead: input_patch_embedding not found."
                )
            self._hook_module = self._model.input_patch_embedding

        # Lazy init for projector (Chronos D -> input F)
        self._proj = None  # created on first forward once we know D and F

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

        context_df = pd.DataFrame(
            {"id": ids, "timestamp": stamps, "target": tgts}
        ).sort_values(["id", "timestamp"])

        collected = []

        def _hook(_, __, out):
            collected.append(
                out.detach().to("cpu") if self.offload_cpu else out.detach()
            )

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
        enc = torch.cat(collected, dim=0)  # [B, P, D]
        if enc.size(0) != B:
            enc = enc[:B]  # trim if Chronos filtered items
        # pool
        if self.reduction == "mean":
            emb = enc.mean(dim=1)  # [B, D]
        else:
            emb = enc[:, -1]  # [B, D]
        return emb  # cpu or gpu depending on offload

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
        B, T, F = x.shape

        emb = self._get_embeddings(x)  # [B, D]
        D = emb.size(-1)
        device = x.device
        dtype = x.dtype
        emb = emb.to(device=device, dtype=dtype)

        # Projector if needed (lazy)
        if (self.attach in {"time_token", "replace"}) and self.proj_to_input_dim:
            if (
                (self._proj is None)
                or (self._proj.in_features != D)
                or (self._proj.out_features != F)
            ):
                self._proj = nn.Linear(D, F, bias=True).to(device)
            proj_emb = self._proj(emb)  # [B, F]
        else:
            proj_emb = None  # may remain None

        if self.attach == "feature":
            # Repeat embedding across time and concat on features
            rep = emb.unsqueeze(1).repeat(1, T, 1)  # [B,T,D]
            out = torch.cat([x, rep], dim=-1)  # [B,T,F+D]
            return out

        elif self.attach == "time_token":
            # Append a time token (projected to F, or expand emb to F if already D==F and proj disabled)
            if self.proj_to_input_dim:
                tok = proj_emb.unsqueeze(1)  # [B,1,F]
            else:
                if D != F:
                    # Expand D->F by repeating or truncating as a fallback
                    if D > F:
                        tok = emb[:, :F].unsqueeze(1)
                    else:
                        pad = torch.zeros(B, F - D, device=device, dtype=dtype)
                        tok = torch.cat([emb, pad], dim=-1).unsqueeze(1)
                else:
                    tok = emb.unsqueeze(1)  # [B,1,F]
            out = torch.cat([x, tok], dim=1)  # [B,T+1,F]
            return out

        else:  # 'replace'
            # Replace features by repeating the (projected) embedding across time
            if self.proj_to_input_dim:
                base = proj_emb  # [B,F]
            else:
                base = emb  # [B,D] (keeps D features)
            rep = base.unsqueeze(1).repeat(1, T, 1)  # [B,T,F or D]
            return rep
