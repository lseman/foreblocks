# heads_all.py
# ──────────────────────────────────────────────────────────────────────────────
# Time-series heads: decomposition, normalization, conv, FFT Top-K, Time2Vec,
# differencing, learnable Fourier seasonal, DAIN, patch embedding, and
# Haar wavelet Top-K. Each wrapped (when appropriate) with BaseHead.
# ──────────────────────────────────────────────────────────────────────────────

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core import BaseHead

# ──────────────────────────────────────────────────────────────────────────────
# Decomposition (Autoformer-style moving average via depthwise Conv1d)
# Splits x into (seasonal, trend). Seasonal can go to encoder; add trend back.
# ──────────────────────────────────────────────────────────────────────────────

class DecompositionHead(nn.Module):
    """
    Series decomposition head for trend-seasonal separation.
    Inspired by Autoformer: learnable moving average (depthwise Conv1d) per feature.
    Forward: (seasonal, trend)  with shape [B,T,F].
    """

    def __init__(
        self,
        kernel_size: int = 25,
        feature_dim: int = None,        # Input features F (required)
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

        xt = x.transpose(1, 2)              # [B,F,T] for Conv1d
        trend = self.decomp(xt)             # [B,F,T]
        seasonal = xt - trend               # [B,F,T]
        seasonal = seasonal.transpose(1, 2) # [B,T,F]
        trend = trend.transpose(1, 2)       # [B,T,F]

        seasonal = self.post_proj(seasonal) # optional projection
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
        mu = x.mean(dim=1, keepdim=True)                 # [B,1,F]
        var = x.var(dim=1, unbiased=False, keepdim=True) # [B,1,F]
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


class RevINHead(BaseHead):
    """
    BaseHead wrapper for RevIN. Forward -> (x_norm, ctx)
    """
    def __init__(self, feature_dim: int, affine: bool = True, eps: float = 1e-5):
        super().__init__(module=RevIN(feature_dim, affine=affine, eps=eps), name="revin")

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
                        feature_dim, feature_dim, kernel_size=k, padding=pad,
                        dilation=dilation, groups=feature_dim, bias=False
                    ),
                    nn.Conv1d(feature_dim, feature_dim, kernel_size=1, bias=True),
                    nn.GELU(),
                )
            )
        self.fuse = nn.Conv1d(feature_dim * len(kernels), feature_dim, kernel_size=1, bias=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.zeros_(self.fuse.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xt = x.transpose(1, 2)            # [B,F,T]
        outs = [b(xt) for b in self.branches]
        y = torch.cat(outs, dim=1)        # [B,F*K,T]
        y = self.fuse(y)                  # [B,F,T]
        y = self.dropout(y).transpose(1, 2)
        return x + y


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
        super().__init__(module=MultiScaleConv(feature_dim, kernels, dilation, dropout),
                         name="msconv")

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
        Xf = fft.rfft(x, dim=1)     # [B, T_r, F]
        mag = Xf.abs()

        k = min(self.topk, mag.size(1))
        topk_idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
        mask = torch.zeros_like(mag, dtype=torch.bool)
        mask.scatter_(dim=1, index=topk_idx, value=True)

        Xf_seasonal = torch.where(mask, Xf, torch.zeros_like(Xf))
        Xf_residual = Xf - Xf_seasonal

        seasonal = fft.irfft(Xf_seasonal, n=T, dim=1)  # [B,T,F]
        main = fft.irfft(Xf_residual, n=T, dim=1)      # [B,T,F]
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
            z = z.expand(B, -1, -1)                                    # [B,T,k]
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
            self.W = nn.Parameter(torch.randn(1, in_dim))         # shared across features
        else:
            self.W = nn.Parameter(torch.randn(feature_dim, in_dim))  # per-feature
        nn.init.normal_(self.W, mean=0.0, std=0.02)

    def _bases(self, T: int, device, dtype):
        t = torch.arange(T, device=device, dtype=dtype).unsqueeze(-1)      # [T,1]
        ks = torch.arange(1, self.K + 1, device=device, dtype=dtype).view(1, self.K)
        ang = 2 * math.pi * t * ks / float(T)  # [T,K]
        sin = torch.sin(ang)
        cos = torch.cos(ang)
        return torch.cat([sin, cos], dim=-1)   # [T,2K]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, F_ = x.shape
        Bx = self._bases(T, x.device, x.dtype)        # [T,2K]
        W = self.W.expand(F_, -1) if self.W.size(0) == 1 else self.W  # [F,2K]
        seasonal = Bx @ W.t()                         # [T,F]
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
            raise RuntimeError(f"Input feature dim {F_} != expected {self.feature_dim}.")

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
        self.upsample = nn.Upsample(scale_factor=patch_size, mode="linear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F_ = x.shape
        if F_ != self.feature_dim:
            raise RuntimeError(f"Input feature dim {F_} != expected {self.feature_dim}.")
        if T % self.patch_size != 0:
            raise ValueError(f"T={T} must be divisible by patch_size={self.patch_size}.")

        xt = x.transpose(1, 2)      # [B,F,T]
        patches = self.patch_proj(xt)  # [B,F,T/patch]
        patches = self.dropout(patches)
        embedded = self.upsample(patches)   # [B,F,T]
        embedded = embedded.transpose(1, 2) + x
        return embedded


class PatchEmbedHead(BaseHead):
    """
    BaseHead wrapper for PatchEmbed. Forward -> [B,T,F]
    """
    def __init__(self, feature_dim: int, patch_size: int = 16, dropout: float = 0.0):
        super().__init__(module=PatchEmbed(feature_dim, patch_size, dropout), name="patchemb")

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
        x_odd  = x[:, 1::2, :]
        x_low  = (x_even + x_odd) / math.sqrt(2.0)  # approx  [B,T/2,F]
        x_high = (x_even - x_odd) / math.sqrt(2.0)  # detail  [B,T/2,F]

        # Top-K on |detail| along time per (B,F)
        k = min(self.topk, x_high.size(1))
        if k > 0:
            mag = x_high.abs()  # [B,T/2,F]
            idx = torch.topk(mag, k=k, dim=1, largest=True, sorted=False).indices  # [B,k,F]
            mask = torch.zeros_like(mag, dtype=torch.bool)
            mask.scatter_(1, idx, True)
            xh_sparse = torch.where(mask, x_high, torch.zeros_like(x_high))
        else:
            xh_sparse = torch.zeros_like(x_high)

        # Inverse 1-level Haar to original length
        out_len = T
        main          = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)
        detail_sparse = torch.empty(B, out_len, F_, device=x.device, dtype=x.dtype)

        main[:, 0::2, :]          = (x_low + torch.zeros_like(xh_sparse)) / math.sqrt(2.0)
        main[:, 1::2, :]          = (x_low - torch.zeros_like(xh_sparse)) / math.sqrt(2.0)
        detail_sparse[:, 0::2, :] = (torch.zeros_like(x_low) + xh_sparse) / math.sqrt(2.0)
        detail_sparse[:, 1::2, :] = (torch.zeros_like(x_low) - xh_sparse) / math.sqrt(2.0)

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
# Helpful one-liners to surface in UIs/tooltips (students love these)
# ──────────────────────────────────────────────────────────────────────────────

HEAD_COMMENTS: Dict[str, str] = {
    "decomposition": (
        "Learnable moving-average style trend filter (depthwise Conv1d). "
        "Outputs (seasonal, trend). Feed seasonal to the encoder and add trend back."
    ),
    "revin": (
        "Reversible Instance Norm (per series). Stabilizes scale/shift. "
        "Remember to invert predictions using ctx."
    ),
    "msconv": (
        "Multi-scale depthwise temporal convs + residual. Captures local patterns "
        "at several receptive fields. Good general-purpose front-end."
    ),
    "fft_topk": (
        "Top-K frequency carry. Preserves dominant harmonics explicitly; main is residual. "
        "Great for narrowband seasonality; watch leakage on very short windows."
    ),
    "time2vec": (
        "Time2Vec-style periodic/linear encodings projected back to F. "
        "Low-cost inductive bias for periodic behaviors (k≈4–16)."
    ),
    "diff": (
        "First-order differencing for stationarity. Returns (delta, ctx) and is invertible."
    ),
    "lfourier": (
        "Learnable Fourier seasonal bases (sin/cos up to K). Cheap global periodic modeling. "
        "Returns (main, seasonal). Combine with Decomposition/RevIN."
    ),
    "dain": (
        "Deep Adaptive Input Normalization. Adaptive shift/scale/gate from summaries. "
        "Robustifies inputs under distribution shift."
    ),
    "patchemb": (
        "Local patch mixing (downsample + upsample + residual). "
        "Encourages locality and reduces aliasing for long sequences. T must be divisible by patch_size."
    ),
    "haar_topk": (
        "1-level Haar wavelet with Top-K detail keep. Highlights bursts/spikes/high-freq events. "
        "Auto-pads if T is odd; increase K for richer high-frequency carry."
    ),
}
