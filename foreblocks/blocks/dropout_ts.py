# dropout_ts.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F


# ----------------------------
# Helpers
# ----------------------------
def _safe_minmax(x: Tensor, dim: int, eps: float = 1e-8) -> Tensor:
    xmin = x.amin(dim=dim, keepdim=True)
    xmax = x.amax(dim=dim, keepdim=True)
    return (x - xmin) / (xmax - xmin + eps)


def _ols_detrend(x: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]:
    """
    Global linear detrending per sample & channel:
      xtrend(t) = w * t + b
      xdetrend = x - xtrend
    x: [B, L, C]
    Returns: xdetrend [B, L, C], xtrend [B, L, C]
    Implements Eq. (2). :contentReference[oaicite:5]{index=5}
    """
    B, L, C = x.shape
    device = x.device
    dtype = x.dtype

    t = torch.arange(L, device=device, dtype=dtype).view(1, L, 1)  # [1,L,1]
    t_mean = t.mean(dim=1, keepdim=True)                           # [1,1,1]
    x_mean = x.mean(dim=1, keepdim=True)                           # [B,1,C]

    t_centered = t - t_mean                                        # [1,L,1]
    x_centered = x - x_mean                                        # [B,L,C]

    # w = cov(t,x)/var(t) per channel
    var_t = (t_centered * t_centered).mean(dim=1, keepdim=True)    # [1,1,1]
    cov_tx = (t_centered * x_centered).mean(dim=1, keepdim=True)   # [B,1,C]
    w = cov_tx / (var_t + eps)                                     # [B,1,C]
    b = x_mean - w * t_mean                                        # [B,1,C]

    xtrend = w * t + b                                             # [B,L,C]
    xdetrend = x - xtrend
    return xdetrend, xtrend


def _spectral_flatness(a: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Spectral Flatness Measure (SFM), Eq. (5). :contentReference[oaicite:6]{index=6}
    a: nonnegative amplitude spectrum, shape [B, F, C]
    Returns: [B, C]
    """
    a = a.clamp_min(eps)
    geo = torch.exp(torch.mean(torch.log(a), dim=1))        # [B,C]
    arith = torch.mean(a, dim=1).clamp_min(eps)             # [B,C]
    return (geo / arith).clamp(0.0, 10.0)                   # keep sane


# ----------------------------
# Spectral Noise Scorer
# ----------------------------
class SpectralNoiseScorer(nn.Module):
    """
    Implements Sec 4.1:
      detrend -> rFFT -> log-norm -> SFM-anchored soft mask -> iFFT -> residual MAE score.
    Produces a per-sample noise score s (Eq. 7). :contentReference[oaicite:7]{index=7}
    """

    def __init__(
        self,
        alpha_init: float = 10.0,  # mask sharpness (softplus(alpha))
        ws_init: float = 1.0,      # affine weight for SFM->threshold
        bs_init: float = 0.0,      # affine bias  for SFM->threshold
        eps: float = 1e-8,
    ):
        super().__init__()
        # Learnable parameters in Eq. (6) and threshold in Eq. (5->6). :contentReference[oaicite:8]{index=8}
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        self.ws = nn.Parameter(torch.tensor(float(ws_init)))
        self.bs = nn.Parameter(torch.tensor(float(bs_init)))
        self.eps = eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        x: [B, L, C]
        Returns:
          s: [B] noise score (residual MAE) Eq. (7)
          p_debug: dict-like Tensor pack (optional usage) -> here we return Ahat.mean for quick debugging
        """
        assert x.dim() == 3, "Expected x as [B, L, C]"
        B, L, C = x.shape

        # (1) Detrend (Eq. 2) :contentReference[oaicite:9]{index=9}
        xdetrend, xtrend = _ols_detrend(x, eps=self.eps)

        # (2) FFT + amplitude (Eq. 3) :contentReference[oaicite:10]{index=10}
        Z = torch.fft.rfft(xdetrend, dim=1)     # [B, F, C], F = L//2 + 1
        A = torch.abs(Z)                       # [B, F, C]

        # (3) Log-scale + instance min-max (Eq. 4) :contentReference[oaicite:11]{index=11}
        Lspec = torch.log1p(A)                 # log(1 + A)
        Ahat = _safe_minmax(Lspec, dim=1, eps=self.eps)  # [B,F,C] in [0,1]

        # (4) SFM anchor + dynamic threshold tau (Eq. 5) :contentReference[oaicite:12]{index=12}
        sfm = _spectral_flatness(A, eps=self.eps)         # [B,C]
        tau = torch.sigmoid(self.ws * sfm + self.bs)      # [B,C]
        tau = tau.unsqueeze(1)                             # [B,1,C] broadcast to [B,F,C]

        # Soft mask (Eq. 6) :contentReference[oaicite:13]{index=13}
        sharp = F.softplus(self.alpha)                    # positive
        M = torch.sigmoid(sharp * (Ahat - tau))           # [B,F,C]

        # (5) Reconstruct: iFFT of masked spectrum + add trend back (Sec 4.1) :contentReference[oaicite:14]{index=14}
        xd_rec = torch.fft.irfft(Z * M, n=L, dim=1)        # [B,L,C]
        x_rec = xd_rec + xtrend                             # [B,L,C]

        # (6) Residual MAE as noise score s (Eq. 7) :contentReference[oaicite:15]{index=15}
        s = (x - x_rec).abs().mean(dim=(1, 2))             # [B]
        # Provide a small extra tensor for debugging/monitoring if desired
        debug = Ahat.mean(dim=(1, 2))                      # [B]
        return s, debug


# ----------------------------
# Sample-Adaptive Dropout (STE)
# ----------------------------
class SampleAdaptiveDropout(nn.Module):
    """
    Implements Sec 4.2:
      - batch-wise min-max normalize s -> ŝ in [0,1]
      - sensitivity curve: s~ = tanh(ŝ * softplus(gamma)) (Eq. 8)
      - p = pmin + (pmax-pmin)*s~ (Eq. 9)
      - STE Bernoulli mask and rescale (Eq. 10-11)
    :contentReference[oaicite:16]{index=16}
    """

    def __init__(
        self,
        p_min: float = 0.05,
        p_max: float = 0.50,
        gamma_init: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert 0.0 <= p_min < p_max < 1.0
        self.p_min = float(p_min)
        self.p_max = float(p_max)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.eps = eps

    def map_score_to_p(self, s: Tensor) -> Tensor:
        """
        s: [B] positive
        returns p: [B] in [pmin, pmax]
        """
        # Batch min-max (Sec 4.2) :contentReference[oaicite:17]{index=17}
        s_min = s.min()
        s_max = s.max()
        shat = (s - s_min) / (s_max - s_min + self.eps)    # [B] in [0,1]

        # Sensitivity curve (Eq. 8) :contentReference[oaicite:18]{index=18}
        sens = F.softplus(self.gamma)
        stilde = torch.tanh(shat * sens)

        # Bound p (Eq. 9) :contentReference[oaicite:19]{index=19}
        p = self.p_min + (self.p_max - self.p_min) * stilde
        return p.clamp(self.p_min, self.p_max)

    def forward(self, h: Tensor, p: Tensor) -> Tensor:
        """
        Apply adaptive dropout to feature map h using per-sample dropout prob p.
        h: [B, ...] any shape with batch dim first
        p: [B]
        Returns: same shape as h
        """
        if not self.training:
            return h

        B = h.shape[0]
        assert p.shape == (B,), f"p should be [B], got {tuple(p.shape)}"
        # Broadcast p to h dims: [B, 1, 1, ...]
        shape = [B] + [1] * (h.dim() - 1)
        p_view = p.view(*shape)

        keep = (1.0 - p_view).clamp_min(self.eps)

        # Bernoulli sampling (discrete) + STE (Eq. 10) :contentReference[oaicite:20]{index=20}
        b = torch.bernoulli(keep)                           # [B,...] broadcasted
        mdrop = b + (keep - keep.detach())

        # Rescale (Eq. 11) :contentReference[oaicite:21]{index=21}
        return h * (mdrop / keep)


# ----------------------------
# DropoutTS plugin wrapper
# ----------------------------
@dataclass
class DropoutTSOutput:
    y: Tensor
    p: Tensor           # [B] dropout probabilities used
    s: Tensor           # [B] noise scores
    debug: Tensor       # [B] optional


class DropoutTS(nn.Module):
    """
    End-to-end DropoutTS plugin:
      - Given raw input window x: [B,L,C], compute noise score s
      - Map to per-sample dropout prob p
      - Provide an `apply(h, p)` that you can use wherever you'd normally call nn.Dropout
    This keeps the task objective unchanged and backprops through p via STE. :contentReference[oaicite:22]{index=22}
    """

    def __init__(
        self,
        p_min: float = 0.05,
        p_max: float = 0.50,
        gamma_init: float = 1.0,
        alpha_init: float = 10.0,
        ws_init: float = 1.0,
        bs_init: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.scorer = SpectralNoiseScorer(
            alpha_init=alpha_init, ws_init=ws_init, bs_init=bs_init, eps=eps
        )
        self.adrop = SampleAdaptiveDropout(
            p_min=p_min, p_max=p_max, gamma_init=gamma_init, eps=eps
        )

    @torch.no_grad()
    def infer_p(self, x: Tensor) -> Tensor:
        """
        Convenience: compute p in eval mode (no dropout applied anyway).
        """
        s, _ = self.scorer(x)
        return self.adrop.map_score_to_p(s)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        x: [B,L,C]
        Returns: (p, s, debug)
        """
        s, debug = self.scorer(x)
        p = self.adrop.map_score_to_p(s)
        return p, s, debug

    def apply(self, h: Tensor, p: Tensor) -> Tensor:
        """
        Apply sample-adaptive dropout to any feature map h (batch-first).
        """
        return self.adrop(h, p)


# ----------------------------
# Example integration pattern
# ----------------------------
class ExampleBackboneWithDropoutTS(nn.Module):
    """
    Minimal example showing how to "replace default dropout with DropoutTS":
      - compute p from the *input window* x
      - pass p to each dropout site
    """

    def __init__(self, in_channels: int, hidden: int, out_horizon: int, p_min=0.05, p_max=0.5):
        super().__init__()
        self.dt = DropoutTS(p_min=p_min, p_max=p_max, gamma_init=1.0)
        self.proj1 = nn.Linear(in_channels, hidden)
        self.proj2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, in_channels * out_horizon)
        self.out_horizon = out_horizon
        self.in_channels = in_channels

    def forward(self, x: Tensor) -> DropoutTSOutput:
        """
        x: [B,L,C]
        y: [B,H,C]
        """
        B, L, C = x.shape
        p, s, debug = self.dt(x)  # p per sample from spectral scorer

        # A toy "encoder": per-time linear -> mean pool
        h = self.proj1(x)         # [B,L,Hid]
        h = F.gelu(h)
        h = self.dt.apply(h, p)   # adaptive dropout site 1

        h = self.proj2(h)
        h = F.gelu(h)
        h = self.dt.apply(h, p)   # adaptive dropout site 2

        h = h.mean(dim=1)         # [B,Hid]
        h = self.dt.apply(h, p)   # adaptive dropout site 3 (works for any [B,...])

        y = self.head(h).view(B, self.out_horizon, self.in_channels)
        return DropoutTSOutput(y=y, p=p, s=s, debug=debug)
