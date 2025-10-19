# timesnet_head_custom.py
from typing import List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_aux import (
    create_norm_layer,  # your norm factory (supports 'rms', 'temporal', etc.)
)


# ------------------------------------------------------------
# Period estimation (FFT-based autocorrelation top-k periods)
# ------------------------------------------------------------
def _topk_periods(x: torch.Tensor, k: int, min_period: int = 2, max_period: Optional[int] = None) -> torch.Tensor:
    """
    Estimate top-k dominant periods via autocorrelation in frequency domain.
    x: [B, L, C]
    Returns: periods [B, k] (int64)
    """
    B, L, C = x.shape
    # mean-center
    x0 = x - x.mean(dim=1, keepdim=True)
    # FFT along time
    Xf = torch.fft.rfft(x0.float(), dim=1)                 # [B, F, C]
    Sxx = (Xf * torch.conj(Xf)).real.sum(dim=-1)           # [B, F] power
    ac = torch.fft.irfft(Sxx, n=L, dim=1).real             # [B, L]
    # restrict lags
    lo = max(min_period, 1)
    hi = L - 1 if max_period is None else min(max_period, L - 1)
    if lo > hi:
        lo, hi = 1, L - 1
    band = ac[:, lo:hi+1]
    # pick global top-k per batch
    k = min(k, band.size(1))
    idx = torch.topk(band, k=k, dim=1, largest=True).indices + lo  # [B, k], 1..L-1
    return idx

# ------------------------------------------------------------
# Inception-style 2D conv block over (patches x period)
# ------------------------------------------------------------
class Inception2D(nn.Module):
    """
    Multiple kernel sizes in parallel, depthwise-separable conv2d for efficiency.
    Input:  X [B, C, Np, P]
    Output: Y [B, D, Np, P]
    """
    def __init__(self, in_ch: int, out_ch: int, ks: Tuple[int, ...] = (3,5,7), expand: int = 2, dropout: float = 0.0):
        super().__init__()
        hidden = max(in_ch, out_ch) * expand
        self.branches = nn.ModuleList()
        for k in ks:
            pad_h = (k - 1) // 2
            pad_w = (k - 1) // 2
            self.branches.append(
                nn.Sequential(
                    # depthwise
                    nn.Conv2d(in_ch, in_ch, kernel_size=(k, k), padding=(pad_h, pad_w), groups=in_ch, bias=False),
                    nn.GELU(),
                    nn.Conv2d(in_ch, hidden, kernel_size=1, bias=False),
                )
            )
        self.proj = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden * len(ks), out_ch, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [b(x) for b in self.branches]
        y = torch.cat(feats, dim=1)
        return self.proj(y)

# ------------------------------------------------------------
# TimesBlock: pick k periods, fold -> 2D convs -> unfold + fuse
# ------------------------------------------------------------
class TimesBlock(nn.Module):
    """
    A single TimesNet block:
      1) select K dominant periods (per batch)
      2) for each period p: fold [B,L,C] -> [B,C,Np,p]
      3) apply Inception2D over (Np x p)
      4) unfold back to [B,L,C] (trim/pad), sum over K, residual + norm
    """
    def __init__(
        self,
        d_model: int,
        k_periods: int = 3,
        hidden: Optional[int] = None,
        ks: Tuple[int, ...] = (3,5,7),
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_glu_gate: bool = True,
    ):
        super().__init__()
        self.k = k_periods
        self.proj_in = nn.Linear(d_model, d_model)  # channel mixing before conv
        self.inception = Inception2D(d_model, d_model if hidden is None else hidden, ks=ks, expand=expand, dropout=dropout)
        self.proj_out = nn.Linear(d_model if hidden is None else hidden, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = create_norm_layer(norm_type, d_model, eps=layer_norm_eps)
        self.glu = nn.GLU(dim=-1) if use_glu_gate else None

    def _fold(self, x: torch.Tensor, p: int) -> torch.Tensor:
        """
        x: [B, L, C] -> [B, C, Np, p] (pad if necessary)
        """
        B, L, C = x.shape
        Np = (L + p - 1) // p
        pad = Np * p - L
        if pad > 0:
            x = F.pad(x, (0,0,0,pad))  # pad time on the right
        x = x.view(B, Np, p, C).permute(0, 3, 1, 2).contiguous()
        return x, pad

    def _unfold(self, x: torch.Tensor, pad: int, L: int) -> torch.Tensor:
        """
        x: [B, C, Np, p] -> [B, L, C]
        """
        B, C, Np, p = x.shape
        y = x.permute(0, 2, 3, 1).contiguous().view(B, Np * p, C)
        if pad > 0:
            y = y[:, :-pad, :]
        return y[:, :L, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C=d_model]
        """
        B, L, C = x.shape
        res = x

        # (optional) GLU gate on normalized input
        h = self.norm(x)
        if self.glu is not None:
            # project to 2C for GLU: h -> [B,L,2C] -> gate
            h = self.glu(torch.cat([h, h], dim=-1))

        h = self.proj_in(h)

        # select top-k periods per batch (shared across channels for efficiency)
        periods = _topk_periods(h, k=self.k, min_period=2, max_period=L//2 if L >= 4 else None)  # [B,k]

        agg = 0.0
        for i in range(self.k):
            p = periods[:, i].max().item()  # simple choice: use max across batch for consistent folding
            p = max(2, int(p))

            Xp, pad = self._fold(h, p)                   # [B,C,Np,p]
            Yp = self.inception(Xp)                      # [B,D?,Np,p]
            y = self._unfold(Yp, pad, L)                 # [B,L,D?]
            agg = agg + y

        agg = agg / float(self.k)
        out = self.proj_out(self.drop(agg))
        return res + out

# ------------------------------------------------------------
# TimesNet Head
# ------------------------------------------------------------
class TimesNetHeadCustom(nn.Module):
    """
    TimesNet-style forecasting head.
    Input:  x  [B, L_in, C_in]
    Output: y  [B, pred_len, C_out]  (or C_out*Q if quantiles)
    """
    def __init__(
        self,
        pred_len: int,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 512,
        n_blocks: int = 2,
        k_periods: int = 3,
        inception_kernels: Tuple[int, ...] = (3,5,7),
        expand: int = 2,
        dropout: float = 0.1,
        norm_type: str = "rms",                 # try 'temporal' to use your TemporalNorm
        layer_norm_eps: float = 1e-5,
        use_glu_gate: bool = True,
        use_channel_mixer: bool = False,
        quantiles: Optional[Tuple[float, ...]] = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.quantiles = quantiles

        # Input projection (joint multivariate)
        self.enc_in = nn.Linear(in_channels, d_model)

        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(
                d_model=d_model,
                k_periods=k_periods,
                ks=inception_kernels,
                expand=expand,
                dropout=dropout,
                norm_type=norm_type,
                layer_norm_eps=layer_norm_eps,
                use_glu_gate=use_glu_gate,
            )
            for _ in range(n_blocks)
        ])

        # Readout: condense encoder states to horizon with a non-causal linear “head”
        # Strategy: pool (mean + last) -> fuse -> MLP -> produce horizon templates, then affine per step.
        self.pool_mean = nn.AdaptiveAvgPool1d(1)
        self.pool_last = lambda z: z[:, -1:, :]  # [B,1,D]

        d_out = out_channels if quantiles is None else out_channels * len(quantiles)

        self.horizon_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * d_model),
        )
        self.step_proj = nn.Linear(d_model, d_out)

        self.post_mixer = nn.Linear(out_channels, out_channels) if (use_channel_mixer and quantiles is None) else nn.Identity()

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C_in]
        """
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(f"Expected x [B, L, C_in={self.in_channels}], got {tuple(x.shape)}")
        B, L, Cin = x.shape

        z = self.enc_in(x)                 # [B,L,D]
        for blk in self.blocks:
            z = blk(z)                     # [B,L,D]

        # summary token (mean + last)
        z_mean = self.pool_mean(z.permute(0,2,1)).permute(0,2,1)   # [B,1,D]
        z_last = self.pool_last(z)                                 # [B,1,D]
        z_sum = torch.cat([z_mean, z_last], dim=-1).squeeze(1)     # [B, 2D]

        # produce horizon templates [B, H, D]
        htem = self.horizon_mlp(z_sum).view(B, self.pred_len, -1)  # [B,H,D]
        y = self.step_proj(htem)                                   # [B,H,C_out*(Q?)]

        if isinstance(self.post_mixer, nn.Linear) and (self.quantiles is None):
            y = self.post_mixer(y)                                 # [B,H,C_out]

        return y

    # Utility to split quantiles if configured
    def split_quantiles(self, y: torch.Tensor):
        assert self.quantiles is not None
        Q = len(self.quantiles)
        B, H, _ = y.shape
        y = y.view(B, H, self.out_channels, Q)
        return {q: y[..., i] for i, q in enumerate(self.quantiles)}
