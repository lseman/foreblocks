# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Optional

import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node

# ──────────────────────────────────────────────────────────────────────────────
# Multi-Scale Pyramid + (optional) Frequency-Domain Filtering + Hierarchical Fuse
# + (optional) Parallel Channel Mixer (feature mixing) summed at the end
# Input/Output: [B, T, F]
# ──────────────────────────────────────────────────────────────────────────────


class _ScaleFFTFilter(nn.Module):
    """
    Per-scale learnable spectral filter:
      y = irfft(rfft(x) * H)
    where H is complex-valued learnable weights over frequency bins per feature.

    x: [B, T, F]
    returns: [B, T, F]
    """

    def __init__(self, feature_dim: int, seq_len: int, init: str = "identity"):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.seq_len = int(seq_len)
        self.freq_bins = self.seq_len // 2 + 1

        # Store complex weights as (real, imag) to avoid dtype/device gotchas.
        # Shape: [F, freq_bins, 2]
        self.H = nn.Parameter(torch.zeros(self.feature_dim, self.freq_bins, 2))

        if init == "identity":
            # identity filter => multiply by 1+0j
            with torch.no_grad():
                self.H[..., 0].fill_(1.0)
                self.H[..., 1].zero_()
        elif init == "small":
            with torch.no_grad():
                self.H[..., 0].fill_(1.0)
                self.H[..., 1].normal_(mean=0.0, std=1e-3)
        else:
            raise ValueError(f"Unknown init={init!r}. Use 'identity' or 'small'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        B, T, Fdim = x.shape
        assert Fdim == self.feature_dim, (Fdim, self.feature_dim)
        assert T == self.seq_len, (T, self.seq_len)

        # rFFT along time dimension
        X = fft.rfft(x, dim=1, norm="ortho")  # [B, freq_bins, F]

        Hr = self.H[..., 0]  # [F, freq_bins]
        Hi = self.H[..., 1]  # [F, freq_bins]
        Hc = torch.complex(Hr, Hi).transpose(0, 1)  # [freq_bins, F]

        Y = X * Hc.unsqueeze(0)  # [B, freq_bins, F]
        y = fft.irfft(Y, n=self.seq_len, dim=1, norm="ortho")  # [B, T, F]
        return y


class _ChannelwiseProj(nn.Module):
    """
    Lightweight channel-wise projection used in hierarchical fusion.
    Keeps channels independent (groups=F) like the paper's "channel independent" fusion spirit.
    Operates on [B, F, T].
    """

    def __init__(self, feature_dim: int, dropout: float = 0.0):
        super().__init__()
        self.dw1x1 = nn.Conv1d(
            feature_dim, feature_dim, kernel_size=1, groups=feature_dim, bias=True
        )
        self.act = nn.GELU()
        self.drop = nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()

        # Start close to identity-ish behavior
        nn.init.zeros_(self.dw1x1.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        # y: [B,F,T]
        return self.drop(self.act(self.dw1x1(y)))


class _ChannelMixer(nn.Module):
    """
    Parallel feature/channel mixer that runs on x in parallel to decomposition.

    Input/Output: [B, T, F]
    Two lightweight implementations:
      - type="mlp": LN + (F -> hidden -> F) per time step
      - type="conv1x1": 1x1 Conv over channels in [B,F,T] space
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        mixer_type: str = "mlp",
        hidden_mult: float = 2.0,
        dropout: float = 0.0,
        residual: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.mixer_type = str(mixer_type).lower()
        self.residual = bool(residual)
        p = float(dropout)

        if self.mixer_type == "mlp":
            hidden = max(4, int(round(self.feature_dim * float(hidden_mult))))
            self.ln = nn.LayerNorm(self.feature_dim)
            self.fc1 = nn.Linear(self.feature_dim, hidden, bias=True)
            self.act = nn.GELU()
            self.drop = nn.Dropout(p) if p > 0 else nn.Identity()
            self.fc2 = nn.Linear(hidden, self.feature_dim, bias=True)

            # mild stability: start close to 0 update
            nn.init.zeros_(self.fc2.bias)

        elif self.mixer_type in ("conv", "conv1x1", "1x1"):
            # Mix channels (groups=1) across feature dim, per time position.
            self.conv = nn.Conv1d(
                self.feature_dim, self.feature_dim, kernel_size=1, bias=True
            )
            self.act = nn.GELU()
            self.drop = nn.Dropout(p) if p > 0 else nn.Identity()
            nn.init.zeros_(self.conv.bias)

        else:
            raise ValueError(
                f"Unknown channel mixer_type={mixer_type!r}. Use 'mlp' or 'conv1x1'."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,T,F]
        if x.ndim != 3 or x.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected [B,T,{self.feature_dim}], got {tuple(x.shape)}")

        if self.mixer_type == "mlp":
            y = self.ln(x)
            y = self.fc1(y)
            y = self.act(y)
            y = self.drop(y)
            y = self.fc2(y)
            return (x + y) if self.residual else y

        # conv1x1 path
        xt = x.transpose(1, 2)  # [B,F,T]
        y = self.conv(xt)
        y = self.act(y)
        y = self.drop(y)
        y = y.transpose(1, 2)  # [B,T,F]
        return (x + y) if self.residual else y


class MultiScalePyramid(nn.Module):
    """
    MSFMoE-like multiscale pyramid head (preprocessing):
      - Recursive AvgPool downsampling to build scales
      - (Optional) per-scale FFT filtering
      - Coarse-to-fine hierarchical fusion with residuals
      - (Optional) Parallel channel mixer branch summed at the end

    Input:  x  [B, T, F]
    Output: y  [B, T, F]

    Notes:
      - This is a *feature* head (same length out), not a predictor head.
      - Fusion uses upsampling (linear) + channel-wise 1x1 projection + residual add.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_scales: int = 4,
        pool_factor: int = 2,
        use_fft_filter: bool = True,
        fft_init: str = "identity",
        fuse_dropout: float = 0.0,
        keep_input_residual: bool = True,
        # ── New: parallel channel mixer branch ────────────────────────────────
        use_channel_mixer: bool = False,
        channel_mixer_type: str = "mlp",  # "mlp" | "conv1x1"
        channel_mixer_hidden_mult: float = 2.0,  # only for mlp
        channel_mixer_dropout: float = 0.0,
        channel_mixer_residual: bool = True,
    ):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.num_scales = int(num_scales)
        self.pool_factor = int(pool_factor)
        self.use_fft_filter = bool(use_fft_filter)
        self.keep_input_residual = bool(keep_input_residual)

        self.use_channel_mixer = bool(use_channel_mixer)
        self._chan_mixer: nn.Module = (
            _ChannelMixer(
                self.feature_dim,
                mixer_type=channel_mixer_type,
                hidden_mult=channel_mixer_hidden_mult,
                dropout=channel_mixer_dropout,
                residual=channel_mixer_residual,
            )
            if self.use_channel_mixer
            else nn.Identity()
        )

        if self.num_scales < 1:
            raise ValueError("num_scales must be >= 1")
        if self.pool_factor < 2:
            raise ValueError("pool_factor must be >= 2 (e.g., 2)")

        # Will be lazily built on first forward because sequence length T is only known then.
        self._built_for_T: Optional[int] = None
        self._filters: nn.ModuleList = nn.ModuleList()
        self._fuse_proj: nn.ModuleList = nn.ModuleList()

        self._fft_init = fft_init
        self._fuse_dropout = float(fuse_dropout)

    def _build(self, T: int, *, device: torch.device):
        self._filters = nn.ModuleList()
        self._fuse_proj = nn.ModuleList()

        # Determine per-scale lengths (must stay >= 2 to make sense)
        lengths: List[int] = []
        cur = int(T)
        for _ in range(self.num_scales):
            lengths.append(cur)
            cur = max(2, cur // self.pool_factor)

        # Filters per scale
        if self.use_fft_filter:
            for Li in lengths:
                self._filters.append(
                    _ScaleFFTFilter(self.feature_dim, Li, init=self._fft_init)
                )
        else:
            for _ in lengths:
                self._filters.append(nn.Identity())

        # Fusion projections from scale i+1 -> i (we will upsample then proj)
        # There are (num_scales-1) fusion steps.
        for _ in range(max(0, self.num_scales - 1)):
            self._fuse_proj.append(
                _ChannelwiseProj(self.feature_dim, dropout=self._fuse_dropout)
            )

        # Runtime-built modules must follow the current execution device.
        self._filters.to(device=device)
        self._fuse_proj.to(device=device)

        self._built_for_T = int(T)

    @staticmethod
    def _avg_pool_time(x: torch.Tensor, factor: int) -> torch.Tensor:
        # x: [B,T,F] -> pool over T
        xt = x.transpose(1, 2)  # [B,F,T]
        xt = F.avg_pool1d(xt, kernel_size=factor, stride=factor, ceil_mode=False)
        return xt.transpose(1, 2)  # [B,T',F]

    @staticmethod
    def _upsample_time(y: torch.Tensor, target_T: int) -> torch.Tensor:
        # y: [B,T,F] -> [B,target_T,F]
        yt = y.transpose(1, 2)  # [B,F,T]
        yt = F.interpolate(yt, size=int(target_T), mode="linear", align_corners=False)
        return yt.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B,T,F]
        """
        if x.ndim != 3:
            raise ValueError(f"Expected x as [B,T,F], got {tuple(x.shape)}")
        B, T, Fdim = x.shape
        if Fdim != self.feature_dim:
            raise ValueError(
                f"feature_dim mismatch: got F={Fdim}, expected {self.feature_dim}"
            )

        if self._built_for_T != int(T):
            self._build(T, device=x.device)
        else:
            # Safety for long-lived modules: if previously built on another device,
            # align runtime-built submodules with the current input device.
            need_move = any(p.device != x.device for p in self._filters.parameters())
            if not need_move:
                need_move = any(
                    p.device != x.device for p in self._fuse_proj.parameters()
                )
            if need_move:
                self._filters.to(device=x.device)
                self._fuse_proj.to(device=x.device)

        # ── Parallel branch: channel mixer (on original x) ────────────────────
        cm = self._chan_mixer(x) if self.use_channel_mixer else torch.zeros_like(x)

        # 1) Build pyramid (scales)
        Xs: List[torch.Tensor] = [x]
        for _ in range(1, self.num_scales):
            Xs.append(self._avg_pool_time(Xs[-1], self.pool_factor))

        # 2) Per-scale spectral filtering (or identity)
        Ys: List[torch.Tensor] = [f(Xi) for f, Xi in zip(self._filters, Xs)]

        # 3) Coarse-to-fine hierarchical fusion:
        #    for i from coarse-1 down to 0:
        #      Y_i = Y_i + Proj( Upsample(Y_{i+1} -> len(Y_i)) )
        for i in reversed(range(self.num_scales - 1)):
            target_T = Ys[i].shape[1]
            up = self._upsample_time(Ys[i + 1], target_T)  # [B,target_T,F]

            # project channel-wise in [B,F,T] space
            up_t = up.transpose(1, 2)  # [B,F,T]
            up_t = self._fuse_proj[i](up_t)  # [B,F,T]
            up = up_t.transpose(1, 2)  # [B,T,F]

            Ys[i] = Ys[i] + up

        ym = Ys[0]

        # ── Sum both paths at the end ─────────────────────────────────────────
        fused = ym + cm
        return (x + fused) if self.keep_input_residual else fused


@node(
    type_id="multiscale_pyramid_head",
    name="MultiScalePyramidHead",
    category="Preprocessing",
    outputs=["multiscale_pyramid_head"],
    color="bg-gradient-to-r from-indigo-500 to-cyan-500",
)
class MultiScalePyramidHead(BaseHead):
    """
    BaseHead wrapper for MultiScalePyramid.
    Forward: [B,T,F] -> [B,T,F]
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        num_scales: int = 5,
        pool_factor: int = 2,
        use_fft_filter: bool = True,
        fft_init: str = "identity",
        fuse_dropout: float = 0.0,
        keep_input_residual: bool = True,
        # ── New: channel mixer options ────────────────────────────────────────
        use_channel_mixer: bool = True,
        channel_mixer_type: str = "mlp",
        channel_mixer_hidden_mult: float = 2.0,
        channel_mixer_dropout: float = 0.0,
        channel_mixer_residual: bool = True,
    ):
        super().__init__(
            module=MultiScalePyramid(
                feature_dim,
                num_scales=num_scales,
                pool_factor=pool_factor,
                use_fft_filter=use_fft_filter,
                fft_init=fft_init,
                fuse_dropout=fuse_dropout,
                keep_input_residual=keep_input_residual,
                use_channel_mixer=use_channel_mixer,
                channel_mixer_type=channel_mixer_type,
                channel_mixer_hidden_mult=channel_mixer_hidden_mult,
                channel_mixer_dropout=channel_mixer_dropout,
                channel_mixer_residual=channel_mixer_residual,
            ),
            name="multiscale_pyramid",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)


# Backward-compatible aliases kept for older imports.
MultiScaleConv = MultiScalePyramid
MultiScaleConvHead = MultiScalePyramidHead
