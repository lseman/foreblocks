# fedformer.py
"""FEDformer-style frequency-enhanced decomposed transformer components.

Based on: Zhou et al., "FEDformer: Frequency Enhanced Decomposed Transformer for
Long-term Series Forecasting", NeurIPS 2022.
Paper: https://arxiv.org/abs/2201.12740
"""

from typing import Literal

import torch
import torch.nn as nn

from foreblocks.layers.norms import create_norm_layer
from foreblocks.modules.attention.modules.frequency_att import FourierBlock


# ---------------------------
# 1) Series decomposition
# ---------------------------


class SeriesDecomp(nn.Module):
    """
    Moving-average decomposition: x = seasonal + trend
    Trend = MA(x, kernel_size), Seasonal = x - Trend.
    """

    def __init__(self, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size should be odd"
        self.kernel_size = kernel_size
        # Depthwise "smoothing" over time
        pad = (kernel_size - 1) // 2
        self.avg = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=pad,
            bias=False,
            groups=1,
        )
        nn.init.constant_(self.avg.weight, 1.0 / kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, T, C]
        Returns: seasonal [B, T, C], trend [B, T, C]
        """
        B, T, C = x.shape
        # apply independently per channel via view+conv
        u = x.permute(0, 2, 1).contiguous().view(B * C, 1, T)  # [B*C,1,T]
        t = self.avg(u)  # [B*C,1,T]
        t = t.view(B, C, T).permute(0, 2, 1).contiguous()  # [B,T,C]
        s = x - t
        return s, t


# ---------------------------
# 2) FEDformer encoder/decoder blocks
# ---------------------------


class FeedForwardBlock(nn.Module):
    def __init__(
        self, d_model: int, dim_ff: int, dropout: float = 0.0, activation: str = "gelu"
    ):
        super().__init__()
        act = (
            nn.GELU()
            if activation.lower() in {"gelu", "geglu", "swiglu"}
            else nn.ReLU()
        )
        self.net = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            act,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


class FEDEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_ff: int = 512,
        modes: int = 32,
        mode_select: Literal["topk", "fixed"] = "topk",
        dropout: float = 0.1,
        custom_norm: str = "rms",
        eps: float = 1e-5,
        activation: str = "gelu",
    ):
        super().__init__()
        self.freq = FourierBlock(d_model, modes, mode_select, dropout, custom_norm, eps)
        self.ffn = FeedForwardBlock(d_model, dim_ff, dropout, activation)
        self.drop = nn.Dropout(dropout)
        self.norm1 = create_norm_layer(custom_norm, d_model, eps)
        self.norm2 = create_norm_layer(custom_norm, d_model, eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.freq(self.norm1(x))
        x = x + self.drop(y)
        y = self.ffn(self.norm2(x))
        return x + self.drop(y)


class FEDDecoderLayer(nn.Module):
    """
    Decoder layer uses Fourier mix on the target side + cross-frequency gating.
    For simplicity, we do: y = Freq(dec) + linear cross from memory.
    """

    def __init__(
        self,
        d_model: int,
        dim_ff: int = 512,
        modes: int = 32,
        mode_select: Literal["topk", "fixed"] = "topk",
        dropout: float = 0.1,
        custom_norm: str = "rms",
        eps: float = 1e-5,
        activation: str = "gelu",
    ):
        super().__init__()
        self.freq = FourierBlock(d_model, modes, mode_select, dropout, custom_norm, eps)
        self.cross_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sigmoid()
        self.ffn = FeedForwardBlock(d_model, dim_ff, dropout, activation)
        self.drop = nn.Dropout(dropout)
        self.norm1 = create_norm_layer(custom_norm, d_model, eps)
        self.norm2 = create_norm_layer(custom_norm, d_model, eps)
        self.normx = create_norm_layer(custom_norm, d_model, eps)

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        tgt:    [B, T_out, C]
        memory: [B, T_in,  C]  (encoded seasonal context)
        """
        # Simple memory pooling (mean) — FEDformer uses cross decomposed info
        m = memory.mean(dim=1, keepdim=True).expand_as(tgt)  # [B,T_out,C]
        m = self.cross_proj(self.normx(m))  # [B,T_out,C]
        # Frequency mixing on normalized tgt
        y = self.freq(self.norm1(tgt))
        # Gated cross fusion
        fused = y * self.gate(m)
        x = tgt + self.drop(fused)
        y = self.ffn(self.norm2(x))
        return x + self.drop(y)


# ---------------------------
# 3) Full FEDformer Head
# ---------------------------


class FEDformer(nn.Module):
    """
    Minimal, faithful FEDformer head with seasonal–trend decomposition and
    frequency-enhanced encoder/decoder mixing.

    Input:
      x: [B, L_in, C_in]

    Output:
      y: [B, pred_len, C_out]
    """

    def __init__(
        self,
        pred_len: int,
        d_model: int = 256,
        n_layers_enc: int = 2,
        n_layers_dec: int = 1,
        dim_ff: int = 512,
        dropout: float = 0.1,
        modes: int = 32,
        mode_select: Literal["topk", "fixed"] = "topk",
        kernel_size: int = 25,  # moving average window (odd)
        in_channels: int = 1,
        out_channels: int = 1,
        activation: str = "gelu",
        custom_norm: str = "rms",
        layer_norm_eps: float = 1e-5,
        use_channel_mixer: bool = True,
        quantiles: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.quantiles = quantiles
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model

        # Decomposition
        self.decomp = SeriesDecomp(kernel_size)

        # Input/Output projections (joint multivariate processing)
        self.enc_in = nn.Linear(in_channels, d_model)
        self.dec_in = nn.Linear(in_channels, d_model)

        # Encoder
        self.encoder = nn.ModuleList([
            FEDEncoderLayer(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                modes=modes,
                mode_select=mode_select,
                custom_norm=custom_norm,
                eps=layer_norm_eps,
                activation=activation,
            )
            for _ in range(n_layers_enc)
        ])

        # Decoder
        self.decoder = nn.ModuleList([
            FEDDecoderLayer(
                d_model=d_model,
                dim_ff=dim_ff,
                dropout=dropout,
                modes=modes,
                mode_select=mode_select,
                custom_norm=custom_norm,
                eps=layer_norm_eps,
                activation=activation,
            )
            for _ in range(n_layers_dec)
        ])

        # Trend forecasting head (linear extrapolation from last trend state)
        self.trend_proj = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                dim_ff,
                out_channels if quantiles is None else out_channels * (len(quantiles)),
            ),
        )

        # Seasonal forecasting head (from decoder outputs)
        self.seasonal_proj = nn.Linear(
            d_model,
            out_channels if quantiles is None else out_channels * (len(quantiles)),
        )

        # Optional channel mixer after sum (kept here for parity with your PatchTST/TFT style)
        self.post_mixer = (
            nn.Linear(out_channels, out_channels, bias=True)
            if (use_channel_mixer and (quantiles is None))
            else nn.Identity()
        )

        # Learnable decoder "queries" in time for seasonal part
        self.query_pos = nn.Parameter(
            torch.randn(1, pred_len, d_model) * (1.0 / d_model**0.5)
        )

        self._reset()

    def _reset(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.query_pos, std=0.02)

    # -------- helper: build decoder seasonal queries --------
    def _build_decoder_queries(self, B: int) -> torch.Tensor:
        return self.query_pos.expand(B, -1, -1)  # [B, pred_len, d_model]

    # -------- forward --------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L_in, C_in]
        """
        if x.dim() != 3:
            raise ValueError(f"Expected [B, L, C], got {tuple(x.shape)}")
        B, L, Cin = x.shape
        if Cin != self.in_channels:
            raise ValueError(
                f"in_channels mismatch: got {Cin}, expected {self.in_channels}"
            )

        # 1) Decompose
        s, t = self.decomp(x)  # seasonal, trend  [B, L, C]

        # 2) Encode seasonal part
        sez = self.enc_in(s)  # [B, L, d]
        for layer in self.encoder:
            sez = layer(sez)  # [B, L, d]
        memory = sez  # [B, L, d]

        # 3) Decode seasonal for pred_len using learned queries
        tgt = self._build_decoder_queries(B)  # [B, H, d]
        # simple start signal: last seasonal token repeated (can be improved)
        start = sez[:, -1:, :].expand(B, self.pred_len, -1)  # [B, H, d]
        tgt = tgt + start

        for layer in self.decoder:
            tgt = layer(tgt, memory)  # [B, H, d]
        sez_out = self.seasonal_proj(tgt)  # [B, H, C_out or C_out*Q]

        # 4) Trend extrapolation: take last trend state -> project across horizon
        tr = self.dec_in(t)  # [B, L, d]
        tr_last = tr[:, -1, :]  # [B, d]
        tr_expand = tr_last.unsqueeze(1).expand(B, self.pred_len, -1)  # [B, H, d]
        tr_out = self.trend_proj(tr_expand)  # [B, H, C_out or C_out*Q]

        # 5) Combine seasonal + trend (both already in output channel space)
        y = sez_out + tr_out  # [B, H, Cout*(Q?)]

        # Optional per-channel mixer (if no quantiles)
        if isinstance(self.post_mixer, nn.Linear) and (self.quantiles is None):
            y = self.post_mixer(y)

        return y

    # Convenience splitter for quantile outputs if configured
    def split_quantiles(self, y: torch.Tensor):
        """
        y: [B, H, C_out * Q] -> {q: [B, H, C_out]}
        """
        assert self.quantiles is not None
        Q = len(self.quantiles)
        B, H, _ = y.shape
        y = y.view(B, H, self.out_channels, Q)
        return {q: y[..., i] for i, q in enumerate(self.quantiles)}
