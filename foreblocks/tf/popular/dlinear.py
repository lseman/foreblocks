# dlinear_head_custom.py
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

import foreblocks.tf.moe as txmoe
import foreblocks.tf.norms as txaux

# Use your project modules
from foreblocks.tf.embeddings import PositionalEncoding
from foreblocks.tf.multi_att import MultiAttention
from foreblocks.tf.norms import create_norm_layer


class DLinearHeadCustom(nn.Module):
    """
    DLinear-style head.
    Input : x [B, L, C_in]
    Output: y [B, T, C_out]  (C_out=C_in if channel_mixer is None)

    Args
    ----
    pred_len: int             # horizon T
    in_channels: Optional[int]
    out_channels: Optional[int]  # set to enable a C_in->C_out 1x1 mixer
    individual: bool          # per-channel weights (classic) vs shared
    use_decomposition: bool   # moving-average trend + seasonal heads
    ma_kernel: int            # MA window
    bias: bool
    """
    def __init__(
        self,
        pred_len: int,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        individual: bool = True,
        use_decomposition: bool = True,
        ma_kernel: int = 25,
        bias: bool = True,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.individual = individual
        self.use_decomposition = use_decomposition
        self.ma_kernel = max(int(ma_kernel), 1)
        self.bias_flag = bias

        self.channel_mixer = None
        if in_channels is not None and out_channels is not None and out_channels != in_channels:
            self.channel_mixer = nn.Linear(in_channels, out_channels, bias=True)

        self.register_parameter("W_season", None)
        self.register_parameter("b_season", None)
        self.register_parameter("W_trend",  None)
        self.register_parameter("b_trend",  None)

    @staticmethod
    def _moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
        if k <= 1: return x
        B, L, C = x.shape
        x_n = x.permute(0, 2, 1)                      # [B,C,L]
        pad = (k - 1) // 2
        x_pad = F.pad(x_n, (pad, pad), mode="reflect")
        w = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / k
        trend = F.conv1d(x_pad, w.expand(C, -1, -1), groups=C)  # [B,C,L]
        return trend.permute(0, 2, 1)

    def _lazy_build(self, L: int, C: int, device, dtype):
        def make_params():
            if self.individual:
                W = nn.Parameter(torch.empty(C, self.pred_len, L, device=device, dtype=dtype))
                b = nn.Parameter(torch.zeros(C, self.pred_len,      device=device, dtype=dtype)) if self.bias_flag else None
            else:
                W = nn.Parameter(torch.empty(self.pred_len, L, device=device, dtype=dtype))
                b = nn.Parameter(torch.zeros(self.pred_len,    device=device, dtype=dtype)) if self.bias_flag else None
            nn.init.xavier_uniform_(W)
            return W, b

        if self.W_season is None:
            self.W_season, self.b_season = make_params()
        if self.use_decomposition and self.W_trend is None:
            self.W_trend,  self.b_trend  = make_params()

    def _proj(self, x: torch.Tensor, W: torch.Tensor, b: Optional[torch.Tensor], individual: bool) -> torch.Tensor:
        # x: [B,L,C] -> [B,T,C]
        if individual:
            y = torch.einsum("blc,ctl->btc", x, W)
            if b is not None: y = y + b.transpose(0, 1)  # [T,C]
        else:
            y = torch.einsum("blc,tl->btc", x, W)
            if b is not None: y = y + b.unsqueeze(-1)    # [T,1]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B,L,C], got {tuple(x.shape)}")
        B, L, C = x.shape
        self._lazy_build(L, C, x.device, x.dtype)

        if self.use_decomposition:
            trend    = self._moving_average(x, self.ma_kernel)
            seasonal = x - trend
            y = self._proj(seasonal, self.W_season, self.b_season, self.individual) \
              + self._proj(trend,   self.W_trend,  self.b_trend,  self.individual)
        else:
            y = self._proj(x, self.W_season, self.b_season, self.individual)

        if self.channel_mixer is not None:
            y = self.channel_mixer(y)  # [B,T,C_out]

        return y
