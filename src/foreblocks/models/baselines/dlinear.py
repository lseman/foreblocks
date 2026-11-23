"""DLinear-style time series forecasting head.

Decomposes input into trend (via moving average) and seasonal components, each
passed through a lazy-built linear projection. Supports per-channel or shared
weights, optional trend decomposition, and channel mixing.

Based on: Zeng et al., "Are Transformers Effective for Time Series Forecasting?" (DLinear),
Paper: https://arxiv.org/abs/2205.13504

Core API:
- DLinear: trend+seasonal decomposition with lazy linear projection heads

"""

import torch
import torch.nn as nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedParameter, is_lazy

from ._smoothing import moving_average


class DLinear(LazyModuleMixin, nn.Module):
    def __init__(
        self,
        pred_len: int,
        in_channels: int | None = None,
        out_channels: int | None = None,
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
        if (
            in_channels is not None
            and out_channels is not None
            and out_channels != in_channels
        ):
            self.channel_mixer = nn.Linear(in_channels, out_channels, bias=True)

        self.W_season = UninitializedParameter()
        self.b_season = UninitializedParameter() if bias else None
        self.W_trend = UninitializedParameter() if use_decomposition else None
        self.b_trend = UninitializedParameter() if use_decomposition and bias else None

    _moving_average = staticmethod(moving_average)

    @torch.no_grad()
    def initialize_parameters(self, x: torch.Tensor) -> None:
        if x.dim() != 3:
            raise ValueError(f"Expected [B,L,C], got {tuple(x.shape)}")
        _, length, channels = x.shape
        output_shape = (channels, self.pred_len) if self.individual else (self.pred_len,)
        for weight, bias in ((self.W_season, self.b_season), (self.W_trend, self.b_trend)):
            if is_lazy(weight):
                weight.materialize((*output_shape, length), device=x.device, dtype=x.dtype)
                nn.init.xavier_uniform_(weight)
            if is_lazy(bias):
                bias.materialize(output_shape, device=x.device, dtype=x.dtype)
                nn.init.zeros_(bias)

    def _proj(
        self, x: torch.Tensor, W: torch.Tensor, b: torch.Tensor | None, individual: bool
    ) -> torch.Tensor:
        # x: [B,L,C] -> [B,T,C]
        if individual:
            y = torch.einsum("blc,ctl->btc", x, W)
            if b is not None:
                y = y + b.transpose(0, 1)  # [T,C]
        else:
            y = torch.einsum("blc,tl->btc", x, W)
            if b is not None:
                y = y + b.unsqueeze(-1)  # [T,1]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B,L,C], got {tuple(x.shape)}")

        if self.use_decomposition:
            trend = self._moving_average(x, self.ma_kernel)
            seasonal = x - trend
            y = self._proj(
                seasonal, self.W_season, self.b_season, self.individual
            ) + self._proj(trend, self.W_trend, self.b_trend, self.individual)
        else:
            y = self._proj(x, self.W_season, self.b_season, self.individual)

        if self.channel_mixer is not None:
            y = self.channel_mixer(y)  # [B,T,C_out]

        return y
