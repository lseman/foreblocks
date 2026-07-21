"""N-HiTS time-series forecasting head.

Neural hierarchical interpolation with multi-rate pooling and interpolation-based
forecasting. Each block applies MaxPool → MLP → coefficient interpolation,
stacked hierarchically with coarse-to-fine granularity. Weight sharing across
channels via batch folding.

Faithful reimplementation of:

    Challu, Olivares, Oreshkin, Garza, Mergenthaler-Canseco & Dubrawski,
    "N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting",
    AAAI 2023.  Paper: https://arxiv.org/abs/2201.12886

Core API:
- NHiTSBlock: one N-HiTS block with MaxPool → MLP → interpolation
- NHiTS: full N-HiTS head with hierarchical multi-rate stacks
- _interpolate: coefficient interpolation (nearest, linear, cubic)

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ACT = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "tanh": nn.Tanh,
}


def _interpolate(coeff: torch.Tensor, out_len: int, mode: str) -> torch.Tensor:
    if coeff.shape[-1] == out_len:
        return coeff
    x = coeff.unsqueeze(1)  # [B, 1, K]
    if mode == "nearest":
        y = F.interpolate(x, size=out_len, mode="nearest")
    elif mode == "linear":
        y = F.interpolate(x, size=out_len, mode="linear", align_corners=False)
    elif mode == "cubic":
        # cubic via 1D as a degenerate 2D bicubic (H=1)
        y = F.interpolate(
            x.unsqueeze(2), size=(1, out_len), mode="bicubic", align_corners=False
        ).squeeze(2)
    else:
        raise ValueError(f"Unknown interpolation mode: {mode}")
    return y.squeeze(1)


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        input_size: int,  # L (lookback)
        horizon: int,  # H (pred_len)
        n_pool_kernel: int,  # MaxPool kernel/stride (multi-rate sampling)
        n_freq_downsample: int,  # horizon expressivity ratio (#knots = ceil(H/this))
        mlp_units: list[int],
        dropout: float = 0.0,
        activation: str = "relu",
        interpolation_mode: str = "linear",
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_pool_kernel = max(1, int(n_pool_kernel))
        self.interpolation_mode = interpolation_mode

        # number of interpolation knots for backcast / forecast
        self.n_theta_forecast = max(
            1, (horizon + n_freq_downsample - 1) // n_freq_downsample
        )
        self.n_theta_backcast = max(
            1, (input_size + n_freq_downsample - 1) // n_freq_downsample
        )

        self.pool = nn.MaxPool1d(
            kernel_size=self.n_pool_kernel, stride=self.n_pool_kernel, ceil_mode=True
        )
        pooled_len = (input_size + self.n_pool_kernel - 1) // self.n_pool_kernel

        act_cls = _ACT[activation]
        layers: list[nn.Module] = []
        in_dim = pooled_len
        for u in mlp_units:
            layers += [nn.Linear(in_dim, u), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = u
        self.mlp = nn.Sequential(*layers)
        # one head emitting both backcast and forecast coefficients
        self.theta = nn.Linear(in_dim, self.n_theta_backcast + self.n_theta_forecast)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, L]
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)  # [N, pooled_len]
        theta = self.theta(self.mlp(pooled))  # [N, Kb + Kf]
        theta_b = theta[:, : self.n_theta_backcast]
        theta_f = theta[:, self.n_theta_backcast :]
        backcast = _interpolate(theta_b, self.input_size, self.interpolation_mode)
        forecast = _interpolate(theta_f, self.horizon, self.interpolation_mode)
        return backcast, forecast


class NHiTS(nn.Module):
    def __init__(
        self,
        pred_len: int,
        input_size: int,
        stacks: int = 3,
        blocks_per_stack: int = 1,
        n_pool_kernels: list[int] | None = None,
        n_freq_downsample: list[int] | None = None,
        mlp_units: list[int] | None = None,
        dropout: float = 0.0,
        activation: str = "relu",
        interpolation_mode: str = "linear",
    ):
        super().__init__()
        self.pred_len = int(pred_len)
        self.input_size = int(input_size)
        self.stacks = int(stacks)

        if n_pool_kernels is None:
            # coarse → fine, geometric (paper default style)
            n_pool_kernels = [max(1, 2 ** (stacks - 1 - i)) for i in range(stacks)]
        if n_freq_downsample is None:
            n_freq_downsample = [max(1, 2 ** (stacks - 1 - i)) for i in range(stacks)]
        if mlp_units is None:
            mlp_units = [512, 512]
        if not (len(n_pool_kernels) == len(n_freq_downsample) == stacks):
            raise ValueError(
                "n_pool_kernels and n_freq_downsample must have length == stacks"
            )

        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(
                    input_size=self.input_size,
                    horizon=self.pred_len,
                    n_pool_kernel=n_pool_kernels[s],
                    n_freq_downsample=n_freq_downsample[s],
                    mlp_units=mlp_units,
                    dropout=dropout,
                    activation=activation,
                    interpolation_mode=interpolation_mode,
                )
                for s in range(stacks)
                for _ in range(blocks_per_stack)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, L, C], got {tuple(x.shape)}")
        B, L, C = x.shape
        if L != self.input_size:
            raise ValueError(f"input_size={self.input_size} but got lookback L={L}")

        # share weights across channels: fold C into the batch → [B*C, L]
        residual = x.permute(0, 2, 1).reshape(B * C, L)
        forecast = residual.new_zeros(B * C, self.pred_len)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        # [B*C, H] → [B, H, C]
        return forecast.reshape(B, C, self.pred_len).permute(0, 2, 1)
