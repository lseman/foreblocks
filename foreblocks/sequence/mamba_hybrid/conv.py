from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.ops.mamba import causal_depthwise_conv1d


class CausalDepthwiseConv1d(nn.Module):
    def __init__(
        self,
        d_inner: int,
        kernel_size: int,
        bias: bool = True,
        conv_init: float | None = None,
    ):
        super().__init__()
        self.d_inner = d_inner
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=kernel_size,
            groups=d_inner,
            bias=bias,
        )
        if conv_init is not None:
            nn.init.uniform_(self.conv.weight, -conv_init, conv_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        weight = self.conv.weight.view(self.d_inner, self.kernel_size).contiguous()
        bias = self.conv.bias.contiguous() if self.conv.bias is not None else None
        x = causal_depthwise_conv1d(x, weight, bias)
        return x.transpose(1, 2)
