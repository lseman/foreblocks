from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from foreblocks.core.model import BaseHead
from foreblocks.ui.node_spec import node


class MultiKernelConv(nn.Module):
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
        self.feature_dim = int(feature_dim)
        self.branches = nn.ModuleList()
        for k in kernels:
            k = int(k)
            pad = (k // 2) * int(dilation)
            self.branches.append(
                nn.Sequential(
                    nn.Conv1d(
                        self.feature_dim,
                        self.feature_dim,
                        kernel_size=k,
                        padding=pad,
                        dilation=int(dilation),
                        groups=self.feature_dim,
                        bias=False,
                    ),
                    nn.Conv1d(
                        self.feature_dim, self.feature_dim, kernel_size=1, bias=True
                    ),
                    nn.GELU(),
                )
            )
        self.fuse = nn.Conv1d(
            self.feature_dim * len(kernels), self.feature_dim, kernel_size=1, bias=True
        )
        self.dropout = (
            nn.Dropout(float(dropout)) if float(dropout) > 0 else nn.Identity()
        )
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
    name="MultiKernelConvHead",
    category="Preprocessing",
    outputs=["msconv_head"],
    color="bg-gradient-to-r from-pink-400 to-yellow-500",
)
class MultiKernelConvHead(BaseHead):
    """BaseHead wrapper for MultiKernelConv. Forward -> [B,T,F]."""

    def __init__(
        self,
        feature_dim: int,
        kernels=[3, 5, 7, 11],
        dilation: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__(
            module=MultiKernelConv(feature_dim, kernels, dilation, dropout),
            name="msconv",
        )

    def forward(self, x: torch.Tensor):
        return self.module(x)
