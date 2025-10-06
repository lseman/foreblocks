from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleTemporalConv(nn.Module):
    """
    Parallel multi-dilation temporal conv block.
    Input:  [B, T, C_in]
    Output: [B, T, C_out]
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dilation_rates: Optional[Sequence[int]] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.causal = causal
        self.dilation_rates = tuple(dilation_rates or (1, 2, 4, 8))

        act = self._get_activation(activation)
        drop = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # Build branches
        branches = []
        for d in self.dilation_rates:
            if causal:
                # left-pad manually; conv uses padding=0
                conv = nn.Conv1d(input_size, output_size,
                                 kernel_size=kernel_size, dilation=d, padding=0)
                branch = nn.Sequential(conv, act, drop)
            else:
                # “same-length” padding (for odd kernels) via symmetric pad
                pad = (kernel_size - 1) * d // 2
                conv = nn.Conv1d(input_size, output_size,
                                 kernel_size=kernel_size, dilation=d, padding=pad)
                branch = nn.Sequential(conv, act, drop)
            branches.append(branch)
        self.branches = nn.ModuleList(branches)

        # 1x1 to fuse branches
        self.combiner = nn.Conv1d(output_size * len(self.dilation_rates),
                                  output_size, kernel_size=1)

        # Residual
        self.residual = (
            nn.Conv1d(input_size, output_size, kernel_size=1)
            if input_size != output_size else nn.Identity()
        )

        # LN on last dim after we transpose back to [B, T, C]
        self.layer_norm = nn.LayerNorm(output_size)

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":  return nn.ReLU()
        if name == "gelu":  return nn.GELU()
        if name == "silu":  return nn.SiLU()
        if name == "tanh":  return nn.Tanh()
        return nn.GELU()

    def _causal_pad(self, x: torch.Tensor, dilation: int) -> torch.Tensor:
        # x: [B, C, T]; left-pad only
        pad = (self.kernel_size - 1) * dilation
        return F.pad(x, (pad, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C_in] -> [B, C_in, T]
        x_ch_first = x.transpose(1, 2)
        T = x_ch_first.size(-1)

        outs = []
        for d, branch in zip(self.dilation_rates, self.branches):
            if self.causal:
                xin = self._causal_pad(x_ch_first, d)
                y = branch(xin)                    # padding=0 in conv
                # keep original length exactly
                y = y[..., :T]
            else:
                y = branch(x_ch_first)             # symmetric padding inside conv
            outs.append(y)

        # fuse multi-scale features
        y = torch.cat(outs, dim=1)                 # [B, C_out*k, T]
        y = self.combiner(y)                       # [B, C_out, T]

        # residual
        y = y + self.residual(x_ch_first)

        # back to [B, T, C] and LayerNorm on C
        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        return y
