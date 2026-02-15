from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_padding(
    x: torch.Tensor, kernel_size: int, dilation: int = 1
) -> torch.Tensor:
    """Left-pad only (causal)."""
    pad = (kernel_size - 1) * dilation
    return F.pad(x, (pad, 0)) if pad > 0 else x


from foreblocks.ui.node_spec import node


class CausalTCNBlock(nn.Module):
    """One dilated causal depthwise-separable block with gating."""

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.1,
        expansion: float = 2.0,
        use_weight_norm: bool = True,
        activation: str = "gelu",
    ):
        super().__init__()
        mid_channels = int(channels * expansion)

        # 1×1 expansion
        self.conv_in = nn.Conv1d(channels, mid_channels, 1, bias=False)
        # Depthwise dilated causal
        self.dw = nn.Conv1d(
            mid_channels,
            mid_channels * 2,  # for gating
            kernel_size=kernel_size,
            dilation=dilation,
            groups=mid_channels,
            padding=0,
            bias=False,
        )
        # 1×1 projection back
        self.conv_out = nn.Conv1d(mid_channels, channels, 1, bias=False)

        if use_weight_norm:
            self.conv_in = nn.utils.weight_norm(self.conv_in)
            self.dw = nn.utils.weight_norm(self.dw)
            self.conv_out = nn.utils.weight_norm(self.conv_out)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(channels)  # most common in recent TCNs
        self.act = nn.GELU() if activation == "gelu" else nn.SiLU()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B*N, C, T)
        Returns:
            residual out: (B*N, C, T)
            skip:         (B*N, C, T)
        """
        res = x
        z = self.conv_in(x)
        z = causal_padding(z, self.dw.kernel_size[0], self.dw.dilation[0])
        z = self.dw(z)  # (B*N, 2*mid, T')

        gate, val = z.chunk(2, dim=1)
        z = val * torch.sigmoid(gate)  # GLU-style gating (no tanh → better gradients)

        z = self.conv_out(z)
        z = self.dropout(z)

        out = self.norm(res + z)
        out = self.act(out)

        return out, z  # z = skip connection


@node(
    type_id="tcn_plus",
    name="TCN+",
    category="Backbone",
    color="bg-gradient-to-br from-emerald-600 to-emerald-800",
)
class TCNPlus(nn.Module):
    """
    Improved strong TCN baseline for multivariate / spatiotemporal forecasting.
    Uses full skip aggregation + final temporal reduction.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        output_len: int,
        kernel_size: int = 3,
        num_levels: int = 8,  # controls receptive field
        stacks: int = 2,
        expansion: float = 2.0,
        dropout: float = 0.1,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.hidden_dim = hidden_dim

        # Input lifting
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Build blocks
        self.blocks = nn.ModuleList()
        dilation = 1
        for _ in range(stacks):
            for _ in range(num_levels):
                self.blocks.append(
                    CausalTCNBlock(
                        channels=hidden_dim,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                        expansion=expansion,
                        use_weight_norm=use_weight_norm,
                    )
                )
                dilation *= 2

        # Skip aggregation (1×1 per block)
        self.skip_projs = nn.ModuleList(
            [nn.Conv1d(hidden_dim, hidden_dim, 1, bias=False) for _ in self.blocks]
        )
        if use_weight_norm:
            for m in self.skip_projs:
                nn.utils.weight_norm(m)

        # Final temporal reduction + head
        self.reduction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, output_len)

        self.final_norm = nn.LayerNorm(hidden_dim)

        # Approximate receptive field (for logging / debugging)
        rf = 1 + (kernel_size - 1) * (dilation - 1) // (dilation // 2)
        print(f"TCNPlus initialized — approx. receptive field: {rf} steps")

    def forward(
        self, x: torch.Tensor, *args
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        x: (B, T_in, N, D_in)
        Returns: (B, T_out, N), None, {}
        """
        B, T_in, N, D_in = x.shape

        # Project input features
        x = self.in_proj(x)  # (B, T, N, H)

        # Vectorize nodes → treat as batch
        z = x.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, T_in)

        skip_sum = 0.0
        for block, skip_proj in zip(self.blocks, self.skip_projs):
            z, skip = block(z)  # z: residual, skip: features
            skip_sum += skip_proj(skip)  # accumulate all scales

        # Final normalization on aggregated features
        h = self.final_norm(skip_sum.permute(0, 2, 1))  # (B*N, T, H) → norm over H

        # Take last timestep or global average — here: last (common in autoregressive TCN)
        h_last = h[:, -1, :]  # (B*N, H)

        # Optional: reduction MLP
        h_last = self.reduction(h_last)

        # Final linear prediction
        y_bn = self.head(h_last)  # (B*N, T_out)

        y = y_bn.view(B, N, self.output_len).permute(0, 2, 1)  # (B, T_out, N)

        return y, None, {}


# Example instantiation
if __name__ == "__main__":
    model = TCNPlus(
        input_dim=7,
        hidden_dim=256,
        num_nodes=321,
        output_len=96,
        kernel_size=5,
        num_levels=8,
        stacks=3,
        expansion=2.0,
    )

    x = torch.randn(16, 336, 321, 7)
    y, _, _ = model(x)
    print(y.shape)  # torch.Size([16, 96, 321])
