from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TCNPlus(nn.Module):
    """
    Strong TCN baseline:
      - depthwise-separable dilated causal convs with GLU gating
      - residual + skip connections
      - vectorized across nodes: (B,T,N,D) -> (B,T_out,N)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_nodes: int,
        output_len: int,
        num_stacks: int = 2,
        layers_per_stack: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1,
        bottleneck_ratio: float = 0.5,
        use_weight_norm: bool = True,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size

        Hb = max(8, int(hidden_dim * bottleneck_ratio))

        # Input projection
        self.in_proj = nn.Linear(input_dim, hidden_dim)

        # Build stacks of causal TCN blocks
        blocks = []
        for _ in range(num_stacks):
            for i in range(layers_per_stack):
                d = 2 ** i
                blocks.append(self._make_block(hidden_dim, Hb, d, kernel_size, dropout, use_weight_norm))
        self.blocks = nn.ModuleList(blocks)

        # Skip mapping per block (to aggregate multi-scale features)
        self.skip_convs = nn.ModuleList([
            self._pw(hidden_dim, hidden_dim, use_weight_norm) for _ in self.blocks
        ])

        # Output head (from aggregated skip features at the last timestep)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )

        # Final normalization
        self.final_norm = nn.LayerNorm(hidden_dim)

    # ---------- helpers ----------
    @staticmethod
    def _pw(in_ch: int, out_ch: int, wn: bool) -> nn.Conv1d:
        conv = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        return nn.utils.weight_norm(conv) if wn else conv

    @staticmethod
    def _dw(in_ch: int, out_ch: int, dilation: int, kernel_size: int, wn: bool) -> nn.Conv1d:
        # depthwise conv: groups=in_ch; out_ch must be k*in_ch (we’ll use 2*in_ch for GLU)
        conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation,
                         padding=0, groups=in_ch, bias=False)
        return nn.utils.weight_norm(conv) if wn else conv

    def _make_block(self, H: int, Hb: int, dilation: int, kernel_size: int, dropout: float, wn: bool) -> nn.Module:
        # 1x1 reduce -> depthwise dilated (2*Hb for GLU) -> 1x1 expand, with residual
        return nn.ModuleDict({
            "pw_in":  self._pw(H, Hb, wn),
            "dw":     self._dw(Hb, 2*Hb, dilation, kernel_size, wn),  # GLU needs 2*Hb
            "pw_out": self._pw(Hb, H, wn),
            "norm":   nn.GroupNorm(1, H),   # channel-wise norm
            "drop":   nn.Dropout(dropout),
        })

    @staticmethod
    def _causal_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
        # x: (B*N, C, T) -> left-pad only to maintain causality
        if pad <= 0:
            return x
        return F.pad(x, (pad, 0))

    def _block_forward(self, x_bnct: torch.Tensor, block: nn.ModuleDict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_bnct: (B*N, H, T)
        returns: (res_out, skip_feat) with same time length T
        """
        H = x_bnct.size(1)
        T = x_bnct.size(2)

        # Read dilation from the depthwise conv (tuple -> int)
        dilation = block["dw"].dilation[0]
        pad = (self.kernel_size - 1) * dilation

        z = block["pw_in"](x_bnct)                # (B*N, Hb, T)
        z = self._causal_pad(z, pad)
        z = block["dw"](z)                        # (B*N, 2*Hb, T)
        f, g = torch.chunk(z, 2, dim=1)
        z = torch.tanh(f) * torch.sigmoid(g)      # GLU
        z = block["pw_out"](z)                    # (B*N, H, T)
        z = block["drop"](z)
        out = block["norm"](x_bnct + z)           # residual
        skip = z                                   # use the transformed features as skip
        return out, skip

    def forward(self, x, *args):
        """
        x: (B, T_in, N, D_in)
        returns: preds (B, T_out, N), None, {}
        """
        B, T_in, N, D_in = x.shape
        assert N == self.num_nodes, "num_nodes mismatch"

        # Input projection -> (B,T,N,H)
        x = self.in_proj(x)

        # Vectorize over nodes for temporal convs: (B*N, H, T)
        z = x.permute(0, 2, 3, 1).reshape(B * N, self.hidden_dim, T_in)

        skip_sum = None
        for block, skip_conv in zip(self.blocks, self.skip_convs):
            z, skip = self._block_forward(z, block)         # (B*N, H, T), (B*N, H, T)
            s = skip_conv(skip)                              # (B*N, H, T)
            skip_sum = s if skip_sum is None else (skip_sum + s)

        # Use the last timestep of the aggregated skips
        h_last = skip_sum[:, :, -1]                          # (B*N, H)
        h_last = self.final_norm(h_last)

        # Predict T_out per node
        y_bn = self.head(h_last)                             # (B*N, T_out)
        y = y_bn.view(B, N, self.output_len).permute(0, 2, 1).contiguous()  # (B, T_out, N)

        return y, None, {}
