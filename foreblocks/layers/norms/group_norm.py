import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelLastGroupNorm(nn.Module):
    """GroupNorm for channel-last tensors [B, ..., C]."""

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() < 2:
            raise ValueError(
                "ChannelLastGroupNorm expects at least 2D input with channels in the last dim."
            )
        perm = list(range(x.dim()))
        perm.insert(1, perm.pop(-1))
        x_perm = x.permute(perm).contiguous()
        y_perm = F.group_norm(x_perm, self.num_groups, self.weight, self.bias, self.eps)
        inv = list(range(x.dim()))
        inv.append(inv.pop(1))
        return y_perm.permute(inv)


__all__ = ["ChannelLastGroupNorm"]
