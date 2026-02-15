import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """PatchTST-style temporal patching: [B, T, C] -> [B, T_p, D]."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        assert patch_len >= 1 and patch_stride >= 1
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(
            in_channels,
            embed_dim,
            kernel_size=patch_len,
            stride=patch_stride,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.proj(x).transpose(1, 2)
        return y

    def output_len(self, T: int) -> int:
        return 0 if T < self.patch_len else (T - self.patch_len) // self.patch_stride + 1


class CIPatchEmbedding(nn.Module):
    """Channel-independent grouped patching: [B, T, C] -> [B, T_p, C, D]."""

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_len: int = 16,
        patch_stride: int = 8,
        bias: bool = True,
    ):
        super().__init__()
        assert patch_len >= 1 and patch_stride >= 1
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.proj = nn.Conv1d(
            in_channels,
            in_channels * embed_dim,
            kernel_size=patch_len,
            stride=patch_stride,
            padding=0,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.proj(x)
        B, _, Tp = y.shape
        C, D = self.in_channels, self.embed_dim
        return y.view(B, C, D, Tp).permute(0, 3, 1, 2).contiguous()

    def output_len(self, T: int) -> int:
        return 0 if T < self.patch_len else (T - self.patch_len) // self.patch_stride + 1


__all__ = ["PatchEmbedding", "CIPatchEmbedding"]
