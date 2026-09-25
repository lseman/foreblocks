"""Shared multiscale waveform encoder for 10 ms acoustic frames.

Each frame in a context window is encoded independently by the same weights
(a downsampling stem followed by stacked `MultiKernelConv` residual blocks),
then pooled across the frame axis by the caller. This follows the "shared
encoder over ten 10 ms frames" design in
`datasets/_paper_review/model_proposal.md`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from foreblocks.nn.heads.blocks.multikernel_conv_head import MultiKernelConv


class WaveformEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int = 32,
        kernels: tuple[int, ...] = (7, 15, 31, 63),
        n_blocks: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, feature_dim, kernel_size=15, stride=4, padding=7),
            nn.GELU(),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=7, stride=2, padding=3),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [
                MultiKernelConv(feature_dim, kernels=list(kernels), dropout=dropout)
                for _ in range(n_blocks)
            ]
        )
        self.output_dim = feature_dim * 2

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """`frames`: `[N, frame_len]` or `[N, 1, frame_len]` raw waveform
        frames (already flattened across batch and frame position by the
        caller). Returns `[N, output_dim]` per-frame embeddings.
        """
        x = frames if frames.dim() == 3 else frames.unsqueeze(1)
        z = self.stem(x)  # [N, feature_dim, T']
        z = z.transpose(1, 2)  # [N, T', feature_dim]
        for block in self.blocks:
            z = block(z)
        z = z.transpose(1, 2)  # [N, feature_dim, T']
        mean = z.mean(dim=-1)
        peak = z.amax(dim=-1)
        return torch.cat([mean, peak], dim=-1)
