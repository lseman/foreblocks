"""Temporal context mixing and attentive statistics over frame embeddings."""

import torch
from torch import nn


class TemporalAttentionPool(nn.Module):
    """Locally mix frame order, then pool attention-weighted mean and deviation.

    Inspired by attentive statistics pooling (Okabe et al., Interspeech 2018).
    The depthwise temporal residual mixer is a Foreblocks-specific extension.
    Inputs [batch, frames, features]; output [batch, 2 * features].
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        self.mixer = nn.Conv1d(
            feature_dim, feature_dim, 3, padding=1, groups=feature_dim
        )
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, max(4, feature_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(4, feature_dim // 2), 1),
        )
        self.output_dim = 2 * feature_dim

    def forward(self, frames):
        if frames.ndim != 3 or frames.shape[1] == 0:
            raise ValueError("Expected [batch, nonempty_frames, features].")
        mixed = frames + torch.nn.functional.gelu(
            self.mixer(frames.transpose(1, 2)).transpose(1, 2)
        )
        weights = torch.softmax(self.attention(mixed), dim=1)
        mean = (weights * mixed).sum(dim=1)
        variance = (weights * (mixed - mean[:, None]).square()).sum(dim=1)
        return torch.cat([mean, variance.clamp_min(1e-8).sqrt()], dim=-1)
