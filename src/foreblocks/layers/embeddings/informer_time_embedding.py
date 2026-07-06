"""foreblocks.layers.embeddings.informer_time_embedding.

Calendar/time feature embedding for time series forecasting.

Implements the InformerTimeEmbedding from Informer (Zhou et al., AAAI 2021)
— a learned embedding for hour, weekday, day-of-month, and month fields,
concatenated and projected to d_model. Used for time-aware transformer
models where calendar features improve forecasting accuracy.

Core API:
- InformerTimeEmbedding: calendar field embedding with projection

"""

import math

import torch
import torch.nn as nn


class InformerTimeEmbedding(nn.Module):
    """Compact calendar/time embedding with projection and normalized output scale.

    Learns a small embedding for each calendar field, concatenates them, and
    optionally projects to ``d_model``. Based on the temporal embedding used in
    Informer (Zhou et al., AAAI 2021, https://arxiv.org/abs/2012.07436).

    Args:
        d_model: output feature dimension. Must be >= 4 (each of the four
            calendar fields needs a non-empty per-field embedding).
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model < 4:
            raise ValueError(
                f"d_model must be >= 4 for InformerTimeEmbedding, got {d_model}"
            )
        embed_dim = min(d_model // 4, 64)
        self.hour_embed = nn.Embedding(24, embed_dim)
        self.weekday_embed = nn.Embedding(7, embed_dim)
        self.day_embed = nn.Embedding(32, embed_dim)
        self.month_embed = nn.Embedding(13, embed_dim)
        self.projection = (
            nn.Linear(embed_dim * 4, d_model) if embed_dim * 4 != d_model else None
        )
        self.norm_factor = 1.0 / math.sqrt(4.0)

        for emb in [
            self.hour_embed,
            self.weekday_embed,
            self.day_embed,
            self.month_embed,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def forward(self, time_feats: torch.Tensor) -> torch.Tensor:
        """Embed calendar features.

        Args:
            time_feats: tensor of shape ``[..., 4]`` whose last dimension holds
                the calendar fields in this exact order:
                ``[month, weekday, hour, day]``. Values are cast to long and
                clamped to valid ranges (month 0-12, weekday 0-6, hour 0-23,
                day 0-31), so out-of-range inputs are tolerated, not errored.

        Returns:
            Tensor of shape ``[..., d_model]``.
        """
        month = torch.clamp(time_feats[..., 0].long(), 0, 12)
        weekday = torch.clamp(time_feats[..., 1].long(), 0, 6)
        hour = torch.clamp(time_feats[..., 2].long(), 0, 23)
        day = torch.clamp(time_feats[..., 3].long(), 0, 31)
        embs = torch.cat(
            [
                self.month_embed(month),
                self.weekday_embed(weekday),
                self.hour_embed(hour),
                self.day_embed(day),
            ],
            dim=-1,
        )
        if self.projection is not None:
            embs = self.projection(embs)
        return embs * self.norm_factor


__all__ = ["InformerTimeEmbedding"]
