import math

import torch
import torch.nn as nn


class InformerTimeEmbedding(nn.Module):
    """Compact time embedding with projection and normalized output scale."""

    def __init__(self, d_model: int):
        super().__init__()
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
