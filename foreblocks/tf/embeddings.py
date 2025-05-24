import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import torch
import torch.nn as nn
import math

from typing import Dict, Union, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding with support for:
    - Standard sinusoidal encoding
    - Scaled injection
    """

    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 10000,
        scale: float = 1.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale

        # Precompute sinusoidal encoding
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(
            1
        )  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        sinusoid = torch.zeros(max_len, d_model, dtype=torch.float32)
        sinusoid[:, 0::2] = torch.sin(position * div_term)
        sinusoid[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", sinusoid.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        if D == self.pe.size(-1):
            # Perfect match
            pe = self.pe[:, :T]
        else:
            # Create PE for this dimension on-the-fly
            pe = self._create_pe_for_dim(D, T, x.device)

        return self.dropout(x + self.scale * pe)


class InformerTimeEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.hour_embed = nn.Embedding(24, d_model)
        self.weekday_embed = nn.Embedding(7, d_model)
        self.day_embed = nn.Embedding(32, d_model)
        self.month_embed = nn.Embedding(13, d_model)

    def forward(self, time_feats):
        """
        time_feats: dict of integer tensors of shape [B, T]
        expected keys: 'hour', 'weekday', 'day', 'month'
        """
        embs = (
            self.hour_embed(time_feats["hour"])
            + self.weekday_embed(time_feats["weekday"])
            + self.day_embed(time_feats["day"])
            + self.month_embed(time_feats["month"])
        )
        return embs  # [B, T, d_model]


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class RotaryEmbedding(nn.Module):
    """
    Rotary position embeddings (RoPE) with caching and partial application support.
    """

    def __init__(self, dim: int, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.base = base
        self.cache = {}

    def _build_cache(self, seq_len: int, device: torch.device):
        if seq_len in self.cache:
            return self.cache[seq_len]

        theta = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2, device=device) / self.dim)
        )  # [dim/2]
        seq_idx = (
            torch.arange(seq_len, device=device).unsqueeze(1).float()
        )  # [seq_len, 1]
        freqs = seq_idx * theta  # [seq_len, dim/2]
        emb = torch.stack((freqs.cos(), freqs.sin()), dim=-1)  # [seq_len, dim/2, 2]

        self.cache[seq_len] = emb
        return emb

    def apply_rotary(self, x, freqs):
        x_even, x_odd = x[..., 0::2], x[..., 1::2]
        cos, sin = freqs[..., 0], freqs[..., 1]
        return torch.cat(
            [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
        ).reshape_as(x)

    def forward(self, q, k, q_pos=None, k_pos=None):
        """
        q, k: [batch, heads, seq_len, head_dim]
        Returns: rotated q, k
        """
        *_, q_len, head_dim = q.shape
        _, _, k_len, _ = k.shape

        rotary_dim = min(head_dim, self.dim)
        q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

        freqs_cis = self._build_cache(max(q_len, k_len), q.device)

        # Position-aware frequency selection
        q_freqs = freqs_cis[q_pos] if q_pos is not None else freqs_cis[:q_len]
        k_freqs = freqs_cis[k_pos] if k_pos is not None else freqs_cis[:k_len]

        q_freqs = q_freqs.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim/2, 2]
        k_freqs = k_freqs.unsqueeze(0).unsqueeze(0)

        q_rot = self.apply_rotary(q_rot, q_freqs)
        k_rot = self.apply_rotary(k_rot, k_freqs)

        q_out = torch.cat([q_rot, q_pass], dim=-1) if q_pass is not None else q_rot
        k_out = torch.cat([k_rot, k_pass], dim=-1) if k_pass is not None else k_rot
        return q_out, k_out


class InformerTimeEmbedding(nn.Module):
    """
    Time embedding similar to Informer paper approach.
    Supports multiple temporal features with learnable embeddings.
    """

    def __init__(
        self,
        d_model: int,
        embed_type: str = "timeF",
        freq: str = "h",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.embed_type = embed_type
        self.freq = freq

        # Time feature dimensions based on frequency
        self.time_dims = self._get_time_dims(freq)

        if embed_type == "timeF":
            # Fixed time features (as used in original Informer)
            self.time_linear = nn.Linear(len(self.time_dims), d_model)
        elif embed_type == "fixed":
            # Learnable embeddings for each time component
            self.embeddings = nn.ModuleDict()
            for name, dim in self.time_dims.items():
                self.embeddings[name] = nn.Embedding(dim, d_model)
        elif embed_type == "learned":
            # Hybrid approach - some fixed, some learned
            self.embeddings = nn.ModuleDict()
            for name, dim in self.time_dims.items():
                if name in ["hour", "weekday", "month"]:
                    self.embeddings[name] = nn.Embedding(dim, d_model)
                else:
                    self.embeddings[name] = nn.Linear(1, d_model)

        self.dropout = nn.Dropout(dropout)

    def _get_time_dims(self, freq: str) -> Dict[str, int]:
        """Get time feature dimensions based on frequency"""
        base_dims = {
            "minute": 60,
            "hour": 24,
            "weekday": 7,
            "day": 32,
            "month": 13,  # 1-12 + padding
            "year": 1,  # normalized
        }

        if freq == "h":  # hourly
            return {
                k: v
                for k, v in base_dims.items()
                if k in ["hour", "weekday", "day", "month"]
            }
        elif freq == "t":  # minute
            return base_dims
        elif freq == "d":  # daily
            return {
                k: v for k, v in base_dims.items() if k in ["weekday", "day", "month"]
            }
        elif freq == "w":  # weekly
            return {k: v for k, v in base_dims.items() if k in ["day", "month"]}
        elif freq == "m":  # monthly
            return {k: v for k, v in base_dims.items() if k in ["month"]}
        else:
            return base_dims

    def forward(
        self, timestamps: Union[torch.Tensor, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        """
        Args:
            timestamps: Either datetime timestamps [B, T] or dict of time features
        Returns:
            Time embeddings [B, T, d_model]
        """
        if isinstance(timestamps, dict):
            time_feats = timestamps
        else:
            time_feats = self._extract_time_features(timestamps)

        if self.embed_type == "timeF":
            # Concatenate all time features and project
            feat_list = []
            for name in sorted(self.time_dims.keys()):
                if name in time_feats:
                    feat_list.append(time_feats[name].unsqueeze(-1))

            time_vec = torch.cat(feat_list, dim=-1).float()  # [B, T, num_features]
            embeddings = self.time_linear(time_vec)  # [B, T, d_model]

        elif self.embed_type == "fixed":
            # Sum all embedding components
            embeddings = None
            for name, embedding_layer in self.embeddings.items():
                if name in time_feats:
                    emb = embedding_layer(time_feats[name])  # [B, T, d_model]
                    embeddings = emb if embeddings is None else embeddings + emb

        elif self.embed_type == "learned":
            # Hybrid approach
            embeddings = None
            for name, embedding_layer in self.embeddings.items():
                if name in time_feats:
                    if isinstance(embedding_layer, nn.Embedding):
                        emb = embedding_layer(time_feats[name])
                    else:  # Linear layer
                        emb = embedding_layer(time_feats[name].unsqueeze(-1).float())

                    embeddings = emb if embeddings is None else embeddings + emb

        return self.dropout(embeddings)

    def _extract_time_features(
        self, timestamps: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract time features from timestamp tensor"""
        # Assuming timestamps are Unix timestamps or datetime objects
        device = timestamps.device
        batch_size, seq_len = timestamps.shape

        time_feats = {}

        # Convert to pandas datetime for easier feature extraction
        if timestamps.dtype in [torch.int64, torch.int32, torch.float32, torch.float64]:
            # Unix timestamp
            timestamps_np = timestamps.cpu().numpy()
            dt_index = pd.to_datetime(timestamps_np.flatten(), unit="s")
        else:
            # Already datetime
            dt_index = pd.to_datetime(timestamps.cpu().numpy().flatten())

        dt_index = dt_index.reshape(batch_size, seq_len)

        # Extract features based on required dimensions
        for name in self.time_dims.keys():
            if name == "minute":
                feat = torch.tensor(
                    [[dt.minute for dt in row] for row in dt_index], device=device
                )
            elif name == "hour":
                feat = torch.tensor(
                    [[dt.hour for dt in row] for row in dt_index], device=device
                )
            elif name == "weekday":
                feat = torch.tensor(
                    [[dt.weekday() for dt in row] for row in dt_index], device=device
                )
            elif name == "day":
                feat = torch.tensor(
                    [[dt.day for dt in row] for row in dt_index], device=device
                )
            elif name == "month":
                feat = torch.tensor(
                    [[dt.month for dt in row] for row in dt_index], device=device
                )
            elif name == "year":
                # Normalize year to [0, 1] range
                years = torch.tensor(
                    [[dt.year for dt in row] for row in dt_index],
                    device=device,
                    dtype=torch.float,
                )
                feat = (years - years.min()) / (years.max() - years.min() + 1e-8)

            time_feats[name] = feat

        return time_feats
