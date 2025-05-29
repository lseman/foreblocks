import math

import torch
import torch.nn as nn


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
        # month = first column
        # weekday = second column
        # hour = third column
        # day = fourth column
        month = time_feats[:, :, 0].long()
        weekday = time_feats[:, :, 1].long()
        hour = time_feats[:, :, 2].long()
        day = time_feats[:, :, 3].long()

        embs = (
            self.month_embed(month)
            + self.weekday_embed(weekday)
            + self.hour_embed(hour)
            + self.day_embed(day)
        ) / 4.0
        return embs  # [B, T, d_model]


class LearnablePositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model,
        max_len=5000,
        dropout=0.1,
        initialization="normal",
        scale_factor=None,
    ):
        super().__init__()

        # Store parameters
        self.d_model = d_model
        self.max_len = max_len
        self.scale_factor = scale_factor or math.sqrt(d_model)

        # Initialize positional embeddings with better initialization
        if initialization == "normal":
            # Xavier/Glorot normal initialization
            self.pe = nn.Parameter(
                torch.randn(1, max_len, d_model) * math.sqrt(2.0 / d_model)
            )
        elif initialization == "uniform":
            # Xavier uniform initialization
            bound = math.sqrt(6.0 / d_model)
            self.pe = nn.Parameter(
                torch.empty(1, max_len, d_model).uniform_(-bound, bound)
            )
        elif initialization == "zero":
            # Zero initialization - let model learn from scratch
            self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        else:
            # Default small random initialization
            self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        # Optional dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, positions=None):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            positions: Optional tensor of shape (batch_size, seq_len) with custom positions
        """
        batch_size, seq_len, _ = x.shape

        if positions is not None:
            # Use custom positions (useful for irregular time series)
            pe = self.pe.expand(batch_size, -1, -1)
            pe = torch.gather(
                pe, 1, positions.unsqueeze(-1).expand(-1, -1, self.d_model)
            )
        else:
            # Standard sequential positions
            pe = self.pe[:, :seq_len, :]

        # Scale positional encoding
        pe = pe * self.scale_factor

        # Add positional encoding
        x = x + pe

        # Apply layer normalization
        x = self.layer_norm(x)

        # Apply dropout if specified
        if self.dropout is not None:
            x = self.dropout(x)

        return x


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
