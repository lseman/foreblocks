"""PatchTST-style patch-based temporal modeling for anomaly detection.

Patches the input time series into non-overlapping patches,
then applies transformer attention per-patch.
Inspired by PatchTST (ACL 2023) — current SOTA for time series.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.anomaly.models.base import choose_heads


@dataclass
class PatchTSTForward:
    reconstruction: torch.Tensor
    patch_errors: torch.Tensor
    per_patch_scores: torch.Tensor


class PatchTSTForecaster(nn.Module):
    """Patch-based transformer forecaster for anomaly detection.

    Splits windows into patches, encodes with transformer,
    reconstructs or forecasts. Anomaly = per-patch reconstruction error.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        patch_size: int = 4,
        d_model: int = 128,
        n_heads: int | None = None,
        n_layers: int = 2,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        layer_attention_type: str = "standard",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.patch_size = patch_size
        n_patches = self.window_size // self.patch_size
        if n_patches * self.patch_size != self.window_size:
            raise ValueError(
                f"window_size {self.window_size} must be divisible by patch_size {self.patch_size}"
            )

        d_model = d_model
        n_heads = choose_heads(d_model, n_heads)
        dim_feedforward = dim_feedforward or max(4 * d_model, 128)

        # Patch embedding: [batch, n_features, window_size] → [batch, n_patches, d_model]
        self.patch_proj = nn.Linear(
            self.n_features * self.patch_size, d_model
        )
        self.position = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        # Use custom encoder stack from foreblocks
        from foreblocks.anomaly.models.base import ForeblocksEncoderStack
        self.encoder = ForeblocksEncoderStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_attention_type=layer_attention_type,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, self.n_features * self.patch_size),
        )

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Split [batch, window, features] → [batch, n_patches, patch_size*features]."""
        bsz, window, feat = x.shape
        n_patches = window // self.patch_size
        patches = x.reshape(bsz, n_patches, self.patch_size, feat)
        return patches.reshape(bsz, n_patches, -1)

    def _from_patches(self, h: torch.Tensor, bsz: int) -> torch.Tensor:
        """Reconstruct [batch, n_patches, d_model] → [batch, window, features]."""
        h = self.head(h)  # [batch, n_patches, patch_size*features]
        n_patches = h.shape[1]
        return h.reshape(bsz, n_patches, self.patch_size, self.n_features)

    def forward(self, x: torch.Tensor) -> PatchTSTForward:
        bsz = x.shape[0]
        patches = self._to_patches(x)
        h = self.patch_proj(patches) + self.position
        h = self.dropout(h)
        h = self.encoder(h)
        recon_patches = self._from_patches(h, bsz)
        recon = recon_patches.reshape_as(x)

        # Per-patch error
        patch_errors = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))  # per-patch
        patch_scores = patch_errors.mean(dim=1)  # per-window

        return PatchTSTForward(recon, patch_errors, patch_scores)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).per_patch_scores

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(x).reconstruction, x)

    @torch.no_grad()
    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)


class CrossVarTransformer(nn.Module):
    """Cross-attention transformer for multivariate anomaly detection.

    Each variable gets its own embedding path; cross-attention
    models variable interactions. Anomaly = which variables
    deviate from joint distribution.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        n_heads: int | None = None,
        n_layers: int = 2,
        dropout: float = 0.1,
        layer_attention_type: str = "standard",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size

        d_model = d_model
        n_heads = choose_heads(d_model, n_heads)

        from foreblocks.anomaly.models.base import ForeblocksEncoderStack

        # Per-variable encoders
        self.var_encoders = nn.ModuleList(
            ForeblocksEncoderStack(
                d_model=d_model,
                n_heads=max(1, n_heads // max(self.n_features, 1)),
                n_layers=1,
                dim_feedforward=max(d_model * 2, 64),
                dropout=dropout,
                layer_attention_type=layer_attention_type,
            )
            for _ in range(self.n_features)
        )
        # Cross-attention layers
        self.cross_layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=max(1, n_heads),
                dim_feedforward=max(d_model * 4, 128),
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(n_layers)
        )
        self.input_proj = nn.Linear(self.n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.window_size, d_model))
        self.output_proj = nn.Linear(d_model, self.n_features)

    def forward(self, x: torch.Tensor) -> PatchTSTForward:
        bsz = x.shape[0]
        # Encode each variable independently
        var_representations = []
        for i, enc in enumerate(self.var_encoders):
            var_x = x[:, :, i : i + 1]  # [batch, window, 1]
            h = enc(var_x)  # [batch, window, d_model]
            var_representations.append(h)
        # Stack: [batch, n_features, window, d_model] → [batch * n_features, window, d_model]
        stacked = torch.stack(var_representations, dim=1)  # [batch, n_feat, window, d]
        stacked_flat = stacked.reshape(bsz * self.n_features, self.window_size, -1)

        # Cross-attention
        for layer in self.cross_layers:
            stacked_flat = layer(stacked_flat)

        # Reshape and aggregate
        stacked_flat = stacked_flat.reshape(bsz, self.n_features, self.window_size, -1)
        # Mean over features → [batch, window, d_model]
        h = stacked_flat.mean(dim=1)
        h = h + self.position
        recon = self.output_proj(h)

        recon_error = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2))
        return PatchTSTForward(recon, recon_error, recon_error)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).per_patch_scores

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(self(x).reconstruction, x)

    @torch.no_grad()
    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)


class MaskedForecaster(nn.Module):
    """Masked autoencoder forecaster (MAE-style) for self-supervised learning.

    Randomly masks portions of the input sequence, trains model to
    reconstruct masked tokens. Anomaly = high reconstruction error
    on masked tokens.
    """

    def __init__(
        self,
        n_features: int,
        window_size: int,
        d_model: int = 128,
        n_heads: int | None = None,
        n_layers: int = 4,
        dim_feedforward: int | None = None,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
        layer_attention_type: str = "standard",
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.mask_ratio = mask_ratio

        d_model = d_model
        n_heads = choose_heads(d_model, n_heads)
        dim_feedforward = dim_feedforward or max(4 * d_model, 128)

        from foreblocks.anomaly.models.base import ForeblocksEncoderStack

        self.input_proj = nn.Linear(self.n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, self.window_size, d_model))
        self.encoder = ForeblocksEncoderStack(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            layer_attention_type=layer_attention_type,
        )
        # Decoder layers (lighter for masked tokens)
        self.decoder_layers = nn.ModuleList(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=max(1, n_heads),
                dim_feedforward=max(d_model * 4, 128),
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(2)
        )
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.window_size, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, self.n_features)

        # Memory for encoder output (decoder cross-attn)
        self._memory: torch.Tensor | None = None
        self._mask_indices: torch.Tensor | None = None

    def _create_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Create random mask. Returns (masked_x, mask_bool)."""
        bsz, seq, _ = x.shape
        n_masked = int(seq * self.mask_ratio)
        mask_bool = torch.zeros(bsz, seq, device=x.device, dtype=torch.bool)
        for b in range(bsz):
            indices = torch.randperm(seq, device=x.device)[:n_masked]
            mask_bool[b, indices] = True
        masked = x.clone()
        masked[mask_bool] = 0.0
        return masked, mask_bool

    def forward(self, x: torch.Tensor) -> PatchTSTForward:
        bsz = x.shape[0]
        masked, mask_bool = self._create_mask(x)

        # Encode
        h = self.input_proj(masked) + self.position
        h = self.encoder(h)
        self._memory = h.detach()  # Cache for decoder

        # Decode masked tokens
        # Start from mask tokens
        decoded = self.mask_token.expand(bsz, self.window_size, -1)
        decoded = decoded + self.decoder_pos

        for layer in self.decoder_layers:
            decoded = layer(decoded, self._memory)

        recon = self.head(decoded)
        # Only compute loss on masked tokens
        recon_error = F.mse_loss(recon[mask_bool], x[mask_bool], reduction="none").mean(dim=1)

        # Also reconstruct full for scoring
        full_recon = self.head(decoded)

        return PatchTSTForward(full_recon, recon_error, recon_error)

    def reconstruct_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).reconstruction

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # At inference, mask and reconstruct for consistent scoring
        return self(x).per_patch_scores

    def train_step(self, x: torch.Tensor) -> torch.Tensor:
        out = self(x)
        return out.per_patch_scores.mean()

    @torch.no_grad()
    def infer_error(self, x: torch.Tensor) -> torch.Tensor:
        return self.score(x)
