"""foreblocks.models.popular.timexer.

TimeXer-style forecasting with endogenous/exogenous variable handling via global tokens.

Based on: Wang et al., "TimeXer: Empowering Transformers for Time Series
Forecasting with Exogenous Variables".
Paper: https://arxiv.org/abs/2402.19072

Patches input into tokens, learns global endogenous tokens, runs patch-wise
self-attention, then uses variate-wise cross-attention from globals into exogenous
tokens for endogenous-exogenous interaction.

Core API:
- TimeXer: full TimeXer model with endogenous/exogenous patching and global-token cross-attention
- TimeXerBlock: self-attention plus exogenous cross-attention through global tokens
- _patchify: convert time series into patch tokens

"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.norms import create_norm_layer


def _num_patches(length: int, patch_len: int, stride: int) -> int:
    if length <= patch_len:
        return 1
    return 1 + (length - patch_len + stride - 1) // stride


def _patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Return patches as [B, C, N, P]."""
    B, L, C = x.shape
    n_patches = _num_patches(L, patch_len, stride)
    target_len = (n_patches - 1) * stride + patch_len
    if target_len > L:
        x = F.pad(x, (0, 0, 0, target_len - L))
    return x.unfold(dimension=1, size=patch_len, step=stride).permute(0, 2, 1, 3)


class TimeXerBlock(nn.Module):
    """Patch self-attention plus exogenous cross-attention through global tokens."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dim_feedforward: int,
        dropout: float,
        norm_type: str,
        eps: float,
    ):
        super().__init__()
        self.self_norm = create_norm_layer(norm_type, d_model, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_q_norm = create_norm_layer(norm_type, d_model, eps=eps)
        self.cross_kv_norm = create_norm_layer(norm_type, d_model, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            d_model,
            n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.global_to_patch = nn.Linear(d_model, d_model)
        self.ff_norm = create_norm_layer(norm_type, d_model, eps=eps)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        n_vars: int,
        n_patches: int,
        exog_tokens: torch.Tensor | None,
    ) -> torch.Tensor:
        residual = tokens
        h = self.self_norm(tokens)
        h, _ = self.self_attn(h, h, h, need_weights=False)
        tokens = residual + h

        globals_ = tokens[:, :n_vars, :]
        patch_tokens = tokens[:, n_vars:, :].reshape(
            tokens.size(0), n_vars, n_patches, tokens.size(-1)
        )

        if exog_tokens is not None and exog_tokens.numel() > 0:
            q = self.cross_q_norm(globals_)
            kv = self.cross_kv_norm(exog_tokens)
            cross, _ = self.cross_attn(q, kv, kv, need_weights=False)
            globals_ = globals_ + cross
            patch_tokens = patch_tokens + self.global_to_patch(globals_).unsqueeze(2)

        tokens = torch.cat([globals_, patch_tokens.flatten(1, 2)], dim=1)
        return tokens + self.ff(self.ff_norm(tokens))


class TimeXer(nn.Module):
    """
    TimeXer forecasting model.

    Args:
        pred_len: Forecast horizon.
        in_channels: Number of endogenous variables in x.
        out_channels: Number of output channels.
        exog_channels: Expected number of exogenous variables. Use 0 for optional
            endogenous-only operation.
    """

    def __init__(
        self,
        pred_len: int,
        in_channels: int = 1,
        out_channels: int = 1,
        exog_channels: int = 0,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 2,
        dim_feedforward: int = 512,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        norm_type: str = "rms",
        eps: float = 1e-5,
        max_patches: int = 1024,
        quantiles: tuple[float, ...] | None = None,
    ):
        super().__init__()
        if patch_len <= 0 or stride <= 0:
            raise ValueError("patch_len and stride must be positive")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.pred_len = int(pred_len)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.exog_channels = int(exog_channels)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.quantiles = quantiles

        d_out = (
            self.out_channels
            if quantiles is None
            else self.out_channels * len(quantiles)
        )

        self.endog_patch = nn.Linear(self.patch_len, d_model)
        self.exog_patch = nn.Linear(self.patch_len, d_model)
        self.endog_var_emb = nn.Parameter(
            torch.randn(1, self.in_channels, 1, d_model) * 0.02
        )
        self.exog_var_emb = nn.Parameter(
            torch.randn(1, max(self.exog_channels, 1), 1, d_model) * 0.02
        )
        self.patch_pos_emb = nn.Parameter(
            torch.randn(1, 1, max_patches, d_model) * 0.02
        )
        self.global_tokens = nn.Parameter(
            torch.randn(1, self.in_channels, d_model) * 0.02
        )
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                TimeXerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    norm_type=norm_type,
                    eps=eps,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = create_norm_layer(norm_type, d_model, eps=eps)
        self.horizon_proj = nn.Linear(d_model, self.pred_len)
        self.channel_mixer = nn.Linear(self.in_channels, d_out)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_endogenous(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        patches = _patchify(x, self.patch_len, self.stride)
        n_patches = patches.size(2)
        if n_patches > self.patch_pos_emb.size(2):
            raise ValueError(
                f"input creates {n_patches} patches, above max_patches={self.patch_pos_emb.size(2)}"
            )
        h = self.endog_patch(patches)
        h = h + self.endog_var_emb + self.patch_pos_emb[:, :, :n_patches, :]
        h = self.dropout(h)
        globals_ = self.global_tokens.expand(x.size(0), -1, -1)
        return torch.cat([globals_, h.flatten(1, 2)], dim=1), n_patches

    def _embed_exogenous(self, exog: torch.Tensor | None) -> torch.Tensor | None:
        if exog is None:
            return None
        if exog.dim() != 3:
            raise ValueError(f"Expected exog [B, L, E], got {tuple(exog.shape)}")
        if self.exog_channels and exog.size(-1) != self.exog_channels:
            raise ValueError(
                f"Expected exog_channels={self.exog_channels}, got {exog.size(-1)}"
            )
        if exog.size(-1) > self.exog_var_emb.size(1):
            raise ValueError(
                "exog has more channels than configured; set exog_channels at init"
            )
        patches = _patchify(exog, self.patch_len, self.stride)
        n_patches = patches.size(2)
        if n_patches > self.patch_pos_emb.size(2):
            raise ValueError(
                f"exog creates {n_patches} patches, above max_patches={self.patch_pos_emb.size(2)}"
            )
        h = self.exog_patch(patches)
        h = (
            h
            + self.exog_var_emb[:, : exog.size(-1), :, :]
            + self.patch_pos_emb[:, :, :n_patches, :]
        )
        return self.dropout(h.flatten(1, 2))

    def forward(
        self,
        x: torch.Tensor,
        exog: torch.Tensor | None = None,
        x_exog: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x: [B, L, C_in] endogenous variables.
        exog/x_exog: optional [B, L, E] exogenous variables.
        """
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected x [B, L, C_in={self.in_channels}], got {tuple(x.shape)}"
            )
        if exog is not None and x_exog is not None:
            raise ValueError("Pass only one of exog or x_exog")

        exog_tokens = self._embed_exogenous(exog if exog is not None else x_exog)
        tokens, n_patches = self._embed_endogenous(x)
        for block in self.blocks:
            tokens = block(tokens, self.in_channels, n_patches, exog_tokens)

        tokens = self.final_norm(tokens)
        globals_ = tokens[:, : self.in_channels, :]
        patches = tokens[:, self.in_channels :, :].reshape(
            x.size(0), self.in_channels, n_patches, self.d_model
        )
        pooled = patches.mean(dim=2) + globals_
        per_var = self.horizon_proj(pooled).transpose(1, 2)
        return self.channel_mixer(per_var)

    def split_quantiles(self, y: torch.Tensor) -> dict[float, torch.Tensor]:
        """Split quantile outputs into a dict keyed by quantile value."""
        if self.quantiles is None:
            raise ValueError("No quantiles configured")
        Q = len(self.quantiles)
        B, H, _ = y.shape
        yq = y.view(B, H, self.out_channels, Q)
        return {q: yq[..., i] for i, q in enumerate(self.quantiles)}
