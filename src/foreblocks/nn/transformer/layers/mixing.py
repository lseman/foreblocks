"""Separable sequence and variate attention for multivariate embeddings.

This module implements the mixing block used by TimesFM 3: inputs are first
mixed independently along time, then independently across variates at every
time step.  Keeping the two axes separate avoids flattening variables into the
sequence axis and gives callers explicit control over temporal causality.

Architecture per layer (pre-norm):

    1. Sequence attention:  pre_norm -> MHA -> post_norm + residual
    2. Variate attention:   pre_norm -> MHA -> post_norm + residual
    3. FFN:                 pre_norm -> MLP -> post_norm + residual

Each attention block reshapes the input tensor to treat its axis as the
sequence dimension, so the attention backend receives a 3-D tensor.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.normalization import create_norm_layer
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.transformer.layers.axis_attention import (
    make_sequence_block,
    make_variate_block,
)


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    activations = {
        "gelu": F.gelu,
        "relu": F.relu,
        "silu": F.silu,
        "swish": F.silu,
    }
    try:
        return activations[name.lower()]
    except KeyError as error:
        supported = ", ".join(sorted(activations))
        raise ValueError(
            f"unsupported mixing-transformer activation {name!r}; "
            f"expected one of: {supported}"
        ) from error


class MixingTransformer(nn.Module):
    """A transformer layer with sequential sequence and variate attention.

    The input and output shape is ``[batch, variates, sequence, d_model]``.
    Sequence attention runs over ``[batch * variates, sequence, d_model]``;
    variate attention then runs over ``[batch * sequence, variates, d_model]``.

    ``patch_mask`` follows the rest of Foreblocks: ``True`` marks a padded or
    otherwise blocked position.  It masks keys on both attention axes.

    Parameters are configured via :class:`TransformerConfig`.  When
    ``variate_attention=False`` the layer falls back to sequence-only
    attention (the variate sub-block is skipped entirely).
    """

    def __init__(
        self, config: TransformerConfig | None = None, **overrides: Any
    ) -> None:
        super().__init__()
        config = TransformerConfig.resolve(config, **overrides)
        self.config = config

        # --- Sequence attention ---
        seq_block = make_sequence_block(config, dropout=config.dropout)
        self.seq_attn = seq_block.attention
        self.pre_seq_attn_norm = seq_block.pre_norm
        self.post_seq_attn_norm = seq_block.post_norm

        # --- Variate attention ---
        if config.variate_attention:
            var_block = make_variate_block(
                config,
                variate_position_encoding=config.variate_position,
                dropout=config.dropout,
            )
            self.var_attn = var_block.attention
            self.pre_var_attn_norm = var_block.pre_norm
            self.post_var_attn_norm = var_block.post_norm
        else:
            self.var_attn = None
            self.pre_var_attn_norm = None
            self.post_var_attn_norm = None

        # --- FFN ---
        d_model = config.d_model
        norm = lambda: create_norm_layer(config.norm, d_model, config.norm_eps)  # noqa: E731
        self.pre_ff_norm = norm()
        self.post_ff_norm = norm()
        self.ff0 = nn.Linear(d_model, config.ff_dim)
        self.ff1 = nn.Linear(config.ff_dim, d_model)
        self.ff_dropout = nn.Dropout(config.dropout)
        self.activation = _activation(config.activation)

    # ------------------------------------------------------------------
    # Static helpers used by the encoder when it needs advanced masks
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor | None,
    ) -> tuple[int, int, int, int]:
        """Return ``(b, v, n, d)`` after shape validation."""
        if input_embeddings.ndim != 4:
            raise ValueError(
                "input_embeddings must have shape [B, V, N, D], got "
                f"{tuple(input_embeddings.shape)}"
            )
        b, v, n, d = input_embeddings.shape
        if b == 0 or v == 0 or n == 0:
            raise ValueError("batch, variate, and sequence dimensions must be non-zero")
        if patch_mask is not None and patch_mask.shape != (b, v, n):
            raise ValueError(
                f"patch_mask must have shape {(b, v, n)}, got {tuple(patch_mask.shape)}"
            )
        return b, v, n, d

    @staticmethod
    def _safe_padding_mask(
        padding_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Keep fully-padded rows finite while remembering to zero their update."""
        if padding_mask is None:
            return None, None
        padding_mask = padding_mask.to(dtype=torch.bool)
        fully_padded = padding_mask.all(dim=-1)
        if fully_padded.any():
            padding_mask = padding_mask.clone()
            padding_mask[fully_padded, 0] = False
        return padding_mask, fully_padded

    @staticmethod
    def _expand_batch_mask(
        mask: torch.Tensor | None,
        *,
        batch_size: int,
        repeats: int,
        axis_size: int,
        name: str,
    ) -> torch.Tensor | None:
        """Repeat a per-example square mask over variates or time steps."""
        if mask is None or mask.ndim == 2 or mask.shape[0] in (1, batch_size * repeats):
            return mask
        if mask.shape[-2:] != (axis_size, axis_size):
            raise ValueError(
                f"{name} trailing dimensions must be {(axis_size, axis_size)}, "
                f"got {tuple(mask.shape[-2:])}"
            )
        if mask.shape[0] != batch_size:
            raise ValueError(
                f"{name} batch dimension must be 1, {batch_size}, or "
                f"{batch_size * repeats}; got {mask.shape[0]}"
            )
        if mask.ndim == 3:
            return (
                mask[:, None]
                .expand(-1, repeats, -1, -1)
                .reshape(batch_size * repeats, axis_size, axis_size)
            )
        if mask.ndim == 4:
            heads = mask.shape[1]
            return (
                mask[:, None]
                .expand(-1, repeats, -1, -1, -1)
                .reshape(batch_size * repeats, heads, axis_size, axis_size)
            )
        raise ValueError(f"{name} must be a 2D, 3D, or 4D attention mask")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        *,
        sequence_mask: torch.Tensor | None = None,
        variate_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Mix embeddings along the sequence axis and then the variate axis.

        Args:
            input_embeddings: Tensor shaped ``[B, V, N, D]``.
            patch_mask: Optional blocked-key mask shaped ``[B, V, N]``.
                ``True`` marks a position whose keys should be masked.
            sequence_mask: Optional blocked attention mask broadcastable
                by the attention backend to ``[B*V, H, N, N]``.
            variate_mask: Optional blocked attention mask broadcastable to
                ``[B*N, H, V, V]``.
            is_causal: Apply a causal mask only to sequence attention.
            need_weights: Return per-head attention weights for both axes.

        Returns:
            ``(output, sequence_weights, variate_weights)``.  Weight entries
            are ``None`` unless ``need_weights=True``; variate weights are
            also ``None`` when variate attention is disabled.
        """
        b, v, n, d = self._validate_inputs(input_embeddings, patch_mask)
        if d != self.config.d_model:
            raise ValueError(
                f"embedding dimension {d} does not match d_model={self.config.d_model}"
            )
        if patch_mask is not None:
            patch_mask = patch_mask.to(device=input_embeddings.device, dtype=torch.bool)
        if sequence_mask is not None:
            sequence_mask = sequence_mask.to(
                device=input_embeddings.device, dtype=torch.bool
            )
        if variate_mask is not None:
            variate_mask = variate_mask.to(
                device=input_embeddings.device, dtype=torch.bool
            )

        # --- Sequence attention ---
        seq_input = self.pre_seq_attn_norm(input_embeddings).reshape(b * v, n, d)
        seq_padding = None if patch_mask is None else patch_mask.reshape(b * v, n)
        seq_padding, seq_fully_padded = self._safe_padding_mask(seq_padding)
        seq_mask = self._expand_batch_mask(
            sequence_mask, batch_size=b, repeats=v, axis_size=n, name="sequence_mask"
        )
        seq_update, seq_weights, _ = self.seq_attn(
            seq_input,
            seq_input,
            seq_input,
            seq_mask,
            seq_padding,
            is_causal=is_causal,
            need_weights=need_weights,
        )
        seq_update = self.post_seq_attn_norm(seq_update)
        if seq_fully_padded is not None:
            seq_update = seq_update.masked_fill(seq_fully_padded[:, None, None], 0.0)
            if seq_weights is not None:
                seq_weights = seq_weights.masked_fill(
                    seq_fully_padded[:, None, None, None], 0.0
                )
        seq_update = seq_update.reshape(b, v, n, d)
        hidden = input_embeddings + seq_update

        # --- Variate attention ---
        var_weights = None
        if self.var_attn is not None:
            assert self.pre_var_attn_norm is not None
            assert self.post_var_attn_norm is not None
            var_input = self.pre_var_attn_norm(hidden)
            var_input = var_input.permute(0, 2, 1, 3).reshape(b * n, v, d)
            var_padding = (
                None
                if patch_mask is None
                else patch_mask.permute(0, 2, 1).reshape(b * n, v)
            )
            var_padding, var_fully_padded = self._safe_padding_mask(var_padding)
            var_mask = self._expand_batch_mask(
                variate_mask, batch_size=b, repeats=n, axis_size=v, name="variate_mask"
            )
            var_update, var_weights, _ = self.var_attn(
                var_input,
                var_input,
                var_input,
                var_mask,
                var_padding,
                is_causal=False,
                need_weights=need_weights,
            )
            var_update = self.post_var_attn_norm(var_update)
            if var_fully_padded is not None:
                var_update = var_update.masked_fill(
                    var_fully_padded[:, None, None], 0.0
                )
                if var_weights is not None:
                    var_weights = var_weights.masked_fill(
                        var_fully_padded[:, None, None, None], 0.0
                    )
            var_update = var_update.reshape(b, n, v, d).permute(0, 2, 1, 3)
            hidden = hidden + var_update

        # --- FFN ---
        ff_update = self.ff0(self.pre_ff_norm(hidden))
        ff_update = self.ff_dropout(self.activation(ff_update))
        ff_update = self.ff1(ff_update)
        output = hidden + self.post_ff_norm(ff_update)
        return output, seq_weights, var_weights


class StackedMixingTransformer(nn.Module):
    """A stack of :class:`MixingTransformer` layers."""

    def __init__(
        self, config: TransformerConfig | None = None, **overrides: Any
    ) -> None:
        super().__init__()
        config = TransformerConfig.resolve(config, **overrides)
        self.config = config
        self.layers = nn.ModuleList(
            MixingTransformer(config.for_layer(index))
            for index in range(config.num_layers)
        )

    def forward(
        self,
        input_embeddings: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        *,
        sequence_mask: torch.Tensor | None = None,
        variate_mask: torch.Tensor | None = None,
        is_causal: bool = False,
        need_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        list[torch.Tensor | None],
        list[torch.Tensor | None],
    ]:
        output = input_embeddings
        sequence_weights: list[torch.Tensor | None] = []
        variate_weights: list[torch.Tensor | None] = []
        for layer in self.layers:
            output, seq_weights, var_weights = layer(
                output,
                patch_mask,
                sequence_mask=sequence_mask,
                variate_mask=variate_mask,
                is_causal=is_causal,
                need_weights=need_weights,
            )
            sequence_weights.append(seq_weights)
            variate_weights.append(var_weights)
        return output, sequence_weights, variate_weights


__all__ = ["MixingTransformer", "StackedMixingTransformer"]
