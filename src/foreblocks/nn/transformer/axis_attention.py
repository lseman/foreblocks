"""TimesFM3-style attention blocks for variate mixing.

This module provides a clean, minimal attention block that follows
TimesFM3's design: pre-norm, attention, post-norm + residual.

Each block operates on one axis of a ``[B, V, N, D]`` tensor:
- **Sequence axis** (N): attends over time, causal by default
- **Variate axis** (V): attends across variables, non-causal

The blocks are standalone ``nn.Module`` objects that can be composed
into a :class:`MixingTransformer` layer (sequence -> variate -> FFN).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum, auto

import torch
import torch.nn as nn

from foreblocks.nn.normalization import create_norm_layer
from foreblocks.nn.transformer.config import TransformerConfig
from foreblocks.nn.attention.enums import PositionEncoding
from foreblocks.nn.attention.multihead import MultiAttention


class Axis(Enum):
    """Which axis a :class:`MixingAttentionBlock` attends over."""

    SEQUENCE = auto()
    VARIATE = auto()


@dataclass(frozen=True)
class AttentionAxisConfig:
    """Configuration for one attention axis.

    Parameters:
        axis: Which axis to attend over.
        causal: Apply causal masking. Sequence attention is causal;
            variate attention is never causal.
        position_encoding: Which position encoding to apply. Variate
            attention typically uses :attr:`PositionEncoding.NONE`.
        enforce_standard_attn: If ``True``, always use dense MHA
            regardless of :attr:`TransformerConfig.attention.variant`.
            This is required for variate attention where temporal
            variants (linear, sparse, recurrent) have no meaningful
            semantics on an unordered set of variables.
    """

    axis: Axis
    causal: bool
    position_encoding: PositionEncoding = PositionEncoding.NONE
    enforce_standard_attn: bool = False


class MixingAttentionBlock(nn.Module):
    """One axis of a TimesFM3-style variate mixing layer.

    Architecture (pre-norm):

        input -- pre_norm -- MHA -- post_norm + residual -- output

    The block reshapes the input tensor from ``[B, V, N, D]`` so that
    the attended axis becomes the sequence dimension:

    - **Sequence** (axis=SEQUENCE): ``[B*V, N, D]``
    - **Variate** (axis=VARIATE): ``[B*N, V, D]``

    After attention the tensor is reshaped back to ``[B, V, N, D]``.

    Parameters are configured via :class:`AttentionAxisConfig`, which
    lets the caller control causal masking, position encoding, and
    whether temporal attention variants are forced to dense MHA.
    """

    def __init__(
        self,
        config: TransformerConfig,
        axis_config: AttentionAxisConfig,
        *,
        dropout: float | None = None,
    ) -> None:
        super().__init__()
        self.axis = axis_config.axis
        self._causal = axis_config.causal

        d_model = config.d_model
        self._norm = lambda: create_norm_layer(  # noqa: E731
            config.custom_norm, d_model, config.layer_norm_eps
        )

        # Pre- and post-attention norms
        self.pre_norm = self._norm()
        self.post_norm = self._norm()

        # Build the attention config for this axis
        attn_cfg = config.attention
        assert attn_cfg is not None

        if axis_config.enforce_standard_attn:
            attn_cfg = replace(
                attn_cfg,
                variant=replace(attn_cfg.variant, name="standard"),
            )

        if axis_config.position_encoding is not PositionEncoding.NONE:
            attn_cfg = replace(
                attn_cfg,
                position=replace(
                    attn_cfg.position,
                    encoding=axis_config.position_encoding,
                ),
            )

        dropout = config.dropout if dropout is None else dropout
        attn_cfg = replace(
            attn_cfg,
            shape=replace(attn_cfg.shape, dropout=dropout),
        )
        self.attention = MultiAttention(attn_cfg)

    def forward(
        self,
        input_tensor: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        is_causal: bool | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass.

        Args:
            input_tensor: Tensor shaped ``[B, V, N, D]``.
            patch_mask: Optional blocked-key mask shaped ``[B, V, N]``.
                ``True`` marks a position whose keys should be masked.
            is_causal: Override the configured causal setting. When
                ``None``, the config's value is used.
            need_weights: Return attention weights (may add memory).

        Returns:
            ``(output, attention_weights)``. Weights are ``None`` when
            ``need_weights=False``.
        """
        b, v, n, d = input_tensor.shape
        causal = self._causal if is_causal is None else is_causal

        # Reshape so the attended axis becomes the sequence dim
        if self.axis is Axis.SEQUENCE:
            flat = self.pre_norm(input_tensor).reshape(b * v, n, d)
            update, weights = self.attention(
                flat, flat, flat, is_causal=causal, need_weights=need_weights
            )
            update = update.reshape(b, v, n, d)
        else:
            # VARIATE axis: permute (b,v,n,d) -> (b,n,v,d) -> (b*n, v, d)
            flat = self.pre_norm(input_tensor).permute(0, 2, 1, 3).reshape(b * n, v, d)
            update, weights = self.attention(
                flat, flat, flat, is_causal=False, need_weights=need_weights
            )
            update = update.reshape(b, n, v, d).permute(0, 2, 1, 3)

        # Post-norm + residual
        output = input_tensor + self.post_norm(update)
        return output, weights


def make_sequence_block(
    config: TransformerConfig, *, dropout: float | None = None
) -> MixingAttentionBlock:
    """Create a sequence-axis attention block.

    Sequence attention is causal and uses the configured position encoding.
    """
    return MixingAttentionBlock(
        config,
        AttentionAxisConfig(
            axis=Axis.SEQUENCE,
            causal=True,
            enforce_standard_attn=False,
        ),
        dropout=dropout,
    )


def make_variate_block(
    config: TransformerConfig,
    *,
    variate_position_encoding: bool = False,
    dropout: float | None = None,
) -> MixingAttentionBlock:
    """Create a variate-axis attention block.

    Variate attention is **never** causal and never uses temporal
    attention variants. Position encoding is only applied when
    ``variate_position_encoding=True``.
    """
    pos_enc = (
        PositionEncoding.LEARNED
        if variate_position_encoding
        else PositionEncoding.NONE
    )
    return MixingAttentionBlock(
        config,
        AttentionAxisConfig(
            axis=Axis.VARIATE,
            causal=False,
            position_encoding=pos_enc,
            enforce_standard_attn=True,
        ),
        dropout=dropout,
    )


__all__ = [
    "Axis",
    "AttentionAxisConfig",
    "MixingAttentionBlock",
    "make_sequence_block",
    "make_variate_block",
]
