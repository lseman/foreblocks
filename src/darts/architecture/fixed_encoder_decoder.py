"""Searchable mixed blocks and fixed deployment wrappers."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bridges import LearnedPoolingBridge
from .bb_sequence import (
    ArchitectureNormalizer,
    BaseFixedSequenceBlock,
    SearchableDecomposition,
    SequenceStateAdapter,
)
from .bb_transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
)
from .freeze_utils import (
    _freeze_transformer_cross_attention,
    _freeze_transformer_cross_attention_position,
    _freeze_transformer_decoder_style,
    _freeze_transformer_ffn_mode,
    _freeze_transformer_patch_mode,
    _freeze_transformer_self_attention,
    _freeze_transformer_self_attention_position,
)


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


class FixedEncoder(BaseFixedSequenceBlock):
    """Simple fixed encoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        patching_mode: str | None = None,
    ):
        self.self_attention_type = (
            str(self_attention_type).lower()
            if self_attention_type is not None
            else None
        )
        self.self_attention_position_mode = (
            str(self_attention_position_mode).lower()
            if self_attention_position_mode is not None
            else None
        )
        self.patching_mode = (
            str(patching_mode).lower() if patching_mode is not None else None
        )
        self.ffn_mode = str(ffn_mode).lower() if ffn_mode is not None else None
        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=(
                lambda input_dim, latent_dim, num_layers, dropout: (
                    LightweightTransformerEncoder(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        self_attention_type=self.self_attention_type or "sdp",
                        self_attention_position_mode=(
                            self.self_attention_position_mode or "rope"
                        ),
                        ffn_variant=self.ffn_mode or "swiglu",
                        patching_mode=self.patching_mode or "direct",
                        enable_patch_search=False,
                    )
                )
            ),
        )
        if self.self_attention_type is not None and hasattr(self, "rnn"):
            _freeze_transformer_self_attention(self.rnn, self.self_attention_type)
        if self.self_attention_position_mode is not None and hasattr(self, "rnn"):
            _freeze_transformer_self_attention_position(
                self.rnn, self.self_attention_position_mode
            )
        if self.patching_mode is not None and hasattr(self, "rnn"):
            _freeze_transformer_patch_mode(self.rnn, self.patching_mode)
        if self.ffn_mode is not None and hasattr(self, "rnn"):
            _freeze_transformer_ffn_mode(self.rnn, self.ffn_mode)
        self.normalizer = None
        self.context_proj = None
        self.searchable_decomp = None

    def forward(self, x: torch.Tensor) -> tuple:
        if self.searchable_decomp is not None:
            x = self.searchable_decomp(x, temperature=0.01)

        if self.rnn_type == "transformer":
            output, context, state = self.rnn(
                x,
                temperature=getattr(self.rnn, "temperature", 1.0),
                variant_gdas=False,
            )
        else:
            raw_output, state = self.rnn(x)
            context = raw_output[:, -1:, :]
            output = raw_output

            if isinstance(self.rnn, nn.GRU):
                h = state
                c = torch.zeros_like(h)
                state = (h, c)

        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            state = self.normalizer.normalize_state(state, self.rnn_type)
        if self.context_proj is not None:
            context = self.context_proj(context)

        return output, context, state


class FixedDecoder(BaseFixedSequenceBlock):
    """Simple fixed decoder wrapper for deployment"""

    def __init__(
        self,
        rnn=None,
        rnn_type: str = None,
        input_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        cross_attention_type: str | None = None,
        cross_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        decode_style: str | None = None,
    ):
        self.self_attention_type = (
            str(self_attention_type).lower()
            if self_attention_type is not None
            else None
        )
        self.self_attention_position_mode = (
            str(self_attention_position_mode).lower()
            if self_attention_position_mode is not None
            else "rope"
        )
        self.cross_attention_type = (
            str(cross_attention_type).lower()
            if cross_attention_type is not None
            else "sdp"
        )
        self.cross_attention_position_mode = (
            str(cross_attention_position_mode).lower()
            if cross_attention_position_mode is not None
            else "rope"
        )
        self.decode_style = (
            str(decode_style).lower() if decode_style is not None else "autoregressive"
        )
        self.ffn_mode = str(ffn_mode).lower() if ffn_mode is not None else "swiglu"
        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=(
                lambda input_dim, latent_dim, num_layers, dropout: (
                    LightweightTransformerDecoder(
                        input_dim=input_dim,
                        latent_dim=latent_dim,
                        num_layers=num_layers,
                        dropout=dropout,
                        self_attention_type=self.self_attention_type or "sdp",
                        self_attention_position_mode=self.self_attention_position_mode,
                        cross_attention_type=self.cross_attention_type,
                        cross_attention_position_mode=self.cross_attention_position_mode,
                        ffn_variant=self.ffn_mode,
                    )
                )
            ),
        )
        if self.self_attention_type is not None and hasattr(self, "rnn"):
            _freeze_transformer_self_attention(self.rnn, self.self_attention_type)
        if hasattr(self, "rnn"):
            _freeze_transformer_self_attention_position(
                self.rnn, self.self_attention_position_mode
            )
        if hasattr(self, "rnn"):
            _freeze_transformer_cross_attention(self.rnn, self.cross_attention_type)
            _freeze_transformer_cross_attention_position(
                self.rnn, self.cross_attention_position_mode
            )
            _freeze_transformer_ffn_mode(self.rnn, self.ffn_mode)
        self.normalizer = None
        self.searchable_decomp = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor = None,
    ) -> tuple:
        if self.searchable_decomp is not None:
            tgt = self.searchable_decomp(tgt, temperature=0.01)

        batch_size = tgt.size(0)
        num_layers = getattr(self.rnn, "num_layers", 1)
        hidden_size = getattr(self.rnn, "hidden_size", self.latent_dim)

        # Deferred import to avoid circular dependency with converter.py
        from .converter import ArchitectureConverter as _AC

        hidden_state = _AC.ensure_proper_state_format(
            hidden_state, self.rnn_type, num_layers, batch_size, hidden_size, tgt.device
        )

        if self.rnn_type == "transformer":
            raw_output, new_state = self.rnn(tgt, memory, hidden_state)
        else:
            raw_output, new_state = self.rnn(tgt, hidden_state)

        output = raw_output
        if self.normalizer is not None:
            output = self.normalizer.normalize_output(output, self.rnn_type)
            normalized_state = self.normalizer.normalize_state(new_state, self.rnn_type)
            if normalized_state is not None:
                new_state = normalized_state

        return output, new_state
