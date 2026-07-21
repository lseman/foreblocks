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
from .fixed_encoder_decoder import FixedDecoder, FixedEncoder
from .freeze_utils import (
    _freeze_transformer_cross_attention,
    _freeze_transformer_cross_attention_position,
    _freeze_transformer_decoder_style,
    _freeze_transformer_ffn_mode,
    _freeze_transformer_patch_mode,
    _freeze_transformer_self_attention,
    _freeze_transformer_self_attention_position,
    _resolve_searchable_cross_attention_position,
    _resolve_searchable_cross_attention_type,
    _resolve_searchable_decoder_style,
    _resolve_searchable_ffn_mode,
    _resolve_searchable_patch_mode,
    _resolve_searchable_self_attention_position,
    _resolve_searchable_self_attention_type,
)


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


class ArchitectureConverter:
    """Utility class for converting between mixed and fixed architectures"""

    @staticmethod
    def get_best_architecture(
        alphas: torch.Tensor,
        num_options: int = 4,
        arch_names: list[str] | None = None,
    ) -> str:
        arch_names = list(arch_names or ["transformer"])
        num_options = min(max(1, int(num_options)), len(arch_names))
        best_idx = int(torch.argmax(alphas[:num_options]).item())
        return arch_names[best_idx]

    @staticmethod
    def ensure_proper_state_format(
        state,
        rnn_type: str,
        num_layers: int,
        batch_size: int,
        hidden_size: int,
        device: torch.device,
    ):
        return SequenceStateAdapter.ensure_rnn_state(
            state,
            rnn_type=rnn_type,
            num_layers=num_layers,
            batch_size=batch_size,
            hidden_size=hidden_size,
            device=device,
        )

    @staticmethod
    def fix_mixed_weights(mixed_model, temperature: float = 0.01):
        if hasattr(mixed_model, "set_temperature"):
            mixed_model.set_temperature(temperature)

    @staticmethod
    def create_fixed_encoder(mixed_encoder, **kwargs) -> FixedEncoder:
        best_type = "transformer"
        self_attention_type = None
        self_attention_position_mode = None
        ffn_mode = None
        preserve_soft_choices = not bool(getattr(mixed_encoder, "variant_gdas", True))
        if best_type == "transformer" and not preserve_soft_choices:
            self_attention_type = _resolve_searchable_self_attention_type(
                mixed_encoder.transformer
            )
            self_attention_position_mode = _resolve_searchable_self_attention_position(
                mixed_encoder.transformer
            )
            ffn_mode = _resolve_searchable_ffn_mode(mixed_encoder.transformer)
        patch_mode = (
            None
            if preserve_soft_choices
            else _resolve_searchable_patch_mode(mixed_encoder.transformer)
        )
        fixed_encoder = FixedEncoder(
            rnn=copy.deepcopy(mixed_encoder.transformer),
            input_dim=mixed_encoder.input_dim,
            latent_dim=mixed_encoder.latent_dim,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            patching_mode=patch_mode,
            **kwargs,
        )
        ArchitectureConverter._transfer_encoder_weights(
            mixed_encoder,
            fixed_encoder,
            best_type,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            ffn_mode=ffn_mode,
            patch_mode=patch_mode,
        )
        return fixed_encoder

    @staticmethod
    def create_fixed_decoder(mixed_decoder, **kwargs) -> FixedDecoder:
        explicit_cross_attention_type = kwargs.pop("cross_attention_type", None)
        best_type = "transformer"
        self_attention_type = None
        self_attention_position_mode = None
        decode_style = _resolve_searchable_decoder_style(mixed_decoder)
        cross_attention_type = _resolve_searchable_cross_attention_type(
            mixed_decoder.transformer
        )
        cross_attention_position_mode = _resolve_searchable_cross_attention_position(
            mixed_decoder.transformer
        )
        ffn_mode = _resolve_searchable_ffn_mode(mixed_decoder.transformer)
        if explicit_cross_attention_type is not None:
            cross_attention_type = str(explicit_cross_attention_type).lower()
        if best_type == "transformer":
            self_attention_type = _resolve_searchable_self_attention_type(
                mixed_decoder.transformer
            )
            self_attention_position_mode = _resolve_searchable_self_attention_position(
                mixed_decoder.transformer
            )

        fixed_decoder = FixedDecoder(
            rnn_type=best_type,
            input_dim=mixed_decoder.input_dim,
            latent_dim=mixed_decoder.latent_dim,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            cross_attention_type=cross_attention_type,
            cross_attention_position_mode=cross_attention_position_mode,
            ffn_mode=ffn_mode,
            decode_style=decode_style,
            **kwargs,
        )
        ArchitectureConverter._transfer_decoder_weights(
            mixed_decoder,
            fixed_decoder,
            best_type,
            self_attention_type=self_attention_type,
            self_attention_position_mode=self_attention_position_mode,
            cross_attention_type=cross_attention_type,
            cross_attention_position_mode=cross_attention_position_mode,
            ffn_mode=ffn_mode,
            decode_style=decode_style,
        )
        return fixed_decoder

    @staticmethod
    def _transfer_encoder_weights(
        mixed_encoder,
        fixed_encoder,
        arch_type: str,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        patch_mode: str | None = None,
    ):
        try:
            if arch_type == "transformer":
                source_rnn = mixed_encoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_encoder.rnn.load_state_dict(source_rnn.state_dict(), strict=False)
            if (
                self_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_encoder, "rnn")
            ):
                _freeze_transformer_self_attention(
                    fixed_encoder.rnn, self_attention_type
                )
            if (
                self_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_encoder, "rnn")
            ):
                _freeze_transformer_self_attention_position(
                    fixed_encoder.rnn, self_attention_position_mode
                )
            if patch_mode is not None and arch_type == "transformer":
                _freeze_transformer_patch_mode(fixed_encoder.rnn, patch_mode)
            if ffn_mode is not None and arch_type == "transformer":
                _freeze_transformer_ffn_mode(fixed_encoder.rnn, ffn_mode)

            if hasattr(mixed_encoder, "normalizer"):
                fixed_encoder.normalizer = ArchitectureNormalizer(
                    mixed_encoder.latent_dim
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.normalizer.load_state_dict(
                    mixed_encoder.normalizer.state_dict()
                )
            if hasattr(mixed_encoder, "context_proj"):
                fixed_encoder.context_proj = nn.Linear(
                    mixed_encoder.latent_dim,
                    mixed_encoder.latent_dim,
                ).to(next(fixed_encoder.parameters()).device)
                fixed_encoder.context_proj.load_state_dict(
                    mixed_encoder.context_proj.state_dict()
                )
            if hasattr(mixed_encoder, "searchable_decomp"):
                fixed_encoder.searchable_decomp = copy.deepcopy(
                    mixed_encoder.searchable_decomp
                ).to(next(fixed_encoder.parameters()).device)
                if hasattr(fixed_encoder.searchable_decomp, "alpha_logits"):
                    fixed_encoder.searchable_decomp.alpha_logits.requires_grad_(False)
        except Exception as e:
            print(f"Warning: Could not transfer encoder weights: {e}")

    @staticmethod
    def _transfer_decoder_weights(
        mixed_decoder,
        fixed_decoder,
        arch_type: str,
        self_attention_type: str | None = None,
        self_attention_position_mode: str | None = None,
        cross_attention_type: str | None = None,
        cross_attention_position_mode: str | None = None,
        ffn_mode: str | None = None,
        decode_style: str | None = None,
    ):
        try:
            if arch_type == "transformer":
                source_rnn = mixed_decoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_decoder.rnn.load_state_dict(source_rnn.state_dict(), strict=False)
            if (
                self_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_self_attention(
                    fixed_decoder.rnn, self_attention_type
                )
            if (
                self_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_self_attention_position(
                    fixed_decoder.rnn, self_attention_position_mode
                )
            if (
                cross_attention_type is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_cross_attention(
                    fixed_decoder.rnn, cross_attention_type
                )
            if (
                cross_attention_position_mode is not None
                and arch_type == "transformer"
                and hasattr(fixed_decoder, "rnn")
            ):
                _freeze_transformer_cross_attention_position(
                    fixed_decoder.rnn, cross_attention_position_mode
                )
            if decode_style is not None and hasattr(fixed_decoder, "decode_style"):
                fixed_decoder.decode_style = str(decode_style).lower()
            if decode_style is not None and hasattr(fixed_decoder, "rnn"):
                _freeze_transformer_decoder_style(fixed_decoder, decode_style)
            if ffn_mode is not None and arch_type == "transformer":
                _freeze_transformer_ffn_mode(fixed_decoder.rnn, ffn_mode)

            if hasattr(mixed_decoder, "normalizer"):
                fixed_decoder.normalizer = ArchitectureNormalizer(
                    mixed_decoder.latent_dim
                ).to(next(fixed_decoder.parameters()).device)
                fixed_decoder.normalizer.load_state_dict(
                    mixed_decoder.normalizer.state_dict()
                )
            if hasattr(mixed_decoder, "searchable_decomp"):
                fixed_decoder.searchable_decomp = copy.deepcopy(
                    mixed_decoder.searchable_decomp
                ).to(next(fixed_decoder.parameters()).device)
                if hasattr(fixed_decoder.searchable_decomp, "alpha_logits"):
                    fixed_decoder.searchable_decomp.alpha_logits.requires_grad_(False)

        except Exception as e:
            print(f"Warning: Could not transfer decoder weights: {e}")
