"""Searchable mixed blocks and fixed deployment wrappers."""

from __future__ import annotations

import copy
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_attention import LearnedPoolingBridge
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


__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


class MixedEncoder(nn.Module):
    """Transformer-family encoder search block.

    Sequence architecture search is intentionally limited to one transformer
    branch. Patching is searched inside the encoder itself via patch-mode
    alphas rather than as a separate top-level architecture option.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        variant_gdas: bool = True,
        arch_path_keep_prob: float = 0.85,
        single_path_search: bool | None = None,
        include_patch: bool = True,
        transformer_self_attention_type: str = "auto",
        transformer_use_moe: bool = False,
        transformer_ffn_variant: str = "auto",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if single_path_search is not None:
            variant_gdas = bool(single_path_search)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = max(float(temperature), 1e-3)
        self.num_layers = 2
        self.variant_gdas = bool(variant_gdas)
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self.include_patch = bool(include_patch)
        self.rnn_type = "transformer"
        self.register_buffer("alphas", torch.zeros(3))
        self.register_buffer("layer_alpha_offsets", torch.zeros(self.num_layers, 3))
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)

        resolved_self_attention_type = str(transformer_self_attention_type).lower()
        self.transformer_use_moe = bool(transformer_use_moe)
        self.transformer_ffn_variant = str(transformer_ffn_variant).lower()

        self.transformer = LightweightTransformerEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=2,
            dropout=dropout,
            self_attention_type=resolved_self_attention_type,
            self_attention_position_mode="auto",
            use_moe=self.transformer_use_moe,
            ffn_variant=self.transformer_ffn_variant,
            use_checkpoint=use_checkpoint,
            temperature=self.temperature,
            variant_gdas=self.variant_gdas,
            enable_patch_search=self.include_patch,
            patching_mode="auto" if self.include_patch else "direct",
        )
        self.encoders = nn.ModuleList([self.transformer])
        self.encoder_names = ["transformer"]
        self.encoder_name_to_idx = {
            name: idx for idx, name in enumerate(self.encoder_names)
        }

        self.last_selected_output_idx: int | None = None
        self.last_selected_layer_idx: torch.Tensor | None = None
        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.context_proj = nn.Linear(latent_dim, latent_dim)
        self.rnn = self.transformer

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x = self.searchable_decomp(x, temperature=self.temperature)
        variant_gdas_active = self.training and self.variant_gdas
        self.last_selected_output_idx = 0
        self.last_selected_layer_idx = torch.zeros(
            self.num_layers, device=x.device, dtype=torch.long
        )
        trans_out, trans_ctx, trans_state = self.transformer(
            x,
            temperature=self.temperature,
            variant_gdas=variant_gdas_active,
        )
        output = self.normalizer.normalize_output(trans_out, "transformer")
        context = trans_ctx
        state = self.normalizer.normalize_state(trans_state, "transformer")

        context = self.context_proj(context)

        h_blended = state[0]
        c_blended = state[1]

        return output, context, (h_blended, c_blended)

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)
        if hasattr(self.transformer, "set_temperature"):
            self.transformer.set_temperature(self.temperature)

    def orthogonal_regularization(self) -> torch.Tensor:
        ref = next(self.parameters(), None)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())


class MixedDecoder(nn.Module):
    """Transformer-only decoder search block."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        use_attention_bridge: bool = True,
        attention_layers: int = 1,
        use_learned_memory_pooling: bool = True,
        memory_num_queries: int = 8,
        variant_gdas: bool = True,
        arch_path_keep_prob: float = 0.85,
        single_path_search: bool | None = None,
        attention_temperature_mult: float = 0.7,
        min_attention_temperature: float = 0.25,
        memory_query_options: list[int] | None = None,
        transformer_self_attention_type: str = "auto",
        transformer_cross_attention_type: str = "auto",
        transformer_cross_attention_modes: Sequence[str] | None = None,
        transformer_use_moe: bool = False,
        transformer_ffn_variant: str = "auto",
        use_checkpoint: bool = False,
    ):
        super().__init__()
        if single_path_search is not None:
            variant_gdas = bool(single_path_search)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.dropout = dropout
        self.temperature = max(float(temperature), 1e-3)
        self.num_layers = 2
        self.variant_gdas = bool(variant_gdas)
        self.arch_path_keep_prob = float(min(max(arch_path_keep_prob, 0.0), 1.0))
        self.rnn_type = "transformer"
        self.decode_style_names = ("autoregressive", "informer", "autoformer")
        self.decode_style = "auto"
        self.register_parameter(
            "decode_style_alphas", nn.Parameter(0.01 * torch.randn(3))
        )

        self.use_attention_bridge = False
        self.attention_layers = attention_layers
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.attention_temperature_mult = float(max(attention_temperature_mult, 1e-3))
        self.min_attention_temperature = float(max(min_attention_temperature, 1e-3))
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)
        self.default_memory_num_queries = int(max(1, memory_num_queries))

        resolved_self_attention_type = str(transformer_self_attention_type).lower()
        resolved_cross_attention_type = str(transformer_cross_attention_type).lower()
        self.transformer_use_moe = bool(transformer_use_moe)
        self.transformer_ffn_variant = str(transformer_ffn_variant).lower()

        self.transformer = LightweightTransformerDecoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=2,
            dropout=dropout,
            self_attention_type=resolved_self_attention_type,
            self_attention_position_mode="auto",
            cross_attention_type=resolved_cross_attention_type,
            cross_attention_position_mode="auto",
            cross_attention_modes=transformer_cross_attention_modes,
            use_moe=self.transformer_use_moe,
            ffn_variant=self.transformer_ffn_variant,
            use_checkpoint=use_checkpoint,
            temperature=self.temperature,
            variant_gdas=self.variant_gdas,
        )

        self.decoders = nn.ModuleList([self.transformer])
        self.decoder_names = ["transformer"]
        self.rnn_names = ["transformer"]
        self.decoder_name_to_idx = {
            name: idx for idx, name in enumerate(self.decoder_names)
        }
        self.rnn = self.transformer

        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.memory_query_options = (
            list(memory_query_options)
            if memory_query_options is not None
            else [4, 8, 16]
        )
        self.memory_query_options = [
            int(max(1, q)) for q in self.memory_query_options if int(max(1, q)) > 0
        ]
        if not self.memory_query_options:
            self.memory_query_options = [4, 8, 16]
        if self.default_memory_num_queries not in self.memory_query_options:
            self.memory_query_options = sorted(
                set(self.memory_query_options + [self.default_memory_num_queries])
            )
        self.default_memory_query_idx = self.memory_query_options.index(
            self.default_memory_num_queries
        )

        if use_learned_memory_pooling:
            self.memory_pool_bridges = nn.ModuleList([
                LearnedPoolingBridge(
                    dim=latent_dim,
                    num_queries=num_queries,
                    num_heads=4,
                    dropout=dropout,
                )
                for num_queries in self.memory_query_options
            ])
            memory_query_init = 0.01 * torch.randn(len(self.memory_query_options))
            self.register_parameter(
                "memory_query_alphas", nn.Parameter(memory_query_init)
            )
        else:
            self.memory_pool_bridges = None

    def get_decode_style_weights(self) -> torch.Tensor:
        tau = max(float(self.temperature), 1e-3)
        if self.training:
            if self.variant_gdas:
                return F.gumbel_softmax(
                    self.decode_style_alphas, tau=tau, hard=True, dim=0
                )
            return F.gumbel_softmax(
                self.decode_style_alphas, tau=tau, hard=False, dim=0
            )
        probs = F.softmax(self.decode_style_alphas / tau, dim=0)
        if self.variant_gdas:
            hard = torch.zeros_like(probs)
            hard[int(torch.argmax(probs).item())] = 1.0
            return hard
        return probs

    def get_decode_style_probs(self) -> torch.Tensor:
        return F.softmax(self.decode_style_alphas.detach(), dim=0)

    def resolve_decode_style(self) -> str:
        probs = self.get_decode_style_probs()
        idx = int(torch.argmax(probs).item())
        return self.decode_style_names[idx]

    def freeze_decode_style(self, decode_style: str) -> None:
        resolved = (
            "informer" if str(decode_style).lower() == "informer" else "autoregressive"
        )
        self.decode_style = resolved
        if hasattr(self, "decode_style_alphas"):
            self._parameters.pop("decode_style_alphas", None)
            try:
                delattr(self, "decode_style_alphas")
            except AttributeError:
                pass

    def _get_decoder_weights(self) -> torch.Tensor:
        ref = next(self.parameters())
        return ref.new_ones(1)

    def _get_memory_query_weights(self) -> torch.Tensor | None:
        if not (
            self.use_learned_memory_pooling
            and hasattr(self, "memory_query_alphas")
            and self.memory_pool_bridges is not None
        ):
            return None

        tau = max(float(self.temperature), 1e-3)
        if self.training:
            if self.variant_gdas:
                return F.gumbel_softmax(
                    self.memory_query_alphas, tau=tau, hard=True, dim=0
                )
            return F.gumbel_softmax(
                self.memory_query_alphas, tau=tau, hard=False, dim=0
            )
        probs = F.softmax(self.memory_query_alphas / tau, dim=0)
        if self.variant_gdas:
            hard = torch.zeros_like(probs)
            hard[int(torch.argmax(probs).item())] = 1.0
            return hard
        return probs

    def _build_shared_memory(
        self,
        memory: torch.Tensor | None,
        encoder_output: torch.Tensor | None,
        encoder_context: torch.Tensor | None,
    ) -> torch.Tensor | None:
        source = encoder_output
        if source is None:
            source = memory
        if source is None:
            source = encoder_context
        if source is None:
            return None

        if isinstance(source, tuple):
            source = source[0]

        if source.dim() == 2:
            source = source.unsqueeze(1)

        if self.memory_pool_bridges is not None:
            memory_weights = self._get_memory_query_weights()
            if memory_weights is None:
                pooled = self.memory_pool_bridges[self.default_memory_query_idx](source)
                return self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )

            if self.training and self.variant_gdas:
                chosen = int(torch.argmax(memory_weights.detach()).item())
                pooled = self.memory_pool_bridges[chosen](source)
                pooled = self._resize_memory_queries(
                    pooled, self.default_memory_num_queries
                )
                return memory_weights[chosen] * pooled

            pooled_outputs = [
                self._resize_memory_queries(
                    bridge(source), self.default_memory_num_queries
                )
                for bridge in self.memory_pool_bridges
            ]
            return sum(w * out for w, out in zip(memory_weights, pooled_outputs))

        return source

    @staticmethod
    def _resize_memory_queries(
        memory: torch.Tensor, target_queries: int
    ) -> torch.Tensor:
        if memory.dim() != 3:
            return memory
        q = int(memory.size(1))
        target = int(max(1, target_queries))
        if q == target:
            return memory
        return F.interpolate(
            memory.transpose(1, 2),
            size=target,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        hidden_state=None,
        encoder_output: torch.Tensor | None = None,
        encoder_context: torch.Tensor | None = None,
        forced_output_idx: int | None = None,
        forced_layer_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        tgt = self.searchable_decomp(tgt, temperature=self.temperature)
        batch_size = tgt.size(0)

        trans_state = SequenceStateAdapter.ensure_rnn_state(
            hidden_state,
            rnn_type="transformer",
            num_layers=self.num_layers,
            batch_size=batch_size,
            hidden_size=self.latent_dim,
            device=tgt.device,
            dtype=tgt.dtype,
        )

        shared_memory = self._build_shared_memory(
            memory, encoder_output, encoder_context
        )
        transformer_memory = (
            shared_memory
            if shared_memory is not None
            else (
                encoder_context
                if encoder_context is not None
                else torch.zeros_like(tgt[:, :1, :])
            )
        )
        trans_out, trans_new_state = self.transformer(
            tgt, transformer_memory, trans_state
        )
        output = self.normalizer.normalize_output(trans_out, "transformer")
        state = self.normalizer.normalize_state(trans_new_state, "transformer")

        return output, state

    def set_temperature(self, temp: float):
        self.temperature = max(float(temp), 1e-3)
        if hasattr(self.transformer, "set_temperature"):
            self.transformer.set_temperature(self.temperature)

    def orthogonal_regularization(self) -> torch.Tensor:
        ref = next(self.parameters(), None)
        if ref is None:
            return torch.tensor(0.0)
        return ref.new_zeros(())


