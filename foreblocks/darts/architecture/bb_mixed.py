"""Searchable mixed blocks and fixed deployment wrappers."""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_attention import AttentionBridge, LearnedPoolingBridge
from .bb_sequence import (
    ArchitectureNormalizer,
    BaseFixedSequenceBlock,
    BaseMixedSequenceBlock,
    SearchableDecomposition,
    SequenceStateAdapter,
)
from .bb_transformers import (
    LightweightTransformerDecoder,
    LightweightTransformerEncoder,
    PatchTSTEncoder,
)

__all__ = [
    "MixedEncoder",
    "MixedDecoder",
    "ArchitectureConverter",
    "FixedEncoder",
    "FixedDecoder",
]


class MixedEncoder(BaseMixedSequenceBlock):
    """Improved mixed encoder with better compatibility"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_len: int,
        dropout: float = 0.1,
        temperature: float = 1.0,
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            temperature=temperature,
            num_layers=2,
            num_options=4,
            single_path_search=single_path_search,
            arch_path_keep_prob=arch_path_keep_prob,
        )
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)

        self.transformer = LightweightTransformerEncoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )
        self.patch_encoder = PatchTSTEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            num_layers=2,
            dropout=dropout,
        )
        self.encoders = nn.ModuleList(
            [self.lstm, self.gru, self.transformer, self.patch_encoder]
        )
        self.encoder_names = ["lstm", "gru", "transformer", "patch"]
        self.normalizer = ArchitectureNormalizer(latent_dim)
        self.context_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.searchable_decomp(x, temperature=self.temperature)
        layer_weights = self._get_arch_weights()
        output_weights = self._get_output_arch_weights()

        single_path_active = self.training and self.single_path_search
        selected_output_idx = int(torch.argmax(output_weights.detach()).item())
        selected_layer_idx = torch.argmax(layer_weights.detach(), dim=-1)

        if single_path_active:
            required_arches = {selected_output_idx}
            required_arches.update(int(idx.item()) for idx in selected_layer_idx)
        else:
            required_arches = {0, 1, 2, 3}

        output_by_arch: dict = {}
        context_by_arch: dict = {}
        state_by_arch: dict = {}

        if 0 in required_arches:
            lstm_out, lstm_state = self.lstm(x)
            output_by_arch[0] = self.normalizer.normalize_output(lstm_out, "lstm")
            context_by_arch[0] = lstm_out[:, -1:, :]
            state_by_arch[0] = self.normalizer.normalize_state(lstm_state, "lstm")

        if 1 in required_arches:
            gru_out, gru_state = self.gru(x)
            output_by_arch[1] = self.normalizer.normalize_output(gru_out, "gru")
            context_by_arch[1] = gru_out[:, -1:, :]
            state_by_arch[1] = self.normalizer.normalize_state(gru_state, "gru")

        if 2 in required_arches:
            trans_out, trans_ctx, trans_state = self.transformer(x)
            output_by_arch[2] = self.normalizer.normalize_output(
                trans_out, "transformer"
            )
            context_by_arch[2] = trans_ctx
            state_by_arch[2] = self.normalizer.normalize_state(
                trans_state, "transformer"
            )

        if 3 in required_arches:
            patch_out, patch_ctx, patch_state = self.patch_encoder(x)
            output_by_arch[3] = self.normalizer.normalize_output(patch_out, "patch")
            context_by_arch[3] = patch_ctx
            state_by_arch[3] = self.normalizer.normalize_state(patch_state, "patch")

        if single_path_active:
            w = output_weights[selected_output_idx]
            output = w * output_by_arch[selected_output_idx]
            context = w * context_by_arch[selected_output_idx]
        else:
            n = len(self.encoders)
            output = sum(output_weights[i] * output_by_arch[i] for i in range(n))
            context = sum(output_weights[i] * context_by_arch[i] for i in range(n))

        context = self.context_proj(context)

        if single_path_active:
            h_blended = torch.zeros(
                self.num_layers,
                x.size(0),
                self.latent_dim,
                device=x.device,
                dtype=output.dtype,
            )
            c_blended = torch.zeros_like(h_blended)
            for layer_idx in range(self.num_layers):
                arch_idx = int(selected_layer_idx[layer_idx].item())
                lw = layer_weights[layer_idx, arch_idx]
                h_blended[layer_idx] = lw * state_by_arch[arch_idx][0][layer_idx]
                c_blended[layer_idx] = lw * state_by_arch[arch_idx][1][layer_idx]
        else:
            lw_exp = layer_weights.unsqueeze(-1).unsqueeze(-1)
            n = len(self.encoders)
            h_stack = torch.stack([state_by_arch[i][0] for i in range(n)], dim=1)
            c_stack = torch.stack([state_by_arch[i][1] for i in range(n)], dim=1)
            h_blended = (lw_exp * h_stack).sum(dim=1)
            c_blended = (lw_exp * c_stack).sum(dim=1)

        return output, context, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        base = super().get_alphas()
        if hasattr(self, "searchable_decomp"):
            return torch.cat([base, self.searchable_decomp.get_alphas()])
        return base


class MixedDecoder(BaseMixedSequenceBlock):
    """Improved mixed decoder with better architecture compatibility"""

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
        single_path_search: bool = True,
        arch_path_keep_prob: float = 0.85,
        attention_temperature_mult: float = 0.7,
        min_attention_temperature: float = 0.25,
        memory_query_options: Optional[List[int]] = None,
    ):
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            seq_len=seq_len,
            dropout=dropout,
            temperature=temperature,
            num_layers=2,
            num_options=3,
            single_path_search=single_path_search,
            arch_path_keep_prob=arch_path_keep_prob,
        )

        self.use_attention_bridge = use_attention_bridge
        self.attention_layers = attention_layers
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.attention_temperature_mult = float(max(attention_temperature_mult, 1e-3))
        self.min_attention_temperature = float(max(min_attention_temperature, 1e-3))
        self.searchable_decomp = SearchableDecomposition(c_in=input_dim)
        self.default_memory_num_queries = int(max(1, memory_num_queries))

        self.transformer = LightweightTransformerDecoder(
            input_dim=input_dim, latent_dim=latent_dim, num_layers=2, dropout=dropout
        )

        self.decoders = nn.ModuleList([self.lstm, self.gru, self.transformer])
        self.decoder_names = ["lstm", "gru", "transformer"]
        self.rnn_names = ["lstm", "gru", "transformer"]

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
            self.memory_pool_bridges = nn.ModuleList(
                [
                    LearnedPoolingBridge(
                        dim=latent_dim,
                        num_queries=num_queries,
                        num_heads=4,
                        dropout=dropout,
                    )
                    for num_queries in self.memory_query_options
                ]
            )
            memory_query_init = 0.01 * torch.randn(len(self.memory_query_options))
            self.register_parameter(
                "memory_query_alphas", nn.Parameter(memory_query_init)
            )
        else:
            self.memory_pool_bridges = None

        if use_attention_bridge:
            self.attention_bridge = AttentionBridge(
                latent_dim, num_heads=4, dropout=dropout, attn_type="auto"
            )

    def _get_decoder_weights(self) -> torch.Tensor:
        return self._get_output_arch_weights()

    def _get_memory_query_weights(self) -> Optional[torch.Tensor]:
        if not (
            self.use_learned_memory_pooling
            and hasattr(self, "memory_query_alphas")
            and self.memory_pool_bridges is not None
        ):
            return None

        tau = max(float(self.temperature), 1e-3)
        if self._should_use_stochastic_arch_sampling():
            if self.single_path_search:
                return self._sample_straight_through_gumbel(
                    self.memory_query_alphas, tau=tau, dim=0
                )
            return F.gumbel_softmax(
                self.memory_query_alphas, tau=tau, hard=False, dim=0
            )
        return F.softmax(self.memory_query_alphas / tau, dim=0)

    def _build_shared_memory(
        self,
        memory: Optional[torch.Tensor],
        encoder_output: Optional[torch.Tensor],
        encoder_context: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
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

            if self.training and self.single_path_search:
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
        encoder_output: Optional[torch.Tensor] = None,
        encoder_context: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        tgt = self.searchable_decomp(tgt, temperature=self.temperature)
        batch_size = tgt.size(0)
        single_path_active = self.training and self.single_path_search

        lstm_state, gru_state, trans_state = (
            SequenceStateAdapter.split_mixed_decoder_states(
                hidden_state,
                num_layers=self.num_layers,
                batch_size=batch_size,
                hidden_size=self.latent_dim,
                device=tgt.device,
                dtype=tgt.dtype,
            )
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
        attention_source = (
            shared_memory if shared_memory is not None else encoder_output
        )
        if attention_source is None:
            attention_source = encoder_context

        decoder_layer_weights = self._get_arch_weights()
        decoder_weights = self._get_decoder_weights()
        selected_output_idx = int(torch.argmax(decoder_weights.detach()).item())
        selected_layer_idx = torch.argmax(decoder_layer_weights.detach(), dim=-1)

        if single_path_active:
            required_arches = {selected_output_idx}
            required_arches.update(int(idx.item()) for idx in selected_layer_idx)
        else:
            required_arches = {0, 1, 2}

        output_by_arch: dict = {}
        state_by_arch: dict = {}

        if 0 in required_arches:
            lstm_out, lstm_new_state = self.lstm(tgt, lstm_state)
            output_by_arch[0] = self.normalizer.normalize_output(lstm_out, "lstm")
            state_by_arch[0] = self.normalizer.normalize_state(lstm_new_state, "lstm")

        if 1 in required_arches:
            gru_out, gru_new_state = self.gru(tgt, gru_state)
            output_by_arch[1] = self.normalizer.normalize_output(gru_out, "gru")
            state_by_arch[1] = self.normalizer.normalize_state(gru_new_state, "gru")

        if 2 in required_arches:
            trans_out, trans_new_state = self.transformer(
                tgt, transformer_memory, trans_state
            )
            output_by_arch[2] = self.normalizer.normalize_output(
                trans_out, "transformer"
            )
            state_by_arch[2] = self.normalizer.normalize_state(
                trans_new_state, "transformer"
            )

        can_attend = self.use_attention_bridge and attention_source is not None
        bridge_tau = max(
            float(self.temperature) * self.attention_temperature_mult,
            self.min_attention_temperature,
        )

        output_final_by_arch: dict = {}
        for arch_idx in required_arches:
            base_out = output_by_arch[arch_idx]
            if can_attend:
                output_final_by_arch[arch_idx] = self.attention_bridge(
                    base_out,
                    encoder_output=attention_source,
                    encoder_context=encoder_context,
                    temperature=bridge_tau,
                    single_path=single_path_active,
                )
            else:
                output_final_by_arch[arch_idx] = base_out

        if single_path_active:
            selected_output_weight = decoder_weights[selected_output_idx]
            output = selected_output_weight * output_final_by_arch[selected_output_idx]
        else:
            output = sum(decoder_weights[i] * output_final_by_arch[i] for i in range(3))

        if single_path_active:
            h_blended = torch.zeros(
                self.num_layers,
                batch_size,
                self.latent_dim,
                device=tgt.device,
                dtype=output.dtype,
            )
            c_blended = torch.zeros_like(h_blended)
            for layer_idx in range(self.num_layers):
                arch_idx = int(selected_layer_idx[layer_idx].item())
                lw = decoder_layer_weights[layer_idx, arch_idx]
                h_blended[layer_idx] = lw * state_by_arch[arch_idx][0][layer_idx]
                c_blended[layer_idx] = lw * state_by_arch[arch_idx][1][layer_idx]
        else:
            lw_exp = decoder_layer_weights.unsqueeze(-1).unsqueeze(-1)
            h_stack = torch.stack(
                [state_by_arch[0][0], state_by_arch[1][0], state_by_arch[2][0]], dim=1
            )
            c_stack = torch.stack(
                [state_by_arch[0][1], state_by_arch[1][1], state_by_arch[2][1]], dim=1
            )
            h_blended = (lw_exp * h_stack).sum(dim=1)
            c_blended = (lw_exp * c_stack).sum(dim=1)

        return output, (h_blended, c_blended)

    def get_alphas(self) -> torch.Tensor:
        parts = [super().get_alphas()]

        if hasattr(self, "searchable_decomp"):
            parts.append(self.searchable_decomp.get_alphas())

        if self.use_learned_memory_pooling and hasattr(self, "memory_query_alphas"):
            parts.append(F.softmax(self.memory_query_alphas, dim=0))

        if (
            self.use_attention_bridge
            and hasattr(self, "attention_bridge")
            and hasattr(self.attention_bridge, "attn_alphas")
        ):
            parts.append(F.softmax(self.attention_bridge.attn_alphas, dim=0))

        return torch.cat(parts)


class ArchitectureConverter:
    """Utility class for converting between mixed and fixed architectures"""

    @staticmethod
    def get_best_architecture(alphas: torch.Tensor) -> str:
        arch_names = ["lstm", "gru", "transformer", "patch"]
        best_idx = int(torch.argmax(alphas[:4]).item())
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
        with torch.no_grad():
            alphas = mixed_model.get_alphas()
            best_idx = torch.argmax(alphas[:4])

            new_alphas = torch.full_like(mixed_model.alphas, -10.0)
            new_alphas[best_idx] = 10.0
            mixed_model.alphas.copy_(new_alphas)
            if hasattr(mixed_model, "layer_alpha_offsets"):
                mixed_model.layer_alpha_offsets.zero_()

            mixed_model.set_temperature(temperature)

            if hasattr(mixed_model, "attention_alphas"):
                attention_best = torch.argmax(mixed_model.attention_alphas)
                new_attention_alphas = torch.full_like(
                    mixed_model.attention_alphas, -10.0
                )
                new_attention_alphas[attention_best] = 10.0
                mixed_model.attention_alphas.copy_(new_attention_alphas)

    @staticmethod
    def create_fixed_encoder(mixed_encoder, **kwargs) -> "FixedEncoder":
        best_type = ArchitectureConverter.get_best_architecture(
            mixed_encoder.get_alphas()
        )
        if best_type == "patch":
            fixed_encoder = FixedEncoder(
                rnn=copy.deepcopy(mixed_encoder.patch_encoder),
                input_dim=mixed_encoder.input_dim,
                latent_dim=mixed_encoder.latent_dim,
                **kwargs,
            )
        else:
            fixed_encoder = FixedEncoder(
                rnn_type=best_type,
                input_dim=mixed_encoder.input_dim,
                latent_dim=mixed_encoder.latent_dim,
                **kwargs,
            )
        ArchitectureConverter._transfer_encoder_weights(
            mixed_encoder, fixed_encoder, best_type
        )
        return fixed_encoder

    @staticmethod
    def create_fixed_decoder(mixed_decoder, **kwargs) -> "FixedDecoder":
        best_type = ArchitectureConverter.get_best_architecture(
            mixed_decoder.get_alphas()
        )
        attention_variant = kwargs.get("attention_variant", "sdp")

        fixed_decoder = FixedDecoder(
            rnn_type=best_type,
            input_dim=mixed_decoder.input_dim,
            latent_dim=mixed_decoder.latent_dim,
            use_attention_bridge=kwargs.get(
                "use_attention_bridge", mixed_decoder.use_attention_bridge
            ),
            attention_variant=attention_variant,
            **{
                k: v
                for k, v in kwargs.items()
                if k not in {"use_attention_bridge", "attention_variant"}
            },
        )
        ArchitectureConverter._transfer_decoder_weights(
            mixed_decoder, fixed_decoder, best_type, attention_variant=attention_variant
        )
        return fixed_decoder

    @staticmethod
    def _transfer_encoder_weights(mixed_encoder, fixed_encoder, arch_type: str):
        try:
            if arch_type == "lstm":
                source_rnn = mixed_encoder.lstm
            elif arch_type == "gru":
                source_rnn = mixed_encoder.gru
            elif arch_type == "transformer":
                source_rnn = mixed_encoder.transformer
            elif arch_type == "patch":
                source_rnn = mixed_encoder.patch_encoder
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_encoder.rnn.load_state_dict(source_rnn.state_dict())

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
                    with torch.no_grad():
                        logits = fixed_encoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_encoder.searchable_decomp.alpha_logits.requires_grad_(False)
        except Exception as e:
            print(f"Warning: Could not transfer encoder weights: {e}")

    @staticmethod
    def _transfer_decoder_weights(
        mixed_decoder, fixed_decoder, arch_type: str, attention_variant: str = "sdp"
    ):
        try:
            if arch_type == "lstm":
                source_rnn = mixed_decoder.lstm
            elif arch_type == "gru":
                source_rnn = mixed_decoder.gru
            elif arch_type == "transformer":
                source_rnn = mixed_decoder.transformer
            else:
                raise ValueError(f"Unknown architecture type: {arch_type}")

            fixed_decoder.rnn.load_state_dict(source_rnn.state_dict())

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
                    with torch.no_grad():
                        logits = fixed_decoder.searchable_decomp.alpha_logits
                        hard = torch.full_like(logits, -10.0)
                        hard[int(torch.argmax(logits).item())] = 10.0
                        logits.copy_(hard)
                    fixed_decoder.searchable_decomp.alpha_logits.requires_grad_(False)

            if (
                fixed_decoder.use_attention_bridge
                and hasattr(mixed_decoder, "attention_bridge")
                and hasattr(fixed_decoder, "attention_bridge")
            ):
                try:
                    fixed_decoder.attention_bridge.load_state_dict(
                        mixed_decoder.attention_bridge.state_dict(), strict=False
                    )
                except Exception as e:
                    print(f"Warning: Could not transfer attention bridge weights: {e}")

        except Exception as e:
            print(f"Warning: Could not transfer decoder weights: {e}")


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
    ):
        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=LightweightTransformerEncoder,
        )
        self.normalizer = None
        self.context_proj = None
        self.searchable_decomp = None

    def forward(self, x: torch.Tensor) -> tuple:
        if self.searchable_decomp is not None:
            x = self.searchable_decomp(x, temperature=0.01)

        if self.rnn_type in ("transformer", "patch"):
            output, context, state = self.rnn(x)
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
        use_attention_bridge: bool = False,
        attention_layers: int = 1,
        attention_variant: str = "sdp",
    ):
        self.use_attention_bridge = use_attention_bridge
        self.attention_variant = str(attention_variant).lower()

        super().__init__(
            rnn=rnn,
            rnn_type=rnn_type,
            input_dim=input_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            transformer_factory=LightweightTransformerDecoder,
        )

        if use_attention_bridge:
            attn_type = (
                self.attention_variant
                if self.attention_variant in AttentionBridge.MODES
                else "sdp"
            )
            self.attention_bridge = AttentionBridge(
                latent_dim, num_heads=4, dropout=dropout, attn_type=attn_type
            )
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

        hidden_state = ArchitectureConverter.ensure_proper_state_format(
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

        if self.use_attention_bridge and hasattr(self, "attention_bridge"):
            attention_source = encoder_output if encoder_output is not None else memory
            if attention_source is not None:
                output = self.attention_bridge(output, attention_source)

        return output, new_state
