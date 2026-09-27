"""
TimeSeriesDARTS - Main DARTS model for time series forecasting.

This module contains the TimeSeriesDARTS class which implements the full
DARTS architecture with MixedEncoder, DARTSCells, and MixedDecoder.
"""

from __future__ import annotations

import copy
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.nn.normalization import RevIN

from ...utils.tensors import hard_one_hot
from ..blocks.moe import DARTSFeedForward
from ..common.norms import NORM_NAMES, RMSNorm
from .darts_cell import DARTSCell
from .decoding import DecoderRuntime, _teacher_forcing_flags
from .pruning import OperationPruner
from .mixed_encoder_decoder import MixedDecoder, MixedEncoder
from .mixed_op import MixedOp

__all__ = ["TimeSeriesDARTS"]


class SearchableNorm(nn.Module):
    """Searchable normalization: RevIN / InstanceNorm / RMSNorm / passthrough"""

    def __init__(self, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.inst_norm = nn.InstanceNorm1d(num_features, affine=True)
        self.rms_norm = RMSNorm(num_features, eps=1e-8)

    def forward(self, x, alpha):  # x: [B, L, C]
        w = F.softmax(alpha, dim=0)
        rev = self.revin(x, mode="norm")
        inst = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
        rms = self.rms_norm(x)
        return w[0] * rev + w[1] * inst + w[2] * rms + w[3] * x

    def apply_output_denorm(
        self, y: torch.Tensor, alpha: torch.Tensor, selected_norm: str | None = None
    ) -> torch.Tensor:
        """
        Denormalize output when RevIN is selected.
        - If selected_norm is provided (derived/fixed architecture), use hard behavior.
        - During search, use straight-through hard routing to avoid partial denorm blending.
        """
        if selected_norm is not None:
            if str(selected_norm).lower() != "revin":
                return y
            try:
                return self.revin(y, mode="denorm")
            except Exception:
                return y

        w_soft = F.softmax(alpha, dim=0)
        hard = hard_one_hot(w_soft)
        # Straight-through estimator: forward is hard, backward follows soft weights.
        w = hard - w_soft.detach() + w_soft if self.training else hard
        try:
            rev_denorm = self.revin(y, mode="denorm")
        except Exception:
            return y
        return w[0] * rev_denorm + (1.0 - w[0]) * y



class TimeSeriesDARTS(nn.Module):
    """Simplified TimeSeriesDARTS with essential features"""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        forecast_horizon: int = 24,
        seq_length: int = 48,
        num_cells: int = 2,
        num_nodes: int = 4,
        dropout: float = 0.1,
        initial_search: bool = False,
        selected_ops: list | None = None,
        loss_type: str = "huber",
        use_gradient_checkpointing: bool = False,
        temperature: float = 1.0,
        temperature_schedule: str = "cosine",
        final_temperature: float = 0.1,
        temperature_warmup_epochs: int = 0,
        variant_gdas: bool = True,
        use_learned_memory_pooling: bool = True,
        memory_num_queries: int = 8,
        group_temperature_mult: float = 1.5,
        min_group_temperature: float = 0.7,
        min_op_temperature: float = 0.3,
        group_min_prob: float = 0.03,
        competition_power: float = 1.25,
        adaptive_bias_scale: float = 0.15,
        performance_ema_decay: float = 0.95,
        pc_ratio: float = 0.25,
        max_active_edges_per_node: int = 2,
        progressive_edge_budget: bool = True,
        use_drnas: bool = True,
        drnas_concentration: float = 8.0,
        use_fair_darts_hierarchical: bool = True,
        arch_mode: str = "encoder_decoder",
        tie_encoder_decoder_arch: bool = True,
        transformer_self_attention_type: str = "auto",
        transformer_ffn_variant: str = "auto",
        op_gdas: bool = True,
    ):
        super().__init__()

        _VALID_ARCH_MODES = {"encoder_decoder", "encoder_only", "decoder_only"}
        if arch_mode not in _VALID_ARCH_MODES:
            raise ValueError(
                f"arch_mode must be one of {sorted(_VALID_ARCH_MODES)}, got '{arch_mode}'"
            )

        # Validation: structural constraints
        if seq_length <= 0:
            raise ValueError(f"seq_length must be > 0, got {seq_length}")
        if forecast_horizon <= 0:
            raise ValueError(f"forecast_horizon must be > 0, got {forecast_horizon}")
        if seq_length < forecast_horizon:
            raise ValueError(
                f"seq_length ({seq_length}) must be >= forecast_horizon "
                f"({forecast_horizon}) for non-autoregressive forecast"
            )
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be > 0, got {latent_dim}")
        if num_cells <= 0:
            raise ValueError(f"num_cells must be > 0, got {num_cells}")
        if num_nodes < 2:
            raise ValueError(f"num_nodes must be >= 2 (need edges), got {num_nodes}")
        if not (0.0 <= dropout < 1.0):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if not (0.0 <= pc_ratio <= 1.0):
            raise ValueError(f"pc_ratio must be in [0, 1], got {pc_ratio}")

        resolved_self_attention_type = str(transformer_self_attention_type).lower()
        resolved_transformer_ffn_variant = str(transformer_ffn_variant).lower()
        if resolved_transformer_ffn_variant not in {"auto", "swiglu", "moe"}:
            raise ValueError(
                "transformer_ffn_variant must be one of ['auto', 'moe', 'swiglu'], "
                f"got '{transformer_ffn_variant}'"
            )

        self._config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "latent_dim": latent_dim,
            "forecast_horizon": forecast_horizon,
            "seq_length": seq_length,
            "num_cells": num_cells,
            "num_nodes": num_nodes,
            "selected_ops": selected_ops,
            "variant_gdas": variant_gdas,
            "temperature_schedule": temperature_schedule,
            "final_temperature": final_temperature,
            "temperature_warmup_epochs": temperature_warmup_epochs,
            "use_learned_memory_pooling": use_learned_memory_pooling,
            "memory_num_queries": memory_num_queries,
            "group_temperature_mult": group_temperature_mult,
            "min_group_temperature": min_group_temperature,
            "min_op_temperature": min_op_temperature,
            "group_min_prob": group_min_prob,
            "competition_power": competition_power,
            "adaptive_bias_scale": adaptive_bias_scale,
            "performance_ema_decay": performance_ema_decay,
            "pc_ratio": pc_ratio,
            "max_active_edges_per_node": max_active_edges_per_node,
            "progressive_edge_budget": progressive_edge_budget,
            "use_drnas": use_drnas,
            "drnas_concentration": drnas_concentration,
            "use_fair_darts_hierarchical": use_fair_darts_hierarchical,
            "arch_mode": arch_mode,
            "tie_encoder_decoder_arch": tie_encoder_decoder_arch,
            "transformer_self_attention_type": resolved_self_attention_type,
            "transformer_ffn_variant": resolved_transformer_ffn_variant,
            "op_gdas": op_gdas,
        }

        # Store configuration
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon
        self.seq_length = seq_length
        self.num_cells = num_cells
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.initial_search = initial_search
        self.selected_ops = selected_ops
        self.loss_type = loss_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.temperature = temperature
        self.initial_temperature = float(temperature)
        self.temperature_schedule = temperature_schedule
        self.final_temperature = float(final_temperature)
        self.temperature_warmup_epochs = int(temperature_warmup_epochs)
        self.variant_gdas = variant_gdas
        self.use_learned_memory_pooling = use_learned_memory_pooling
        self.memory_num_queries = memory_num_queries
        self.group_temperature_mult = group_temperature_mult
        self.min_group_temperature = min_group_temperature
        self.min_op_temperature = min_op_temperature
        self.group_min_prob = group_min_prob
        self.competition_power = competition_power
        self.adaptive_bias_scale = adaptive_bias_scale
        self.performance_ema_decay = performance_ema_decay
        self.pc_ratio = float(min(max(pc_ratio, 0.0), 1.0))
        self.max_active_edges_per_node = max(0, int(max_active_edges_per_node))
        self.progressive_edge_budget = bool(progressive_edge_budget)
        self.use_drnas = use_drnas
        self.drnas_concentration = drnas_concentration
        self.use_fair_darts_hierarchical = use_fair_darts_hierarchical
        self.arch_mode = arch_mode
        self.tie_encoder_decoder_arch = bool(tie_encoder_decoder_arch)
        self.transformer_self_attention_type = resolved_self_attention_type
        self.transformer_ffn_variant = resolved_transformer_ffn_variant
        self.transformer_use_moe = resolved_transformer_ffn_variant == "moe"
        self.op_gdas = bool(op_gdas)

        # Searchable normalization
        self.norm_strategy = SearchableNorm(self.input_dim)
        self.norm_alpha = nn.Parameter(torch.zeros(len(NORM_NAMES)))

        # Pruning state tracking
        self.pruning_history = []
        self.operation_performance = {}
        self.pruned_operations = set()
        self._init_components()

    def get_config(self):
        return copy.deepcopy(self._config)

    @classmethod
    def from_config(cls, config: dict):
        return cls(**config)

    def _init_components(self):
        """Initialize all model components"""
        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )

        # DARTS cells with projections and scaling
        self.cells = nn.ModuleList()
        self.cell_proj = nn.ModuleList()
        self.layer_scales = nn.ParameterList()

        for i in range(self.num_cells):
            # Temperature decay for deeper cells
            temp = self.temperature * (0.8**i)

            # Cell
            self.cells.append(
                DARTSCell(
                    input_dim=self.hidden_dim,
                    latent_dim=self.latent_dim,
                    seq_length=self.seq_length,
                    num_nodes=self.num_nodes,
                    initial_search=self.initial_search,
                    selected_ops=self.selected_ops,
                    aggregation="weighted",
                    temperature=temp,
                    use_checkpoint=self.use_gradient_checkpointing,
                    group_temperature_mult=self.group_temperature_mult,
                    min_group_temperature=self.min_group_temperature,
                    min_op_temperature=self.min_op_temperature,
                    group_min_prob=self.group_min_prob,
                    competition_power=self.competition_power,
                    adaptive_bias_scale=self.adaptive_bias_scale,
                    performance_ema_decay=self.performance_ema_decay,
                    pc_ratio=self.pc_ratio,
                    max_active_edges_per_node=self.max_active_edges_per_node,
                    progressive_edge_budget=self.progressive_edge_budget,
                    use_drnas=self.use_drnas,
                    drnas_concentration=self.drnas_concentration,
                    use_fair_darts_hierarchical=self.use_fair_darts_hierarchical,
                    op_gdas=self.op_gdas,
                )
            )

            # Projection layer
            self.cell_proj.append(
                nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim, bias=False),
                    nn.LayerNorm(self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(self.dropout * 0.5),
                )
            )

            # Layer scaling
            self.layer_scales.append(nn.Parameter(torch.ones(1) * 0.1))

        # Cell combination weights
        self.cell_weights = nn.Parameter(torch.ones(self.num_cells) * 0.5)
        self.cell_importance = nn.Parameter(torch.ones(self.num_cells) * 0.8)
        self.global_skip = nn.Parameter(torch.tensor(0.1))

        self.feature_gate = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim), nn.Sigmoid()
        )

        self.feature_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
        )
        # Encoder and decoder — built conditionally on arch_mode
        if self.arch_mode in ("encoder_decoder", "encoder_only"):
            self.forecast_encoder = MixedEncoder(
                self.hidden_dim,
                self.latent_dim,
                seq_len=self.seq_length,
                dropout=self.dropout,
                temperature=self.temperature,
                variant_gdas=self.variant_gdas,
                include_patch=True,
                transformer_self_attention_type=self.transformer_self_attention_type,
                transformer_use_moe=self.transformer_use_moe,
                transformer_ffn_variant=self.transformer_ffn_variant,
                use_checkpoint=self.use_gradient_checkpointing,
            )
        else:
            self.forecast_encoder = None  # type: ignore[assignment]

        if self.arch_mode in ("encoder_decoder", "decoder_only"):
            decoder_cross_attention_modes = (
                ("sdp", "linear", "probsparse", "cosine", "local")
                if self.arch_mode == "encoder_decoder"
                else None
            )
            self.forecast_decoder = MixedDecoder(
                self.input_dim,
                self.latent_dim,
                seq_len=self.seq_length,
                dropout=self.dropout,
                temperature=self.temperature,
                use_learned_memory_pooling=self.use_learned_memory_pooling,
                memory_num_queries=self.memory_num_queries,
                variant_gdas=self.variant_gdas,
                transformer_self_attention_type=self.transformer_self_attention_type,
                transformer_cross_attention_modes=decoder_cross_attention_modes,
                transformer_use_moe=self.transformer_use_moe,
                transformer_ffn_variant=self.transformer_ffn_variant,
                use_checkpoint=self.use_gradient_checkpointing,
            )
        else:
            self.forecast_decoder = None  # type: ignore[assignment]

        # Encoder-only direct forecast head: pool last encoder state → project
        if self.arch_mode == "encoder_only":
            self.enc_only_head = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
                nn.Linear(self.latent_dim, self.forecast_horizon * self.input_dim),
            )
            # Learnable blend between mean-pool and last-token (replaces hardcoded 50/50)
            self.enc_only_pool_weight = nn.Parameter(torch.tensor(0.5))

        # Decoder-only context projection for autoregressive decoder memory.
        if self.arch_mode == "decoder_only":
            self.dec_context_proj = nn.Sequential(
                nn.Linear(self.hidden_dim, self.latent_dim, bias=False),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
                nn.Dropout(self.dropout),
            )

        # Feature fusion
        self.gate_fuse = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
        )

        # output_layer is only used in encoder_decoder mode
        self.output_layer = nn.Linear(self.latent_dim, self.input_dim, bias=False)
        self.decoder_query_mode_names = (
            "repeat_last",
            "zeros",
            "learned_horizon_queries",
            "shifted_target",
            "future_covariate_queries",
        )
        if self.forecast_decoder is not None:
            self.decoder_query_mode = "auto"
            self.decoder_query_alphas = nn.Parameter(
                0.01 * torch.randn(len(self.decoder_query_mode_names))
            )
            self.learned_horizon_queries = nn.Parameter(
                torch.zeros(1, self.forecast_horizon, self.input_dim)
            )
            nn.init.trunc_normal_(self.learned_horizon_queries, std=0.02)
            # Autoformer decoder needs input_dim -> latent_dim projection
            if self.input_dim != self.latent_dim:
                self._autoformer_proj = nn.Linear(self.input_dim, self.latent_dim, bias=False)
            else:
                self._autoformer_proj = nn.Identity()

        else:
            self.decoder_query_mode = "repeat_last"

        self.residual_weights = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1)) for _ in range(self.num_cells)
        ])

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _ensure_dtype(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is float32 for consistent computation"""
        return tensor.float() if tensor.dtype != torch.float32 else tensor

    def forward(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Forward pass"""
        B, L, _ = x_seq.shape

        # Ensure consistent dtype
        x_seq = self._ensure_dtype(x_seq)

        # Searchable normalization (RevIN/InstanceNorm/Identity)
        x_seq = self.norm_strategy(x_seq, self.norm_alpha)

        # Input embedding
        x_emb = self.input_embedding(x_seq)
        original_input = x_emb

        # Process through enhanced DARTS cells
        current_input = x_emb
        cell_outputs = []

        for i, (cell, proj, scale, res_weight, importance) in enumerate(
            zip(
                self.cells,
                self.cell_proj,
                self.layer_scales,
                self.residual_weights,
                self.cell_importance,
            )
        ):
            # Apply cell with optional checkpointing
            if self.training and self.use_gradient_checkpointing:
                cell_out = torch.utils.checkpoint.checkpoint(
                    cell, current_input, use_reentrant=False
                )
            else:
                cell_out = cell(current_input)

            # Project and scale
            projected = proj(cell_out) * scale * torch.sigmoid(importance)
            cell_outputs.append(projected)

            # Enhanced residual connection between cells
            if i > 0:
                residual_contrib = current_input * res_weight
                current_input = cell_out + residual_contrib
            else:
                current_input = cell_out

        # Enhanced cell feature combination
        if len(cell_outputs) > 1:
            # Learnable weighted combination
            cell_weights_norm = F.softmax(self.cell_weights[: len(cell_outputs)], dim=0)
            cell_importance_norm = torch.sigmoid(
                self.cell_importance[: len(cell_outputs)]
            )

            # Combine with both weights and importance
            final_weights = cell_weights_norm * cell_importance_norm
            final_weights = final_weights / final_weights.sum()

            combined_features = sum(w * f for w, f in zip(final_weights, cell_outputs))
        else:
            combined_features = cell_outputs[0]

        # Enhanced feature fusion with gating
        concatenated = torch.cat([combined_features, original_input], dim=-1)
        gate = self.feature_gate(concatenated)
        gated_features = gate * combined_features + (1 - gate) * original_input

        # Apply feature transformation
        final_features = self.feature_transform(gated_features)

        # Global skip connection
        final_features = final_features + self.global_skip * original_input

        # ── Branch on architecture topology ──────────────────────────────
        selected_norm = getattr(self, "selected_norm", None)

        if self.arch_mode == "encoder_only":
            return self._forward_encoder_only(final_features, x_seq, selected_norm)

        if self.arch_mode == "decoder_only":
            return self._forward_decoder_only(
                final_features,
                x_seq,
                selected_norm,
                x_future=x_future,
                decoder_targets=decoder_targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
            )

        # ── encoder_decoder (original path) ──────────────────────────────
        h_enc, context, encoder_state = self.forecast_encoder(final_features)

        decoder_hidden = encoder_state

        context = self._ensure_dtype(context)
        h_enc = self._ensure_dtype(h_enc)

        forecasts_tensor = self._decode_with_style(
            x_seq=x_seq,
            x_future=x_future,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            memory=context,
            encoder_output=h_enc,
            decoder_hidden=decoder_hidden,
        )
        forecasts_tensor = self.norm_strategy.apply_output_denorm(
            forecasts_tensor,
            self.norm_alpha,
            selected_norm=selected_norm,
        )
        return forecasts_tensor

    def _forward_encoder_only(
        self,
        final_features: torch.Tensor,
        x_seq: torch.Tensor,
        selected_norm: str | None,
    ) -> torch.Tensor:
        """Non-autoregressive path: encode → pool last state → direct projection."""
        B = x_seq.shape[0]
        h_enc, _context, _state = self.forecast_encoder(final_features)
        # Learnable blend between mean-pool and last-token
        w = torch.sigmoid(self.enc_only_pool_weight)
        pooled = h_enc.mean(dim=1) * w + h_enc[:, -1, :] * (1.0 - w)  # [B, latent_dim]
        pooled = self._ensure_dtype(pooled)
        out = self.enc_only_head(pooled)  # [B, H*C]
        out = out.reshape(B, self.forecast_horizon, self.input_dim)
        out = self.norm_strategy.apply_output_denorm(
            out, self.norm_alpha, selected_norm=selected_norm
        )
        return out

    def _forward_decoder_only(
        self,
        final_features: torch.Tensor,
        x_seq: torch.Tensor,
        selected_norm: str | None,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        """Decoder-only path: DARTS backbone context + autoregressive decoder."""
        if self.forecast_decoder is None:
            raise RuntimeError("decoder_only mode requires forecast_decoder")

        context = self.dec_context_proj(final_features)
        context = self._ensure_dtype(context)

        decoder_hidden = None
        forecasts_tensor = self._decode_with_style(
            x_seq=x_seq,
            x_future=x_future,
            decoder_targets=decoder_targets,
            teacher_forcing_ratio=teacher_forcing_ratio,
            memory=context,
            encoder_output=context,
            decoder_hidden=decoder_hidden,
        )
        forecasts_tensor = self.norm_strategy.apply_output_denorm(
            forecasts_tensor,
            self.norm_alpha,
            selected_norm=selected_norm,
        )
        return forecasts_tensor

    def _decoder_runtime(self) -> DecoderRuntime:
        return DecoderRuntime(
            forecast_decoder=self.forecast_decoder,
            output_layer=self.output_layer,
            forecast_horizon=self.forecast_horizon,
            latent_dim=self.latent_dim,
            learned_horizon_queries=self.learned_horizon_queries,
            _autoformer_proj=self._autoformer_proj,
            norm_alpha=self.norm_alpha,
            decoder_query_mode_names=self.decoder_query_mode_names,
            decoder_query_alphas=getattr(self, "decoder_query_alphas", None),
            decoder_query_mode=getattr(self, "decoder_query_mode", None),
            temperature=self.temperature,
            variant_gdas=self.variant_gdas,
            training=self.training,
        )

    def _operation_pruner(self) -> OperationPruner:
        return OperationPruner(
            self.cells, self.pruning_history, self.pruned_operations,
            self.operation_performance,
        )

    def _get_decoder_query_mode_weights(self) -> torch.Tensor | None:
        return self._decoder_runtime()._get_decoder_query_mode_weights()

    def get_decoder_query_mode_probs(self) -> torch.Tensor | None:
        return self._decoder_runtime().get_decoder_query_mode_probs()

    def resolve_decoder_query_mode(self) -> str:
        return self._decoder_runtime().resolve_decoder_query_mode()

    def freeze_decoder_query_mode(self, query_mode: str) -> None:
        resolved = (
            str(query_mode).lower()
            if str(query_mode).lower() in self.decoder_query_mode_names
            else "repeat_last"
        )
        self.decoder_query_mode = resolved
        if hasattr(self, "decoder_query_alphas"):
            del self.decoder_query_alphas

    def _build_parallel_decoder_input(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
    ) -> torch.Tensor:
        return self._decoder_runtime()._build_parallel_decoder_input(x_seq, x_future, decoder_targets, teacher_forcing_ratio)

    def _decode_autoregressive_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None = None,
        decoder_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.5,
        memory: torch.Tensor | None = None,
        encoder_output: torch.Tensor | None = None,
        decoder_hidden=None,
    ) -> torch.Tensor:
        return self._decoder_runtime()._decode_autoregressive_path(x_seq, x_future, decoder_targets, teacher_forcing_ratio, memory, encoder_output, decoder_hidden)

    def _decode_parallel_informer_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None,
        decoder_targets: torch.Tensor | None,
        teacher_forcing_ratio: float,
        memory: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden,
    ) -> torch.Tensor:
        return self._decoder_runtime()._decode_parallel_informer_path(x_seq, x_future, decoder_targets, teacher_forcing_ratio, memory, encoder_output, decoder_hidden)

    def _get_decoder_layers(self):
        return self._decoder_runtime()._get_decoder_layers()

    def _decode_autoformer_path(
        self,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None,
        decoder_targets: torch.Tensor | None,
        teacher_foring_ratio: float,
        memory: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden,
    ) -> torch.Tensor:
        return self._decoder_runtime()._decode_autoformer_path(x_seq, x_future, decoder_targets, teacher_foring_ratio, memory, encoder_output, decoder_hidden)

    def _build_decoder_style_weights(self, decoder) -> torch.Tensor | None:
        """Get decoder style weights for 3 styles: ar, informer, autoformer."""
        if decoder is None:
            return None
        getter = getattr(decoder, "get_decode_style_weights", None)
        if callable(getter):
            try:
                weights = getter()
                if weights is not None and weights.numel() >= 3:
                    return weights
            except Exception:
                pass

        # Check for 3-element alphas
        alphas = getattr(decoder, "decode_style_alphas", None)
        if isinstance(alphas, torch.Tensor) and alphas.numel() == 3:
            return alphas

        style = getattr(decoder, "decode_style", None)
        if isinstance(style, str):
            ref = next(decoder.parameters(), None)
            if ref is not None:
                weights = ref.new_zeros(3)
                if style == "informer":
                    weights[1] = 1.0
                elif style == "autoformer":
                    weights[2] = 1.0
                else:
                    weights[0] = 1.0
                return weights
        return None

    def _decode_with_style(
        self,
        *,
        x_seq: torch.Tensor,
        x_future: torch.Tensor | None,
        decoder_targets: torch.Tensor | None,
        teacher_forcing_ratio: float,
        memory: torch.Tensor,
        encoder_output: torch.Tensor,
        decoder_hidden,
    ) -> torch.Tensor:
        style_weights = self._build_decoder_style_weights(self.forecast_decoder)
        if style_weights is None or style_weights.numel() < 2:
            return self._decode_autoregressive_path(
                x_seq=x_seq,
                x_future=x_future,
                decoder_targets=decoder_targets,
                teacher_forcing_ratio=teacher_forcing_ratio,
                memory=memory,
                encoder_output=encoder_output,
                decoder_hidden=decoder_hidden,
            )

        # Resolve decoder style: select best path via argmax
        idx = int(torch.argmax(style_weights.detach()).item())
        decode_methods = [
            self._decode_autoregressive_path,
            self._decode_parallel_informer_path,
            self._decode_autoformer_path,
        ]
        # Clamp to available methods
        idx = min(idx, len(decode_methods) - 1)
        decoded = decode_methods[idx](
            x_seq,
            x_future,
            decoder_targets,
            teacher_forcing_ratio,
            memory,
            encoder_output,
            decoder_hidden,
        )
        # Keep the selected path connected to its architecture logit. With
        # hard Gumbel weights this is the straight-through DARTS gradient.
        gate = 1.0 + style_weights[idx] - style_weights[idx].detach()
        return gate * decoded

    # Analysis methods
    def get_all_alphas(self) -> dict[str, torch.Tensor]:
        """Extract all architecture parameters"""
        alphas = {}

        # Normalization alphas
        if hasattr(self, "norm_alpha"):
            alphas["norm"] = F.softmax(self.norm_alpha, dim=0)

        # Cell alphas
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas"):
                        alphas[f"cell_{i}_edge_{j}"] = edge.get_alphas()

        def _extract_self_attn_probs(component):
            if component is None:
                return None
            submodule = getattr(component, "transformer", None)
            if submodule is None:
                submodule = getattr(component, "rnn", None)
            if submodule is None:
                return None
            layers = getattr(submodule, "layers", None)
            if not layers:
                return None
            first = layers[0]
            self_attn = None
            if isinstance(first, dict) or hasattr(first, "get"):
                self_attn = first.get("self_attn")
            elif hasattr(first, "__contains__") and "self_attn" in first:
                self_attn = first["self_attn"]
            logits = (
                getattr(self_attn, "attn_alphas", None)
                if self_attn is not None
                else None
            )
            if logits is None:
                return None
            return F.softmax(logits, dim=0)

        def _extract_cross_attn_probs(component):
            if component is None:
                return None
            submodule = getattr(component, "transformer", None)
            if submodule is None:
                submodule = getattr(component, "rnn", None)
            if submodule is None:
                return None
            layers = getattr(submodule, "layers", None)
            if not layers:
                return None
            first = layers[0]
            cross_attn = None
            if isinstance(first, dict) or hasattr(first, "get"):
                cross_attn = first.get("cross_attn")
            elif hasattr(first, "__contains__") and "cross_attn" in first:
                cross_attn = first["cross_attn"]
            logits = (
                getattr(cross_attn, "attn_alphas", None)
                if cross_attn is not None
                else None
            )
            if logits is None:
                return None
            return F.softmax(logits, dim=0)

        def _extract_attention_position_probs(component, key: str):
            if component is None:
                return None
            submodule = getattr(component, "transformer", None)
            if submodule is None:
                submodule = getattr(component, "rnn", None)
            if submodule is None:
                return None
            layers = getattr(submodule, "layers", None)
            if not layers:
                return None
            first = layers[0]
            attn = None
            if isinstance(first, dict) or hasattr(first, "get"):
                attn = first.get(key)
            elif hasattr(first, "__contains__") and key in first:
                attn = first[key]
            logits = (
                getattr(attn, "position_alphas", None) if attn is not None else None
            )
            if logits is None:
                return None
            return F.softmax(logits, dim=0)

        def _extract_ffn_probs(component):
            if component is None:
                return None
            submodule = getattr(component, "transformer", None)
            if submodule is None:
                submodule = getattr(component, "rnn", None)
            if submodule is None:
                return None
            layers = getattr(submodule, "layers", None)
            if not layers:
                return None
            first = layers[0]
            ffn = None
            if isinstance(first, dict) or hasattr(first, "get"):
                ffn = first.get("ffn")
            elif hasattr(first, "__contains__") and "ffn" in first:
                ffn = first["ffn"]
            if ffn is None:
                return None
            logits = getattr(ffn, "ffn_alphas", None)
            if logits is not None:
                return F.softmax(logits, dim=0)
            mode = getattr(ffn, "ffn_mode", None)
            modes = tuple(getattr(ffn, "MODE_NAMES", ()))
            if isinstance(mode, str) and mode in modes:
                ref = next(ffn.parameters())
                out = ref.new_zeros(len(modes))
                out[modes.index(mode)] = 1.0
                return out
            return None

        # Encoder/decoder transformer search alphas
        if self.forecast_encoder is not None:
            alphas["encoder"] = torch.tensor([1.0], device=self.norm_alpha.device)
            patch_alpha = getattr(
                getattr(self.forecast_encoder, "transformer", None),
                "patch_alpha_logits",
                None,
            )
            if patch_alpha is not None:
                alphas["encoder_tokenizer"] = F.softmax(patch_alpha, dim=0)
            enc_attn = _extract_self_attn_probs(self.forecast_encoder)
            if enc_attn is not None:
                alphas["encoder_self_attention"] = enc_attn
            enc_pos = _extract_attention_position_probs(
                self.forecast_encoder, "self_attn"
            )
            if enc_pos is not None:
                alphas["encoder_attention_position"] = enc_pos
            enc_ffn = _extract_ffn_probs(self.forecast_encoder)
            if enc_ffn is not None:
                alphas["encoder_ffn"] = enc_ffn
        if self.forecast_decoder is not None:
            alphas["decoder"] = torch.tensor([1.0], device=self.norm_alpha.device)
            if hasattr(self.forecast_decoder, "decode_style_alphas"):
                alphas["decoder_style"] = F.softmax(
                    self.forecast_decoder.decode_style_alphas, dim=0
                )
            if hasattr(self, "decoder_query_alphas"):
                alphas["decoder_query_generator"] = F.softmax(
                    self.decoder_query_alphas, dim=0
                )
            dec_attn = _extract_self_attn_probs(self.forecast_decoder)
            if dec_attn is not None:
                alphas["decoder_self_attention"] = dec_attn
            dec_self_pos = _extract_attention_position_probs(
                self.forecast_decoder, "self_attn"
            )
            if dec_self_pos is not None:
                alphas["decoder_attention_position"] = dec_self_pos
            dec_cross_attn = _extract_cross_attn_probs(self.forecast_decoder)
            if dec_cross_attn is not None:
                alphas["decoder_cross_attention"] = dec_cross_attn
            dec_cross_pos = _extract_attention_position_probs(
                self.forecast_decoder, "cross_attn"
            )
            if dec_cross_pos is not None:
                alphas["decoder_cross_attention_position"] = dec_cross_pos
            dec_ffn = _extract_ffn_probs(self.forecast_decoder)
            if dec_ffn is not None:
                alphas["decoder_ffn"] = dec_ffn
            if hasattr(self.forecast_decoder, "memory_query_alphas"):
                alphas["decoder_memory_queries"] = F.softmax(
                    self.forecast_decoder.memory_query_alphas, dim=0
                )

        return alphas

    def derive_discrete_architecture(self, threshold: float = 0.3) -> dict[str, Any]:
        """Derive discrete architecture from continuous weights"""
        discrete_arch = {}
        weights = self.get_operation_weights()

        for component_name, component_weights in weights.items():
            if not component_weights:  # Skip empty weight dictionaries
                continue

            above_threshold = {
                op: w
                for op, w in component_weights.items()
                if float(w) >= float(threshold)
            }
            candidate_pool = above_threshold if above_threshold else component_weights
            max_op = max(candidate_pool, key=candidate_pool.get)
            max_weight = candidate_pool[max_op]

            if component_name.startswith("cell_"):
                parts = component_name.split("_")
                if len(parts) >= 2:
                    cell_name = f"cell_{parts[1]}"
                    if cell_name not in discrete_arch:
                        discrete_arch[cell_name] = {}
                    edge_name = "_".join(parts[2:]) if len(parts) > 2 else "edge"
                    discrete_arch[cell_name][edge_name] = {
                        "operation": max_op,
                        "weight": max_weight,
                        "passed_threshold": max_op in above_threshold,
                    }
            else:
                discrete_arch[component_name] = {
                    "type": max_op,
                    "weight": max_weight,
                    "passed_threshold": max_op in above_threshold,
                }

        return discrete_arch

    def get_operation_weights(self) -> dict[str, dict[str, float]]:
        """Get normalized operation weights"""
        weights = {}

        # Normalization weights
        if hasattr(self, "norm_alpha"):
            norm_names = list(NORM_NAMES)
            soft_norm = F.softmax(self.norm_alpha, dim=0)
            weights["norm"] = {
                name: weight.item()
                for name, weight in zip(norm_names, soft_norm[: len(norm_names)])
            }

        # Cell weights
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        try:
                            alphas = edge.get_alphas()
                            if alphas.numel() > 0:
                                weights[f"cell_{i}_edge_{j}"] = {
                                    op: weight.item()
                                    for op, weight in zip(
                                        edge.available_ops, F.softmax(alphas, dim=0)
                                    )
                                }
                        except Exception:
                            continue

        weights["encoder"] = (
            {"transformer": 1.0} if self.forecast_encoder is not None else {}
        )
        weights["decoder"] = (
            {"transformer": 1.0} if self.forecast_decoder is not None else {}
        )

        if self.forecast_encoder is not None:
            patch_alpha = getattr(
                getattr(self.forecast_encoder, "transformer", None),
                "patch_alpha_logits",
                None,
            )
            if patch_alpha is not None:
                soft_patch = F.softmax(patch_alpha, dim=0)
                weights["encoder_tokenizer"] = {
                    name: weight.item()
                    for name, weight in zip(
                        getattr(
                            getattr(self.forecast_encoder, "transformer", None),
                            "patch_mode_names",
                            ["direct", "patch_16"],
                        ),
                        soft_patch,
                    )
                }
            enc_transformer = getattr(self.forecast_encoder, "transformer", None)
            if enc_transformer is not None:
                layers = getattr(enc_transformer, "layers", None)
                if layers:
                    first = layers[0]
                    self_attn = None
                    if isinstance(first, dict) or hasattr(first, "get"):
                        self_attn = first.get("self_attn")
                    elif hasattr(first, "__contains__") and "self_attn" in first:
                        self_attn = first["self_attn"]
                    logits = (
                        getattr(self_attn, "attn_alphas", None)
                        if self_attn is not None
                        else None
                    )
                    modes = (
                        getattr(self_attn, "MODES", ()) if self_attn is not None else ()
                    )
                    if logits is not None and modes:
                        probs = F.softmax(logits, dim=0)
                        weights["encoder_self_attention"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(modes, probs)
                        }
                    pos_logits = (
                        getattr(self_attn, "position_alphas", None)
                        if self_attn is not None
                        else None
                    )
                    pos_modes = (
                        getattr(self_attn, "POSITION_MODES", ())
                        if self_attn is not None
                        else ()
                    )
                    if pos_logits is not None and pos_modes:
                        probs = F.softmax(pos_logits, dim=0)
                        weights["encoder_attention_position"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(pos_modes, probs)
                        }
                    ffn = None
                    if isinstance(first, dict) or hasattr(first, "get"):
                        ffn = first.get("ffn")
                    elif hasattr(first, "__contains__") and "ffn" in first:
                        ffn = first["ffn"]
                    ffn_logits = (
                        getattr(ffn, "ffn_alphas", None) if ffn is not None else None
                    )
                    ffn_modes = (
                        getattr(ffn, "MODE_NAMES", ()) if ffn is not None else ()
                    )
                    if ffn_logits is not None and ffn_modes:
                        probs = F.softmax(ffn_logits, dim=0)
                        weights["encoder_ffn"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(ffn_modes, probs)
                        }
                    elif ffn is not None:
                        ffn_mode = getattr(ffn, "ffn_mode", None)
                        if isinstance(ffn_mode, str) and ffn_mode in ffn_modes:
                            weights["encoder_ffn"] = {
                                str(name): float(name == ffn_mode) for name in ffn_modes
                            }

        if self.forecast_decoder is not None:
            if hasattr(self.forecast_decoder, "decode_style_alphas"):
                probs = F.softmax(self.forecast_decoder.decode_style_alphas, dim=0)
                weights["decoder_style"] = {
                    str(name): float(weight.item())
                    for name, weight in zip(
                        getattr(
                            self.forecast_decoder,
                            "decode_style_names",
                            ("autoregressive", "informer"),
                        ),
                        probs,
                    )
                }
            if hasattr(self, "decoder_query_alphas"):
                probs = F.softmax(self.decoder_query_alphas, dim=0)
                weights["decoder_query_generator"] = {
                    str(name): float(weight.item())
                    for name, weight in zip(self.decoder_query_mode_names, probs)
                }
            dec_transformer = getattr(self.forecast_decoder, "transformer", None)
            if dec_transformer is not None:
                layers = getattr(dec_transformer, "layers", None)
                if layers:
                    first = layers[0]
                    self_attn = None
                    cross_attn = None
                    if isinstance(first, dict) or hasattr(first, "get"):
                        self_attn = first.get("self_attn")
                        cross_attn = first.get("cross_attn")
                    elif hasattr(first, "__contains__") and "self_attn" in first:
                        self_attn = first["self_attn"]
                        if "cross_attn" in first:
                            cross_attn = first["cross_attn"]
                    logits = (
                        getattr(self_attn, "attn_alphas", None)
                        if self_attn is not None
                        else None
                    )
                    modes = (
                        getattr(self_attn, "MODES", ()) if self_attn is not None else ()
                    )
                    if logits is not None and modes:
                        probs = F.softmax(logits, dim=0)
                        weights["decoder_self_attention"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(modes, probs)
                        }
                    pos_logits = (
                        getattr(self_attn, "position_alphas", None)
                        if self_attn is not None
                        else None
                    )
                    pos_modes = (
                        getattr(self_attn, "POSITION_MODES", ())
                        if self_attn is not None
                        else ()
                    )
                    if pos_logits is not None and pos_modes:
                        probs = F.softmax(pos_logits, dim=0)
                        weights["decoder_attention_position"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(pos_modes, probs)
                        }
                    cross_logits = (
                        getattr(cross_attn, "attn_alphas", None)
                        if cross_attn is not None
                        else None
                    )
                    cross_modes = (
                        getattr(cross_attn, "MODES", ())
                        if cross_attn is not None
                        else ()
                    )
                    if cross_logits is not None and cross_modes:
                        probs = F.softmax(cross_logits, dim=0)
                        weights["decoder_cross_attention"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(cross_modes, probs)
                        }
                    cross_pos_logits = (
                        getattr(cross_attn, "position_alphas", None)
                        if cross_attn is not None
                        else None
                    )
                    cross_pos_modes = (
                        getattr(cross_attn, "POSITION_MODES", ())
                        if cross_attn is not None
                        else ()
                    )
                    if cross_pos_logits is not None and cross_pos_modes:
                        probs = F.softmax(cross_pos_logits, dim=0)
                        weights["decoder_cross_attention_position"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(cross_pos_modes, probs)
                        }
                    ffn = None
                    if isinstance(first, dict) or hasattr(first, "get"):
                        ffn = first.get("ffn")
                    elif hasattr(first, "__contains__") and "ffn" in first:
                        ffn = first["ffn"]
                    ffn_logits = (
                        getattr(ffn, "ffn_alphas", None) if ffn is not None else None
                    )
                    ffn_modes = (
                        getattr(ffn, "MODE_NAMES", ()) if ffn is not None else ()
                    )
                    if ffn_logits is not None and ffn_modes:
                        probs = F.softmax(ffn_logits, dim=0)
                        weights["decoder_ffn"] = {
                            str(name): float(weight.item())
                            for name, weight in zip(ffn_modes, probs)
                        }
                    elif ffn is not None:
                        ffn_mode = getattr(ffn, "ffn_mode", None)
                        if isinstance(ffn_mode, str) and ffn_mode in ffn_modes:
                            weights["decoder_ffn"] = {
                                str(name): float(name == ffn_mode) for name in ffn_modes
                            }
            if hasattr(self.forecast_decoder, "memory_query_alphas"):
                probs = F.softmax(self.forecast_decoder.memory_query_alphas, dim=0)
                weights["decoder_memory_queries"] = {
                    str(name): float(weight.item())
                    for name, weight in zip(
                        getattr(self.forecast_decoder, "memory_query_options", []),
                        probs,
                    )
                }

        return weights

    def get_moe_balance_loss(self) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for module in self.modules():
            if isinstance(module, DARTSFeedForward) and getattr(
                module, "supports_moe", False
            ):
                losses.append(module.get_balance_loss())
        if not losses:
            ref = next(self.parameters(), None)
            if ref is None:
                return torch.tensor(0.0)
            return ref.new_zeros(())
        return torch.stack(losses).mean()

    def set_temperature(self, temp: float):
        """Update temperature for all components"""
        temp = max(float(temp), 1e-4)
        self.temperature = temp
        for cell in self.cells:
            if hasattr(cell, "set_temperature"):
                cell.set_temperature(temp)
        if self.forecast_encoder is not None:
            self.forecast_encoder.set_temperature(temp)
        if self.forecast_decoder is not None:
            self.forecast_decoder.set_temperature(temp)

    def set_drnas_concentration(self, concentration: float) -> None:
        """Propagate DrNAS Dirichlet concentration to all cells."""
        self.drnas_concentration = float(concentration)
        for cell in self.cells:
            if hasattr(cell, "set_drnas_concentration"):
                cell.set_drnas_concentration(concentration)

    def get_orthogonal_regularization(self) -> torch.Tensor:
        """Aggregate recurrent state-mixing orthogonal regularization."""
        reg = torch.tensor(0.0, device=self.norm_alpha.device)
        if self.forecast_encoder is not None and hasattr(
            self.forecast_encoder, "orthogonal_regularization"
        ):
            reg = reg + self.forecast_encoder.orthogonal_regularization()
        if self.forecast_decoder is not None and hasattr(
            self.forecast_decoder, "orthogonal_regularization"
        ):
            reg = reg + self.forecast_decoder.orthogonal_regularization()
        return reg

    def _schedule_progressive_stage(self, epoch: int, total_epochs: int) -> str:
        """Automatically schedule progressive search stages across all cells."""
        if not self.cells:
            return "basic"

        progress = float(epoch) / float(max(int(total_epochs), 1))
        if progress < 0.25:
            target_stage = "basic"
        elif progress < 0.55:
            target_stage = "intermediate"
        else:
            target_stage = "advanced"

        for cell in self.cells:
            if hasattr(cell, "set_progressive_stage"):
                cell.set_progressive_stage(target_stage)
            else:
                cell.progressive_stage = target_stage
                new_ops = cell.stage_operations.get(target_stage, cell.available_ops)
                cell.available_ops = new_ops
                for edge in getattr(cell, "edges", []):
                    edge.available_ops = list(new_ops)
                    if hasattr(edge, "_init_hierarchical_search"):
                        edge._init_hierarchical_search()
                    if hasattr(edge, "_profile_flops"):
                        edge._profile_flops()

        return target_stage

    def schedule_progressive_stage(self, epoch: int, total_epochs: int) -> str:
        """Public wrapper for epoch-wise progressive-stage scheduling."""
        return self._schedule_progressive_stage(epoch=epoch, total_epochs=total_epochs)

    def schedule_temperature(
        self,
        epoch: int,
        total_epochs: int,
        *,
        schedule_type: str | None = None,
        final_temp: float | None = None,
        warmup_epochs: int | None = None,
    ) -> float:
        """
        Compute and apply temperature using the model's built-in schedule.
        Returns the applied temperature.
        """
        schedule = schedule_type or self.temperature_schedule
        final_t = (
            max(float(final_temp), 1e-4)
            if final_temp is not None
            else max(self.final_temperature, 1e-4)
        )
        warmup = (
            max(int(warmup_epochs), 0)
            if warmup_epochs is not None
            else max(self.temperature_warmup_epochs, 0)
        )

        if total_epochs <= 0:
            self.set_temperature(self.temperature)
            return self.temperature

        if epoch < warmup:
            temp = self.initial_temperature
        else:
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            progress = min(max(progress, 0.0), 1.0)

            if schedule == "cosine":
                temp = (
                    final_t
                    + (self.initial_temperature - final_t)
                    * (1.0 + math.cos(math.pi * progress))
                    / 2.0
                )
            elif schedule == "exponential":
                ratio = final_t / max(self.initial_temperature, 1e-8)
                temp = self.initial_temperature * math.exp(math.log(ratio) * progress)
            elif schedule == "linear":
                temp = (
                    self.initial_temperature
                    + (final_t - self.initial_temperature) * progress
                )
            elif schedule == "step":
                if progress < 0.3:
                    temp = self.initial_temperature
                elif progress < 0.7:
                    temp = 0.5 * self.initial_temperature
                else:
                    temp = final_t
            else:
                temp = self.temperature

        temp = max(temp, final_t)
        self.set_temperature(temp)
        return temp

    # PRUNING METHODS
    def prune_weak_operations(
        self, threshold: float = 0.1, strategy: str = "probability"
    ) -> dict[str, Any]:
        return self._operation_pruner().prune_weak_operations(threshold, strategy)

    def _prune_by_probability(self, threshold: float) -> dict[str, Any]:
        return self._operation_pruner()._prune_by_probability(threshold)

    def _set_edge_op_logit(self, edge: nn.Module, op_idx: int, value: float) -> bool:
        return self._operation_pruner()._set_edge_op_logit(edge, op_idx, value)

    def freeze_pruned_operations(
        self,
        pruning_stats: dict[str, Any] | None = None,
        logit_value: float = -20.0,
    ) -> int:
        return self._operation_pruner().freeze_pruned_operations(pruning_stats, logit_value)

    def _prune_by_gradient_magnitude(self, threshold: float) -> dict[str, Any]:
        return self._operation_pruner()._prune_by_gradient_magnitude(threshold)

    def _prune_by_entropy(self, threshold: float) -> dict[str, Any]:
        return self._operation_pruner()._prune_by_entropy(threshold)

    def _prune_by_performance(self, threshold: float) -> dict[str, Any]:
        return self._operation_pruner()._prune_by_performance(threshold)


__all__ = [
    "MixedOp",
    "DARTSCell",
    "TimeSeriesDARTS",
]
