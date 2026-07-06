"""
Core DARTS blocks - lightweight utilities and configuration.

This module re-exports the core DARTS components. For large classes,
see:
- ``darts.architecture.mixed_op.MixedOp``
- ``darts.architecture.darts_cell.DARTSCell``
- ``darts.architecture.time_series_darts.TimeSeriesDARTS``
"""

from __future__ import annotations

import copy
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.norms import RevIN

from .darts_cell import DARTSCell

# Re-export from split modules for backward compatibility
from .mixed_op import MixedOp
from .time_series_darts import TimeSeriesDARTS


__all__ = [
    "SearchableNorm",
    "DARTSModelConfig",
    "MixedOp",
    "DARTSCell",
    "TimeSeriesDARTS",
]


# --- Searchable Normalization ---
class SearchableNorm(nn.Module):
    """Searchable normalization: RevIN vs InstanceNorm vs passthrough"""

    def __init__(self, num_features):
        super().__init__()
        self.revin = RevIN(num_features)
        self.inst_norm = nn.InstanceNorm1d(num_features, affine=True)

    def forward(self, x, alpha):  # x: [B, L, C]
        w = F.softmax(alpha, dim=0)
        rev = self.revin(x, mode="norm")
        inst = self.inst_norm(x.transpose(1, 2)).transpose(1, 2)
        return w[0] * rev + w[1] * inst + w[2] * x

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
        hard = torch.zeros_like(w_soft)
        hard[int(torch.argmax(w_soft).item())] = 1.0
        # Straight-through estimator: forward is hard, backward follows soft weights.
        w = hard - w_soft.detach() + w_soft if self.training else hard
        try:
            rev_denorm = self.revin(y, mode="denorm")
        except Exception:
            return y
        return w[0] * rev_denorm + (1.0 - w[0]) * y


@dataclass
class DARTSModelConfig:
    """Configuration class for DARTS model"""

    input_dim: int = 3
    hidden_dim: int = 64
    latent_dim: int = 64
    forecast_horizon: int = 24
    seq_length: int = 48
    num_cells: int = 2
    num_nodes: int = 4
    dropout: float = 0.1
    initial_search: bool = False
    selected_ops: list[str] | None = None
    loss_type: str = "huber"
    use_gradient_checkpointing: bool = False
    temperature: float = 1.0
    use_mixed_precision: bool = True
    use_compile: bool = False
    memory_efficient: bool = True
    variant_gdas: bool = True
    use_learned_memory_pooling: bool = True
    memory_num_queries: int = 8

    # New optimization parameters
    arch_lr: float = 3e-4
    weight_lr: float = 1e-3
    alpha_l2_reg: float = 1e-3
    edge_normalization: bool = True
    progressive_shrinking: bool = True

    # MixedOp search-stability knobs
    group_temperature_mult: float = 1.5
    min_group_temperature: float = 0.7
    min_op_temperature: float = 0.3
    group_min_prob: float = 0.03
    competition_power: float = 1.25
    adaptive_bias_scale: float = 0.15
    performance_ema_decay: float = 0.95
    pc_ratio: float = 0.25
    max_active_edges_per_node: int = 2
    progressive_edge_budget: bool = True
    use_drnas: bool = True
    drnas_concentration: float = 8.0
    use_fair_darts_hierarchical: bool = True
    # GDAS: sample a single op per edge via Gumbel-Softmax with a straight-through
    # gradient estimator.  Takes precedence over use_drnas when both are True.
    op_gdas: bool = True

    # β-DARTS: L2 regularization weight on arch logits to prevent premature
    # commitment (skip-connection collapse).  Set to e.g. 1e-3 to enable.
    beta_darts_weight: float = 0.0

    # Architecture topology:
    #   "encoder_decoder" — MixedEncoder + autoregressive MixedDecoder
    #   "encoder_only"    — MixedEncoder + direct projection head
    #   "decoder_only"    — DARTS cells + autoregressive MixedDecoder
    arch_mode: str = "encoder_decoder"
    transformer_self_attention_type: str = "auto"
    transformer_ffn_variant: str = "auto"

    @classmethod
    def with_search_profile(
        cls, profile: str = "conservative", **overrides
    ) -> "DARTSModelConfig":
        """Create config using a named MixedOp search profile."""
        profiles = {
            "conservative": {
                "group_temperature_mult": 1.8,
                "min_group_temperature": 0.9,
                "min_op_temperature": 0.5,
                "group_min_prob": 0.05,
                "competition_power": 1.1,
                "adaptive_bias_scale": 0.10,
                "performance_ema_decay": 0.97,
                "use_drnas": True,
                "drnas_concentration": 10.0,
                "use_fair_darts_hierarchical": True,
            },
            "balanced": {
                # Middle ground: stable enough for small datasets, expressive
                # enough to discover non-trivial operators.
                "group_temperature_mult": 1.5,
                "min_group_temperature": 0.75,
                "min_op_temperature": 0.35,
                "group_min_prob": 0.03,
                "competition_power": 1.25,
                "adaptive_bias_scale": 0.15,
                "performance_ema_decay": 0.95,
                "use_drnas": True,
                "drnas_concentration": 8.0,
                "use_fair_darts_hierarchical": True,
                "beta_darts_weight": 1e-3,
            },
            "aggressive": {
                "group_temperature_mult": 1.2,
                "min_group_temperature": 0.6,
                "min_op_temperature": 0.25,
                "group_min_prob": 0.01,
                "competition_power": 1.5,
                "adaptive_bias_scale": 0.20,
                "performance_ema_decay": 0.90,
                "use_drnas": True,
                "drnas_concentration": 6.0,
                "use_fair_darts_hierarchical": True,
            },
        }

        if profile not in profiles:
            valid = ", ".join(sorted(profiles.keys()))
            raise ValueError(f"Unknown search profile '{profile}'. Valid: {valid}")

        params = {**profiles[profile], **overrides}
        return cls(**params)
