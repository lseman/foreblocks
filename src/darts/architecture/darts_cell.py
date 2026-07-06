"""
DARTSCell - Enhanced DARTS cell with progressive search and better aggregation.

This module contains the DARTSCell class which implements a DARTS cell with
edge importance gating, progressive stage scheduling, and hierarchical search.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bb_primitives import RMSNorm
from .mixed_op import MixedOp


__all__ = ["DARTSCell"]


class DARTSCell(nn.Module):
    """Enhanced DARTS cell with progressive search and better aggregation"""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        seq_length: int,
        num_nodes: int = 4,
        initial_search: bool = False,
        selected_ops: list[str] | None = None,
        aggregation: str = "weighted",
        temperature: float = 1.0,
        use_checkpoint: bool = False,
        progressive_stage: str = "basic",  # "basic", "intermediate", "advanced"
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
        op_gdas: bool = True,
        residual_pattern: str = "auto",  # "additive", "gated", "skip", "auto"
    ):
        super().__init__()
        self.residual_pattern = residual_pattern
        self.residual_searchable = residual_pattern == "auto"
        # Searchable residual pattern: 0=additive, 1=gated, 2=skip
        self.residual_pattern_names = ("additive", "gated", "skip")
        if self.residual_searchable:
            self.residual_pattern_alphas = nn.Parameter(
                0.01 * torch.randn(len(self.residual_pattern_names))
            )
            # Pre-register gates for each node to avoid CPU/CUDA mismatches
            self.gate_weights = nn.ParameterList([
                nn.Parameter(torch.zeros(1)) for _ in range(num_nodes)
            ])
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_nodes = num_nodes
        self.initial_search = initial_search
        self.selected_ops = list(selected_ops) if selected_ops else None
        self.aggregation = aggregation
        self.temperature = temperature
        self.use_checkpoint = use_checkpoint
        self.progressive_stage = progressive_stage
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
        self.op_gdas = bool(op_gdas)

        # Progressive operation selection
        self.stage_operations = {
            "basic": ["Identity", "DLinear", "ResidualMLP", "TimeConv"],
            "intermediate": [
                "Identity",
                "DLinear",
                "TimeConv",
                "TCN",
                "ConvMixer",
                "GRN",
                "TimeMixer",
            ],
            "advanced": [
                "Identity",
                "DLinear",
                "TimeConv",
                "TCN",
                "ConvMixer",
                "GRN",
                "Fourier",
                "Wavelet",
                "MultiScaleConv",
                "PyramidConv",
                "PatchEmbed",
                "InvertedAttention",
                "iTransformerBlock",
                "TimeMixer",
                "NBeats",
                "TimesNet",
            ],
        }

        self.available_ops = self._select_operations(selected_ops)
        self.num_edges = sum(range(num_nodes))

        self._init_components()
        self._edge_indices = self._precompute_edge_indices()

    def _select_operations(self, selected_ops):
        """Select operations based on search stage"""
        if self.initial_search:
            return ["Identity", "TimeConv"]

        if selected_ops:
            return selected_ops

        return self.stage_operations.get(
            self.progressive_stage, self.stage_operations["basic"]
        )

    def _ops_for_stage(self, stage: str) -> list[str]:
        stage_ops = list(
            self.stage_operations.get(stage, self.stage_operations["basic"])
        )
        if self.selected_ops is None:
            return stage_ops

        selected = list(self.selected_ops)
        selected_set = set(selected)
        filtered = [op for op in stage_ops if op in selected_set]
        if "Identity" in selected_set and "Identity" not in filtered:
            filtered.insert(0, "Identity")

        if len(filtered) < 2:
            for op in selected:
                if op not in filtered:
                    filtered.append(op)
                if len(filtered) >= 2:
                    break

        if not filtered:
            filtered = selected[:2] if len(selected) >= 2 else selected

        deduped = []
        seen = set()
        for op in filtered:
            if op not in seen:
                deduped.append(op)
                seen.add(op)
        return deduped

    def set_progressive_stage(self, stage: str) -> bool:
        """Set progressive stage and rebuild edge search spaces if needed."""
        target_ops = self._ops_for_stage(stage)
        needs_update = stage != self.progressive_stage or target_ops != list(
            self.available_ops
        )
        if not needs_update:
            return False

        self.progressive_stage = stage
        self.available_ops = target_ops

        for edge in self.edges:
            edge_param = next(edge.parameters(), None)
            edge_device = edge_param.device if edge_param is not None else None
            edge_dtype = edge_param.dtype if edge_param is not None else None

            edge.available_ops = list(target_ops)
            if hasattr(edge, "_init_hierarchical_search"):
                edge._init_hierarchical_search()
            elif hasattr(edge, "_init_flat_search"):
                edge._init_flat_search()
            if hasattr(edge, "_profile_flops"):
                edge._profile_flops()

            if edge_device is not None:
                edge.to(device=edge_device, dtype=edge_dtype)

            edge.fallback_idx = (
                edge.available_ops.index("Identity")
                if "Identity" in edge.available_ops
                else 0
            )

        return True

    def _precompute_edge_indices(self):
        """Precompute edge indices for faster lookup"""
        return {
            (i, j): sum(range(i)) + j
            for i in range(1, self.num_nodes)
            for j in range(i)
        }

    def _init_components(self):
        """Initialize all components"""
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim, bias=False),
            RMSNorm(self.latent_dim),
            nn.GELU(),
        )

        # Enhanced mixed operations
        self.edges = nn.ModuleList([
            MixedOp(
                self.latent_dim,
                self.latent_dim,
                self.seq_length,
                available_ops=self.available_ops,
                temperature=self.temperature,
                use_hierarchical=True,
                adaptive_sampling=True,
                group_temperature_mult=self.group_temperature_mult,
                min_group_temperature=self.min_group_temperature,
                min_op_temperature=self.min_op_temperature,
                group_min_prob=self.group_min_prob,
                competition_power=self.competition_power,
                adaptive_bias_scale=self.adaptive_bias_scale,
                performance_ema_decay=self.performance_ema_decay,
                pc_ratio=self.pc_ratio,
                use_drnas=self.use_drnas,
                drnas_concentration=self.drnas_concentration,
                use_fair_darts_hierarchical=self.use_fair_darts_hierarchical,
                op_gdas=self.op_gdas,
            )
            for _ in range(self.num_edges)
        ])

        # Learnable residual weights per node
        self.residual_weights = nn.Parameter(torch.full((self.num_nodes,), 0.2))

        # Edge importance weights
        self.edge_importance = nn.Parameter(torch.ones(self.num_edges) * 0.5)

        # Aggregation weights if using weighted aggregation
        if self.aggregation == "weighted":
            self.agg_weights = nn.Parameter(torch.ones(self.num_edges) * 0.1)
        else:
            self.agg_weights = None

        # Output normalization
        self.out_norm = RMSNorm(self.latent_dim)

        # Progressive search parameters
        self.stage_gates = nn.Parameter(
            torch.ones(3)
        )  # [basic, intermediate, advanced]

    def _get_edge_index(self, node_idx, input_idx):
        """Get edge index efficiently"""
        return self._edge_indices[(node_idx, input_idx)]

    def _max_in_edges_for_node(self, node_idx: int) -> int:
        """Active incoming-edge budget for a node."""
        full_count = int(max(node_idx, 1))
        if self.max_active_edges_per_node <= 0:
            return full_count

        max_k = min(int(self.max_active_edges_per_node), full_count)
        if not self.progressive_edge_budget:
            return max_k

        stage = str(getattr(self, "progressive_stage", "advanced"))
        if stage == "basic":
            return min(1, full_count)
        if stage == "intermediate":
            return min(2, max_k, full_count)
        return max_k

    def _aggregate_inputs(self, inputs, edge_indices):
        """Aggregate inputs with different strategies"""
        if len(inputs) == 1:
            return inputs[0]

        stacked = torch.stack(inputs, dim=0)

        if self.aggregation == "weighted" and self.agg_weights is not None:
            # Use edge-specific weights
            weights = F.softmax(
                torch.stack([self.agg_weights[i] for i in edge_indices]), dim=0
            )
            # Match stacked shape [num_inputs, B, L, D] for safe broadcasting.
            weights = weights.reshape(-1, 1, 1, 1)
            return (weights * stacked).sum(dim=0)
        elif self.aggregation == "attention":
            # Simple attention mechanism
            attention_scores = torch.mean(stacked, dim=[2, 3])  # [num_inputs, batch]
            attention_weights = F.softmax(attention_scores, dim=0)
            attention_weights = attention_weights.reshape(-1, 1, 1, 1)
            return (attention_weights * stacked).sum(dim=0)
        elif self.aggregation == "max":
            return torch.max(stacked, dim=0)[0]
        else:  # mean
            return torch.mean(stacked, dim=0)

    def _get_residual_pattern(self) -> str:
        """Resolve the residual pattern (searchable during training)."""
        if not self.residual_searchable:
            return self.residual_pattern
        tau = max(float(self.temperature), 1e-3)
        if self.training:
            probs = F.gumbel_softmax(
                self.residual_pattern_alphas,
                tau=tau,
                hard=False,
                dim=0,
            )
        else:
            probs = F.softmax(self.residual_pattern_alphas / tau, dim=0)
        idx = int(torch.argmax(probs).item())
        return self.residual_pattern_names[idx]

    def _apply_residual(self, node_output, residual_input, node_idx, pattern=None):
        """Apply residual connection with proper dimension handling.

        Patterns:
        - "additive": residual_weight * output + (1 - residual_weight) * skip
        - "gated":    sigmoid(gate) * output + skip
        - "skip":     output + skip (no weighting)
        """
        if pattern is None:
            pattern = self._get_residual_pattern()

        # Handle dimension mismatches
        if node_output.shape != residual_input.shape:
            # Temporal alignment
            if node_output.shape[1] != residual_input.shape[1]:
                residual_input = F.interpolate(
                    residual_input.transpose(1, 2),
                    size=node_output.shape[1],
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)

            # Feature dimension alignment
            if node_output.shape[2] != residual_input.shape[2]:
                if not hasattr(self, f"_residual_proj_{node_idx}"):
                    proj = nn.Linear(residual_input.shape[2], node_output.shape[2]).to(
                        residual_input.device
                    )
                    setattr(self, f"_residual_proj_{node_idx}", proj)
                residual_input = getattr(self, f"_residual_proj_{node_idx}")(
                    residual_input
                )

        if pattern == "gated":
            # Use pre-registered gate weight (avoids CPU/CUDA device mismatch)
            gate_weight = torch.sigmoid(self.gate_weights[node_idx])
            return gate_weight * node_output + residual_input
        if pattern == "skip":
            return node_output + residual_input
        # Default: additive (current behavior)
        residual_weight = torch.sigmoid(self.residual_weights[node_idx])
        return residual_weight * node_output + (1 - residual_weight) * residual_input

    def forward(self, x):
        """Enhanced forward pass with progressive search"""
        if not isinstance(x, torch.Tensor):
            x = torch.stack(x) if isinstance(x, (list, tuple)) else x

        x_proj = self.input_proj(x)
        nodes = [x_proj]

        # Precompute edge importance values as plain Python floats so that
        # control-flow branches (the threshold) never appear inside the
        # dynamo-traced compute graph, avoiding unnecessary graph breaks.
        with torch.no_grad():
            _ew_vals: list[float] = torch.sigmoid(self.edge_importance).tolist()
        _edge_weights = torch.sigmoid(self.edge_importance)  # tracked for backprop

        # GDAS threshold: during GDAS training each edge needs a reasonable
        # chance of being sampled so its alphas can recover from early
        # mis-ranking.  A very low threshold (0.05) causes permanent edge
        # elimination.  We use a floor of ``1/num_edges`` so every edge has
        # at least ``O(1/n)`` weight in the gating distribution, and we
        # require the sigmoid importance to exceed the mean of the edge
        # distribution — this adapts to the actual learned importance
        # landscape rather than a fixed absolute number.
        if self.training and getattr(self.edges[0], "op_gdas", False):
            ew_tensor = torch.sigmoid(self.edge_importance)
            ew_mean = float(ew_tensor.mean().item())
            ew_floor = 1.0 / max(len(self.edges), 1)
            gate_threshold = max(0.12, min(0.25, ew_mean))
        else:
            # Dense (all-edges) mode: only gate out truly dead edges.
            gate_threshold = 0.35

        for node_idx in range(1, self.num_nodes):
            node_inputs, edge_indices = [], []
            candidate_edges = []

            for input_idx in range(node_idx):
                edge_idx = self._get_edge_index(node_idx, input_idx)
                edge = self.edges[edge_idx]

                # Apply edge importance gating (no .item() inside traced graph)
                ew_val = _ew_vals[edge_idx]
                if ew_val > gate_threshold:
                    candidate_edges.append((
                        ew_val,
                        edge_idx,
                        input_idx,
                        _edge_weights[edge_idx],
                    ))

            max_in_edges = self._max_in_edges_for_node(node_idx)
            if len(candidate_edges) > max_in_edges:
                candidate_edges.sort(key=lambda x: x[0], reverse=True)
                candidate_edges = candidate_edges[:max_in_edges]

            for _, edge_idx, input_idx, edge_weight in candidate_edges:
                edge = self.edges[edge_idx]
                # Use gradient checkpointing if enabled
                if self.training and self.use_checkpoint:
                    out = torch.utils.checkpoint.checkpoint(
                        edge, nodes[input_idx], use_reentrant=False
                    )
                else:
                    out = edge(nodes[input_idx])

                # Apply edge weight
                out = out * edge_weight
                node_inputs.append(out)
                edge_indices.append(edge_idx)

            if node_inputs:
                # Aggregate inputs
                agg = self._aggregate_inputs(node_inputs, edge_indices)
                # Apply residual connection
                out = self._apply_residual(agg, nodes[node_idx - 1], node_idx)
            else:
                # Fallback to previous node
                out = nodes[node_idx - 1]

            nodes.append(out)

        # Apply final residual and normalization
        final = self._apply_residual(nodes[-1], x_proj, 0)
        result = self.out_norm(final)
        return result

    def get_residual_pattern_probs(self) -> torch.Tensor | None:
        """Get the residual pattern probabilities for analysis."""
        if self.residual_searchable and hasattr(self, "residual_pattern_alphas"):
            return F.softmax(self.residual_pattern_alphas.detach(), dim=0)
        return None

    def resolve_residual_pattern(self) -> str:
        """Resolve the residual pattern to a hard choice."""
        if self.residual_searchable:
            probs = self.get_residual_pattern_probs()
            if probs is not None:
                idx = int(torch.argmax(probs).item())
                return self.residual_pattern_names[idx]
        return self.residual_pattern

        return result

    def advance_progressive_stage(self):
        """Advance to next progressive search stage"""
        stages = ["basic", "intermediate", "advanced"]
        current_idx = (
            stages.index(self.progressive_stage)
            if self.progressive_stage in stages
            else 0
        )
        if current_idx < len(stages) - 1:
            self.set_progressive_stage(stages[current_idx + 1])

    def get_alphas(self):
        """Get all edge alphas"""
        return [edge.get_alphas() for edge in self.edges]

    def get_entropy_loss(self):
        """Get total entropy loss for exploration"""
        total = sum(edge.get_entropy_loss() for edge in self.edges)

        # Add aggregation entropy if using weighted aggregation
        if self.agg_weights is not None:
            probs = F.softmax(self.agg_weights, dim=0)
            agg_entropy = -(probs * torch.log(probs + 1e-8)).sum()
            total -= 0.005 * agg_entropy

        return total

    def set_temperature(self, temp: float):
        """Update temperature for all edges"""
        self.temperature = temp
        for edge in self.edges:
            edge.set_temperature(temp)

    def set_drnas_concentration(self, concentration: float) -> None:
        """Propagate DrNAS concentration update to all edges."""
        for edge in self.edges:
            if hasattr(edge, "set_drnas_concentration"):
                edge.set_drnas_concentration(concentration)

    def get_edge_statistics(self) -> dict[str, Any]:
        """Get detailed statistics about edge usage"""
        stats = {}
        for i, edge in enumerate(self.edges):
            edge_stats = edge.get_operation_statistics()
            edge_weight = torch.sigmoid(self.edge_importance[i]).item()
            stats[f"edge_{i}"] = {
                "importance_weight": edge_weight,
                "operations": edge_stats,
                "top_ops": edge.describe(top_k=2),
            }
        return stats
