"""Constraint-aware graph representation for ONTS RL.

Extends the bipartite MILP graph to include constraint relaxability features,
allowing the policy to learn when to relax constraints strategically.

Key ideas:
1. Each constraint node has a "relaxability" feature indicating whether it can be relaxed
2. Each constraint edge has a "relaxation cost" feature indicating the penalty for relaxing
3. The policy can select both tasks AND which constraints to relax
4. Relaxing a constraint temporarily removes it from the MILP, allowing the solver
   to find solutions that would otherwise be infeasible

Design rationale: ONTS has many soft constraints (e.g., SOC margins, duty cycle limits)
that can be strategically relaxed to enable better schedules. The policy should learn
which constraints are safe to relax and when.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import math
import numpy as np
import torch
import scipy.sparse as sp

# Import existing graph converter infrastructure
from .graph_converter import (
    GraphConfig,
    GraphTopology,
    MessageProtocol,
    MILPGraphBuilder,
    MILPSolution,
)


class ConstraintType(str, Enum):
    """Types of constraints in ONTS problem."""
    HARD = "hard"           # Must be satisfied (e.g., energy balance)
    SOFT = "soft"           # Can be relaxed with penalty (e.g., SOC margins)
    PREFERENCE = "preference"  # Soft preference, low relaxation cost (e.g., duty cycle)


@dataclass
class ConstraintInfo:
    """Metadata for a single constraint."""
    index: int
    constraint_type: ConstraintType
    relaxable: bool
    relaxation_cost: float  # Penalty per unit violation
    rhs_value: float        # Right-hand side value
    tightness: float        # How tight the constraint is (0=loose, 1=tight)


class ConstraintAwareGraphBuilder(MILPGraphBuilder):
    """Extends MILPGraphBuilder with constraint relaxability features.

    Args:
        A: Constraint matrix [m, n]
        b: RHS vector [m]
        c: Objective coefficients [n]
        bounds: Variable bounds [(lo, hi), ...] [n]
        config: Graph configuration
        constraint_types: List of ConstraintInfo for each constraint [m]
        **kwargs: Passed to parent MILPGraphBuilder
    """

    def __init__(
        self,
        A: torch.Tensor | sp.spmatrix | list[list[float]],
        b: torch.Tensor | list[float],
        c: torch.Tensor | list[float],
        bounds: Optional[list[tuple[float, float]]] = None,
        config: Optional[GraphConfig] = None,
        constraint_info: Optional[list[ConstraintInfo]] = None,
        **kwargs,
    ):
        # Replace Inf values in b with large finite values to avoid NaN in node features
        if hasattr(b, 'numpy'):
            b_arr = b.numpy()
        elif isinstance(b, (list, tuple)):
            b_arr = np.array(b, dtype=np.float64)
        else:
            b_arr = torch.as_tensor(b).numpy()
        
        # Replace Inf with large finite value
        if np.isinf(b_arr).any():
            b_arr = np.where(np.isinf(b_arr), np.sign(b_arr) * 1e6, b_arr)
            # Convert back to tensor if original was tensor
            if isinstance(b, torch.Tensor):
                b = torch.tensor(b_arr, dtype=torch.float32)
            else:
                b = list(b_arr)
        
        super().__init__(A, b, c, bounds, config, **kwargs)
        self.constraint_info = constraint_info or []
        if len(self.constraint_info) != self.m:
            raise ValueError(f"Expected {self.m} constraint info entries, got {len(self.constraint_info)}")

    def _build_node_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build node features with constraint relaxability information.

        Returns:
            v_raw: [n, f_node] variable node features (same as parent)
            c_raw: [m, f_node + 5] constraint node features (extended with relaxability)
        """
        # Get base features from parent
        v_raw, c_raw_base = super()._build_node_features()

        # Extend constraint features with relaxability information
        f_base = c_raw_base.shape[1]
        f_extra = 6  # relaxable(1), relaxation_cost(1), tightness(1), type_onehot(3)
        c_raw = torch.zeros(self.m, f_base + f_extra, dtype=torch.float32)
        c_raw[:, :f_base] = c_raw_base

        # Add constraint relaxability features for each constraint
        # Each constraint gets 5 extra features at fixed offsets from f_base
        for i, info in enumerate(self.constraint_info):
            # Relaxable flag (1.0 if constraint can be relaxed)
            c_raw[i, f_base + 0] = 1.0 if info.relaxable else 0.0

            # Normalized relaxation cost (replace inf with large finite value)
            cost = info.relaxation_cost if math.isfinite(info.relaxation_cost) else 1e6
            c_raw[i, f_base + 1] = cost / max(1.0, self.b.abs().max().item())

            # Constraint tightness (0=loose, 1=tight)
            c_raw[i, f_base + 2] = info.tightness

            # Constraint type one-hot (hard=0, soft=1, preference=2)
            type_onehot = torch.zeros(3, dtype=torch.float32)
            if info.constraint_type == ConstraintType.HARD:
                type_onehot[0] = 1.0
            elif info.constraint_type == ConstraintType.SOFT:
                type_onehot[1] = 1.0
            else:  # preference
                type_onehot[2] = 1.0
            c_raw[i, f_base + 3:f_base + 6] = type_onehot

        return v_raw, c_raw

    def _assemble_observation(self, v_feat: torch.Tensor, c_feat: torch.Tensor) -> dict:
        """Assemble observation with constraint relaxability metadata.

        Returns observation dict compatible with existing NCO env/model, plus
        additional fields for constraint relaxation actions.
        
        Note: We don't call super() because c_feat has different feature dim than v_feat.
        Instead, we replicate the parent's logic with padding to match feature dims.
        """
        f_v = v_feat.shape[1]  # variable feature dim (e.g., 16)
        f_c = c_feat.shape[1]  # constraint feature dim (e.g., 22)
        f_node = max(f_v, f_c)  # use the larger dimension
        
        # Pad v_feat to match c_feat feature dim
        if f_v < f_c:
            v_padded = torch.zeros(v_feat.shape[0], f_c, dtype=v_feat.dtype, device=v_feat.device)
            v_padded[:, :f_v] = v_feat
            v_feat = v_padded
        # Pad c_feat to match v_feat feature dim
        elif f_c < f_v:
            c_padded = torch.zeros(c_feat.shape[0], f_v, dtype=c_feat.dtype, device=c_feat.device)
            c_padded[:, :f_c] = c_feat
            c_feat = c_padded
        
        total_nodes = self.n + self.m

        # Combined node features (order: variables first, then constraints)
        nodes = torch.cat([v_feat, c_feat], dim=0)  # [total_nodes, f_node]

        # Pointer mask: True for feasible targets
        if self.config.pointer_target == "variables":
            mask = torch.zeros(total_nodes, dtype=torch.bool)
            mask[: self.n] = True  # can only pointer at variables
        elif self.config.pointer_target == "constraints":
            mask = torch.zeros(total_nodes, dtype=torch.bool)
            mask[self.n :] = True
        else:  # "all"
            mask = torch.ones(total_nodes, dtype=torch.bool)

        # Global context: summary statistics of the problem
        f_global = self.config.global_feature_dim
        glob = torch.zeros(f_global, dtype=torch.float32)
        f = 0

        # Problem size
        if f < f_global:
            import math
            glob[f] = math.log(self.n + 1)  # log(n)
            f += 1
        if f < f_global:
            glob[f] = math.log(self.m + 1)  # log(m)
            f += 1

        # Constraint statistics
        if f < f_global:
            glob[f] = self.b.mean()
            f += 1
        if f < f_global:
            glob[f] = self.b.std()
            f += 1
        if f < f_global:
            # Constraint tightness: fraction of constraints satisfied at x=0
            slack = self.b - self.A.abs().sum(dim=1)
            glob[f] = (slack >= -1e-6).float().mean()
            f += 1

        # Objective statistics
        if f < f_global and self.c.numel() > 0:
            if f < f_global:
                glob[f] = self.c.mean()
                f += 1
            if f < f_global:
                glob[f] = self.c.std()
                f += 1

        # Add constraint relaxability features to global context
        if f < f_global:
            relaxable_count = sum(1 for info in self.constraint_info if info.relaxable)
            glob[f] = relaxable_count / max(1, self.m)
            f += 1
        if f < f_global:
            avg_cost = sum(info.relaxation_cost for info in self.constraint_info) / max(1, self.m)
            glob[f] = avg_cost / max(1.0, self.b.abs().max().item())
            f += 1
        if f + 3 <= f_global:
            hard_count = sum(1 for info in self.constraint_info if info.constraint_type == ConstraintType.HARD)
            soft_count = sum(1 for info in self.constraint_info if info.constraint_type == ConstraintType.SOFT)
            pref_count = sum(1 for info in self.constraint_info if info.constraint_type == ConstraintType.PREFERENCE)
            glob[f:f+3] = torch.tensor([hard_count, soft_count, pref_count], dtype=torch.float32) / max(1, self.m)
            f += 3

        # Task draw: variable cost coefficients (used as "weight" in pointer decode)
        task_draw = self.c.unsqueeze(1) if self.c.numel() > 1 else self.c  # [n] or [n, 1]

        # Budget: constraint RHS sum (used as stopping criterion)
        budget = self.b.abs().sum()

        obs = {
            "nodes": nodes.unsqueeze(0),  # [1, total_nodes, f_node]
            "mask": mask.unsqueeze(0),  # [1, total_nodes]
            "context": glob.unsqueeze(0),  # [1, f_global]
            "task_draw": task_draw.unsqueeze(1) if task_draw.ndim == 1 else task_draw,
            "budget": budget.unsqueeze(0).unsqueeze(0),  # [1, 1]
            # Legacy keys for backward compat
            "task_static": v_feat.unsqueeze(0),  # [1, n, f_v]
            "task_dynamic": torch.zeros(1, self.n, 0, dtype=torch.float32),
            "glob": glob.unsqueeze(0),
            "candidate": mask.unsqueeze(0),
            "feas_start": mask.unsqueeze(0),
        }

        # Add constraint relaxability metadata for action space
        obs["constraint_relaxable"] = torch.tensor(
            [info.relaxable for info in self.constraint_info], dtype=torch.bool
        ).unsqueeze(0)  # [1, m]
        obs["constraint_relaxation_cost"] = torch.tensor(
            [info.relaxation_cost for info in self.constraint_info], dtype=torch.float32
        ).unsqueeze(0)  # [1, m]

        return obs


def build_constraint_aware_graph(
    A: torch.Tensor | sp.spmatrix | list[list[float]],
    b: torch.Tensor | list[float],
    c: torch.Tensor | list[float],
    bounds: Optional[list[tuple[float, float]]] = None,
    constraint_info: Optional[list[ConstraintInfo]] = None,
    **kwargs,
) -> MILPSolution:
    """One-shot constraint-aware MILP-to-graph converter.

    Args:
        A: Constraint matrix [m, n]
        b: RHS vector [m]
        c: Objective coefficients [n]
        bounds: Variable bounds [(lo, hi), ...] [n]
        constraint_info: List of ConstraintInfo for each constraint [m]
        **kwargs: Passed to GraphConfig (node_feature_dim, global_feature_dim, etc.)

    Returns:
        MILPSolution with observation dict including constraint relaxability metadata
    """
    config = GraphConfig(**kwargs)
    builder = ConstraintAwareGraphBuilder(
        A, b, c, bounds=bounds, config=config, constraint_info=constraint_info
    )
    return builder.build()
