"""Convert MILP/LP (A, b, c) to bipartite directed graph with message passing.

Standard form:
    min/max   c^T x
    s.t.      Ax ≤ b
              x ∈ X (bounds, integrality, etc.)

Graph construction
------------------
Bipartite directed graph G = (V ∪ C, E) where:
    V = {v_1, ..., v_n}   variable nodes
    C = {c_1, ..., c_m}   constraint nodes
    E = {(v_j, c_i) | A_ij ≠ 0}  directed edges var→constraint

Edge directions encode information flow:
    v → c : variables report values to constraints
    c → v : constraints propagate dual/shadow prices back to variables

Message passing protocols
--------------------------
Each protocol defines how node representations are updated.
Composable via compose_messages().

Output
------
Produces observation dict compatible with existing NCO env/model:
    task_static   [B, N_nodes, f_static]
    task_dynamic  [B, N_nodes, f_dynamic]
    glob          [B, f_global]
    mask          [B, N_nodes]     feasible pointer targets (variables only)
    candidate     [B, N_nodes]     alias for mask
    feas_start    [B, N_nodes]
    task_draw     [B, N_nodes]
    budget        [B]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch
import scipy.sparse as sp


# ─── Graph topology ────────────────────────────────────────────────────────


class GraphTopology(str, Enum):
    """Bipartite directed graph layouts."""

    # Default: variables → constraints, pointer decodes over variables only
    BIPARTITE_VAR_TO_CONST = "bipartite_var_to_const"
    # Flip: constraints → variables, pointer decodes over constraints
    BIPARTITE_CONST_TO_VAR = "bipartite_const_to_var"
    # Full bipartite with all edges (dense connectivity for small problems)
    BIPARTITE_FULL = "bipartite_full"
    # Variable-only: constraints collapsed into global context
    #   (compatible with existing single-sequence models)
    VARIABLE_ONLY = "variable_only"
    # Constraint-only: variables collapsed into global context
    CONSTRAINT_ONLY = "constraint_only"


class MessageProtocol(str, Enum):
    """Message passing strategies between node types."""

    # Aggregate neighbors, send weighted sum
    MEAN = "mean"
    # Weight by edge values (coefficient magnitude)
    EDGE_WEIGHTED = "edge_weighted"
    # Max pooling over neighbors
    MAX = "max"
    # Sum over neighbors
    SUM = "sum"
    # Attention-based (learned weights)
    ATTENTION = "attention"


# ─── Feature dimensions ────────────────────────────────────────────────────


@dataclass
class GraphConfig:
    """Configuration for graph construction and message passing."""

    # Topology: how to arrange nodes and edges
    topology: GraphTopology = GraphTopology.BIPARTITE_VAR_TO_CONST

    # Message passing
    message_protocol: MessageProtocol = MessageProtocol.EDGE_WEIGHTED
    num_message_steps: int = 2  # layers of message passing before encoding
    edge_feature_dim: int = 4  # features per edge

    # Feature dimensions (set to None for auto)
    node_feature_dim: int = 16  # hidden dim for node embeddings
    global_feature_dim: int = 32  # global context dimension

    # Normalization
    auto_normalize: bool = True  # normalize A, b, c features

    # Pointer target mask
    pointer_target: str = "variables"  # "variables", "constraints", or "all"

    # Special node features
    include_cost_features: bool = True  # include c in variable features
    include_slack_features: bool = True  # include slack in constraint features
    include_positional_features: bool = True  # positional encoding

    # Edge features
    include_edge_type: bool = True  # inequality/equality/type features


# ─── Edge and node types ───────────────────────────────────────────────────


@dataclass
class EdgeInfo:
    """Single edge in the bipartite graph."""

    src_node: int  # source node index (in combined node list)
    dst_node: int  # destination node index
    coefficient: float  # A[i, j] value
    edge_type: int = 0  # edge type (0=ineq, 1=eq, 2=geq, etc.)


# ─── Graph builder ─────────────────────────────────────────────────────────


class MILPGraphBuilder:
    """Build bipartite directed graph from MILP matrices A, b, c.

    Args:
        A: Constraint matrix [m, n] (m constraints, n variables)
        b: Right-hand side [m]
        c: Objective coefficients [n]
        bounds: Variable bounds, list of (low, high) tuples [n]
        config: Graph construction configuration
        constraint_types: 'ineq', 'eq', or 'geq' for each constraint [m]
    """

    def __init__(
        self,
        A: torch.Tensor | sp.spmatrix | list[list[float]],
        b: torch.Tensor | list[float],
        c: torch.Tensor | list[float],
        bounds: Optional[list[tuple[float, float]]] = None,
        config: Optional[GraphConfig] = None,
        constraint_types: Optional[list[str]] = None,
    ):
        self.A, self.b, self.c = self._to_tensors(A, b, c)

        m, n = self.A.shape  # m constraints, n variables
        self.m = m
        self.n = n

        # Bounds: default to [0, +inf)
        if bounds is None:
            bounds = [(0.0, float("inf"))] * n
        self.bounds = torch.tensor(bounds, dtype=torch.float32)  # [n, 2]
        self._n_vars = n  # store for later use

        # Constraint types: default all inequality (≤)
        if constraint_types is None:
            constraint_types = ["ineq"] * m
        self.constraint_types = constraint_types

        self.config = config or GraphConfig()
        self._edges: list[EdgeInfo] = []
        self._adj_list: dict[int, list[int]] = {}  # src → [dst, ...]
        self._rev_adj: dict[int, list[int]] = {}  # dst → [src, ...]
        self._edge_features: dict[tuple[int, int], torch.Tensor] = {}

    @staticmethod
    def _to_tensors(A, b, c) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert inputs to float32 tensors."""
        if sp.issparse(A):
            A = torch.from_numpy(A.toarray()).float()
        elif isinstance(A, list):
            A = torch.tensor(A, dtype=torch.float32)
        else:
            A = torch.as_tensor(A, dtype=torch.float32)

        if isinstance(b, list):
            b = torch.tensor(b, dtype=torch.float32)
        else:
            b = torch.as_tensor(b, dtype=torch.float32)

        if isinstance(c, list):
            c = torch.tensor(c, dtype=torch.float32)
        else:
            c = torch.as_tensor(c, dtype=torch.float32)

        return A, b.squeeze(-1) if b.ndim > 1 else b, c.squeeze(-1) if c.ndim > 1 else c

    def build(self) -> MILPSolution:
        """Construct the full graph and return MILPSolution.

        Steps:
            1. Parse sparsity pattern of A → edges
            2. Build adjacency lists
            3. Compute edge features
            4. Apply message passing to enrich node representations
            5. Assemble observation dict
        """
        self._extract_edges()
        self._build_adjacency()
        self._compute_edge_features()

        if self.config.auto_normalize:
            self._normalize()

        # Node features after message passing
        v_features, c_features = self._build_node_features()

        # Compose into observation
        obs = self._assemble_observation(v_features, c_features)
        return MILPSolution(
            obs=obs,
            config=self.config,
            edges=self._edges,
            adj_list=self._adj_list,
            n=self.n,
        )

    def _extract_edges(self):
        """Extract directed edges from A's sparsity pattern.

        For topology BIARTITE_VAR_TO_CONST: edges go v_j → c_i where A_ij ≠ 0
        For topology BIARTITE_CONST_TO_VAR: edges go c_i → v_j where A_ij ≠ 0
        For topology BIARTITE_FULL: both directions
        """
        self._edges = []
        top = self.config.topology

        for i in range(self.m):  # constraint index
            for j in range(self.n):  # variable index
                if abs(self.A[i, j].item()) < 1e-12:
                    continue

                coeff = self.A[i, j].item()
                edge_type = self._constraint_type_to_int(self.constraint_types[i])

                if top in (
                    GraphTopology.BIPARTITE_VAR_TO_CONST,
                    GraphTopology.BIPARTITE_FULL,
                ):
                    self._edges.append(
                        EdgeInfo(
                            src_node=j,
                            dst_node=self.n + i,
                            coefficient=coeff,
                            edge_type=edge_type,
                        )
                    )
                if top in (
                    GraphTopology.BIPARTITE_CONST_TO_VAR,
                    GraphTopology.BIPARTITE_FULL,
                ):
                    self._edges.append(
                        EdgeInfo(
                            src_node=self.n + i,
                            dst_node=j,
                            coefficient=coeff,
                            edge_type=edge_type,
                        )
                    )

    @staticmethod
    def _constraint_type_to_int(ctype: str) -> int:
        return {"ineq": 0, "eq": 1, "geq": 2}.get(ctype, 0)

    def _build_adjacency(self):
        """Build forward and reverse adjacency lists."""
        self._adj_list.clear()
        self._rev_adj.clear()

        for edge in self._edges:
            self._adj_list.setdefault(edge.src_node, []).append(edge.dst_node)
            self._rev_adj.setdefault(edge.dst_node, []).append(edge.src_node)

    def _compute_edge_features(self):
        """Compute per-edge feature vectors.

        Features: [coefficient, |coefficient|, sign, edge_type_onehot (padded)]
        """
        dim = self.config.edge_feature_dim
        for edge in self._edges:
            feat = torch.zeros(dim, dtype=torch.float32)
            feat[0] = edge.coefficient
            feat[1] = abs(edge.coefficient)
            feat[2] = 1.0 if edge.coefficient > 0 else -1.0  # sign
            if self.config.include_edge_type and dim >= 6:
                onehot = torch.zeros(3, dtype=torch.float32)
                onehot[edge.edge_type] = 1.0
                feat[3:6] = onehot
            self._edge_features[(edge.src_node, edge.dst_node)] = feat

    def _normalize(self):
        """L2-normalize A, b, c features to improve training stability."""
        A_norm = self.A.norm(p=2)
        if A_norm > 1e-8:
            self.A = self.A / A_norm
        b_norm = self.b.norm(p=2)
        if b_norm > 1e-8:
            self.b = self.b / b_norm
        c_norm = self.c.norm(p=2)
        if c_norm > 1e-8:
            self.c = self.c / c_norm

        # Normalize bounds
        for j in range(self.n):
            low, high = self.bounds[j]
            if abs(high - low) > 1e-8:
                self.bounds[j] = torch.tensor([low, high]) / abs(high - low)

    def _build_node_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build raw (pre-message-passing) node features.

        Returns:
            v_raw: [n, f_node] variable node features
            c_raw: [m, f_node] constraint node features
        """
        f_node = self.config.node_feature_dim

        # --- Variable node features ---
        v_raw = torch.zeros(self.n, f_node, dtype=torch.float32)
        f = 0

        if self.config.include_cost_features:
            v_raw[:, f] = self.c[0] if self.c.numel() == 1 else self.c
            f = min(f + 1, f_node)

        # Bounds
        if f < f_node:
            v_raw[:, f] = self.bounds[:, 0]  # lower bound
            f += 1
        if f < f_node:
            v_raw[:, f] = self.bounds[:, 1]  # upper bound
            f += 1

        # Positional encoding
        if self.config.include_positional_features and f + 2 <= f_node:
            for k in range(2, min(f + 2, f_node)):
                j_range = torch.arange(1, self.n + 1, dtype=torch.float32)
                freq = 10000 ** (k / f_node)
                v_raw[:, k] = torch.sin(j_range / freq)

        # --- Constraint node features ---
        c_raw = torch.zeros(self.m, f_node, dtype=torch.float32)
        f = 0

        # RHS value
        if f < f_node:
            c_raw[:, f] = self.b
            f += 1

        # Constraint type one-hot
        if f + 3 <= f_node:
            for t in range(3):
                c_raw[:, f + t] = torch.tensor(
                    [
                        1.0 if self._constraint_type_to_int(ct) == t else 0.0
                        for ct in self.constraint_types
                    ],
                    dtype=torch.float32,
                )
            f += 3

        # Slack indicator (1 if constraint has zero slack at origin)
        if self.config.include_slack_features and f < f_node:
            c_raw[:, f] = (self.A.abs().sum(dim=1) >= self.b.abs() + 1e-6).float()

        # Constraint density (fraction of non-zero entries)
        if f + 1 <= f_node:
            c_raw[:, f] = (self.A.abs() > 1e-12).float().mean(dim=1)

        return v_raw, c_raw

    def _apply_message_passing(
        self, v_feat: torch.Tensor, c_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run message passing layers on the bipartite graph.

        Each step:
            1. Variables → Constraints: aggregate variable features at each constraint
            2. Constraints → Variables: aggregate constraint features at each variable

        Args:
            v_feat: [n, f_node] initial variable features
            c_feat: [m, f_node] initial constraint features

        Returns:
            v_updated: [n, f_node] after message passing
            c_updated: [m, f_node] after message passing
        """
        protocol = self.config.message_protocol
        steps = self.config.num_message_steps
        v = v_feat.clone()
        c = c_feat.clone()

        for step in range(steps):
            v, c = self._message_step(v, c, protocol)

        return v, c

    def _message_step(
        self, v: torch.Tensor, c: torch.Tensor, protocol: MessageProtocol
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Single message passing step."""
        n, m = self.n, self.m
        f = v.shape[1]
        device = v.device

        # --- Phase 1: v → c (variables send to constraints) ---
        c_msg = torch.zeros(m, f, device=device)
        for edge in self._edges:
            if edge.src_node < n:  # variable → constraint
                src_idx = edge.src_node
                dst_idx = edge.dst_node - n  # constraint index
                msg = self._compute_message(v[src_idx], c[dst_idx], edge, protocol)
                c_msg[dst_idx] += msg

        # --- Phase 2: c → v (constraints send to variables) ---
        v_msg = torch.zeros(n, f, device=device)
        for edge in self._edges:
            if edge.src_node >= n:  # constraint → variable
                src_idx = edge.src_node - n  # constraint index
                dst_idx = edge.src_node - n  # variable index
                msg = self._compute_message(c[src_idx], v[dst_idx], edge, protocol)
                v_msg[dst_idx] += msg

        # Update nodes (residual connection)
        v = v + v_msg
        c = c + c_msg

        return v, c

    def _compute_message(
        self,
        src_feat: torch.Tensor,
        dst_feat: torch.Tensor,
        edge: EdgeInfo,
        protocol: MessageProtocol,
    ) -> torch.Tensor:
        """Compute message from source to destination node."""
        feat = src_feat.clone()

        if protocol == MessageProtocol.EDGE_WEIGHTED:
            # Scale by edge coefficient
            feat = feat * edge.coefficient
        elif protocol == MessageProtocol.MAX:
            feat = feat.abs().max() * torch.sign(feat)
        elif protocol == MessageProtocol.ATTENTION:
            # Simple learned-style attention
            attn = torch.sigmoid(feat + dst_feat)
            feat = feat * attn

        return feat

    def _assemble_observation(self, v_feat: torch.Tensor, c_feat: torch.Tensor) -> dict:
        """Assemble node/edge features into env-compatible observation dict.

        Uses combined node list: [v_0, ..., v_{n-1}, c_0, ..., c_{m-1}]
        Pointer mask targets only variables (or as configured).
        """
        f_node = v_feat.shape[1]
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
        glob = self._compute_global_context()

        # Task draw: variable cost coefficients (used as "weight" in pointer decode)
        task_draw = (
            self.c.unsqueeze(1) if self.c.numel() > 1 else self.c
        )  # [n] or [n, 1]

        # Budget: constraint RHS sum (used as stopping criterion)
        budget = self.b.abs().sum()

        obs = {
            "nodes": nodes.unsqueeze(0),  # [1, total_nodes, f_node]
            "mask": mask.unsqueeze(0),  # [1, total_nodes]
            "context": glob.unsqueeze(0),  # [1, f_global]
            # Legacy keys for backward compat
            "task_static": v_feat.unsqueeze(0),  # [1, n, f_node]
            "task_dynamic": torch.zeros(1, self.n, 0, dtype=torch.float32),
            "glob": glob.unsqueeze(0),
            "candidate": mask.unsqueeze(0),
            "feas_start": mask.unsqueeze(0),
            "task_draw": task_draw.unsqueeze(0).unsqueeze(-1)
            if task_draw.ndim == 1
            else task_draw.unsqueeze(0),
            "budget": budget.unsqueeze(0).unsqueeze(0),  # [1, 1]
        }
        return obs
        return obs

    def _compute_global_context(self) -> torch.Tensor:
        """Compute global problem-level features."""
        f_global = self.config.global_feature_dim
        glob = torch.zeros(f_global, dtype=torch.float32)
        f = 0

        # Problem size
        if f < f_global:
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

        return glob

    def get_graph_structure(self) -> dict:
        """Return raw graph structure for inspection/debugging."""
        return {
            "num_variables": self.n,
            "num_constraints": self.m,
            "num_edges": len(self._edges),
            "topology": self.config.topology.value,
            "edges": [
                {
                    "src": e.src_node,
                    "dst": e.dst_node,
                    "coeff": e.coefficient,
                    "type": e.edge_type,
                }
                for e in self._edges
            ],
            "adjacency": {str(k): v for k, v in self._adj_list.items()},
        }


# ─── Solution container ────────────────────────────────────────────────────


@dataclass
class MILPSolution:
    """Result of MILP graph construction.

    Contains the observation dict and all graph metadata needed for
    downstream model training or inference.
    """

    obs: dict
    config: GraphConfig
    edges: list[EdgeInfo]
    adj_list: dict[int, list[int]]
    n: int = 0  # number of variables

    @property
    def num_variables(self) -> int:
        return (
            self.config.pointer_target == "variables"
            and len(set(e.src_node for e in self.edges if e.src_node < 0))
            or self.obs["nodes"].shape[1]
        )

    def get_node_features(self, node_type: str = "all") -> torch.Tensor:
        """Extract node feature tensor.

        Args:
            node_type: "variables", "constraints", or "all"
        """
        nodes = self.obs["nodes"]  # [1, N, f]
        if node_type == "variables":
            return nodes[:, : self.n]
        elif node_type == "constraints":
            return nodes[:, self.n :]
        return nodes

    def __repr__(self):
        return (
            f"MILPSolution(topology={self.config.topology.value}, "
            f"nodes={self.obs['nodes'].shape[1]}, "
            f"edges={len(self.edges)}, "
            f"protocol={self.config.message_protocol.value})"
        )


# ─── Convenience factory ───────────────────────────────────────────────────


def build_milp_graph(
    A: torch.Tensor | sp.spmatrix | list[list[float]],
    b: torch.Tensor | list[float],
    c: torch.Tensor | list[float],
    bounds: Optional[list[tuple[float, float]]] = None,
    constraint_types: Optional[list[str]] = None,
    topology: GraphTopology = GraphTopology.BIPARTITE_VAR_TO_CONST,
    message_protocol: MessageProtocol = MessageProtocol.EDGE_WEIGHTED,
    **kwargs,
) -> MILPSolution:
    """One-shot MILP-to-graph converter.

    Args:
        A: Constraint matrix [m, n]
        b: RHS vector [m]
        c: Objective coefficients [n]
        bounds: Variable bounds [(lo, hi), ...] [n]
        constraint_types: 'ineq'/'eq'/'geq' per constraint [m]
        topology: Graph layout
        message_protocol: Message passing strategy
        **kwargs: Passed to GraphConfig (node_feature_dim, global_feature_dim, etc.)

    Returns:
        MILPSolution with observation dict and graph metadata
    """
    config = GraphConfig(topology=topology, message_protocol=message_protocol, **kwargs)
    builder = MILPGraphBuilder(
        A, b, c, bounds=bounds, config=config, constraint_types=constraint_types
    )
    return builder.build()


# ─── Message protocol registry ─────────────────────────────────────────────

_MESSAGE_PROTOCOLS: dict[str, MessageProtocol] = {p.value: p for p in MessageProtocol}


def get_protocol(name: str) -> MessageProtocol:
    """Lookup protocol by string name."""
    return _MESSAGE_PROTOCOLS.get(name, MessageProtocol(name))


# ─── Composable message passing (like compose_constraints) ─────────────────


def compose_messages(*protocols: MessageProtocol) -> list[MessageProtocol]:
    """Chain multiple message protocols for multi-stage passing.

    Returns list of protocols to apply in sequence.
    """
    return list(protocols)
