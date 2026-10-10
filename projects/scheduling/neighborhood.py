"""Coefficient-aware bipartite GNN for selecting ONTS repair neighborhoods."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .milp import ONTSMilp


@dataclass
class MilpGraph:
    variables: torch.Tensor
    constraints: torch.Tensor
    row_index: torch.Tensor
    col_index: torch.Tensor
    coefficients: torch.Tensor
    n_x: int

    def to(self, device: str | torch.device) -> MilpGraph:
        return MilpGraph(
            self.variables.to(device),
            self.constraints.to(device),
            self.row_index.to(device),
            self.col_index.to(device),
            self.coefficients.to(device),
            self.n_x,
        )


class GraphBuilder:
    """All x/phi/SoC variables and all actual sparse MILP rows, not summaries."""

    def __init__(self, problem: ONTSMilp):
        self.problem = problem
        A = problem.A.tocsr()
        scales = np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(), 1e-9)
        self.A = A.multiply((1 / scales)[:, None]).tocsr()
        coo = self.A.tocoo()
        self.rows = torch.from_numpy(coo.row.astype(np.int64))
        self.cols = torch.from_numpy(coo.col.astype(np.int64))
        self.coefficients = torch.from_numpy(coo.data.astype(np.float32))
        self.lower = problem.row_lower / scales
        self.upper = problem.row_upper / scales

    def build(self, incumbent: torch.Tensor) -> MilpGraph:
        return self.build_values(self.problem.assignment(incumbent))

    def build_values(self, values: np.ndarray) -> MilpGraph:
        """Features for any assignment, including an infeasible empty one."""
        p = self.problem
        activity = self.A @ values

        # Bounded transforms preserve slack sign without arbitrary clipping.
        def squash(a):
            return a / (1 + np.abs(a))

        lower = np.where(np.isfinite(self.lower), self.lower, 0)
        upper = np.where(np.isfinite(self.upper), self.upper, 0)
        constraints = np.stack(
            [
                squash(lower),
                squash(upper),
                np.isfinite(self.lower),
                np.isfinite(self.upper),
                squash(activity),
                squash(activity - lower),
                squash(upper - activity),
                (lower == upper) & np.isfinite(self.lower) & np.isfinite(self.upper),
            ],
            axis=1,
        )
        n, T = p.n_x, p.T
        variable = np.zeros((p.n_vars, 12), dtype=np.float32)
        variable[:, 0] = values
        variable[:, 1] = -p.c / max(np.max(np.abs(p.c)), 1)
        variable[:, 2], variable[:, 3] = p.lower, p.upper
        variable[:n, 4], variable[n : 2 * n, 5], variable[2 * n :, 6] = 1, 1, 1
        time = np.arange(T) / max(T - 1, 1)
        variable[:n, 7] = np.tile(time, p.J)
        variable[n : 2 * n, 7] = np.tile(time, p.J)
        variable[2 * n :, 7] = time
        solar = p.instance.recurso_p.cpu().double().numpy()
        solar_scale = max(float(np.max(solar)), 1)
        variable[:n, 8] = np.tile(solar / solar_scale, p.J)
        # Future average solar provides information beyond instantaneous capacity.
        future = np.cumsum(solar[::-1])[::-1] / np.arange(T, 0, -1)
        variable[:n, 9] = np.tile(future / solar_scale, p.J)
        variable[:n, 10] = np.repeat(p.instance.uso_p.cpu().numpy() / solar_scale, T)
        variable[:n, 11] = np.tile(values[2 * n :] - p.config.limite_inferior, p.J)
        return MilpGraph(
            torch.from_numpy(variable),
            torch.from_numpy(constraints.astype(np.float32)),
            self.rows,
            self.cols,
            self.coefficients,
            n,
        )


class NeighborhoodPolicy(nn.Module):
    """Bidirectional coefficient-aware message passing; one release logit per x."""

    def __init__(self, hidden: int = 64, layers: int = 2):
        super().__init__()
        if hidden <= 0 or layers <= 0:
            raise ValueError("hidden and layers must be positive")
        self.hidden, self.layers = hidden, layers
        self.variable_input = nn.Sequential(nn.Linear(12, hidden), nn.ReLU())
        self.constraint_input = nn.Sequential(nn.Linear(8, hidden), nn.ReLU())
        self.to_constraint = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU())
                for _ in range(layers)
            ]
        )
        self.to_variable = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(hidden + 2, hidden), nn.ReLU())
                for _ in range(layers)
            ]
        )
        self.constraint_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden)
                )
                for _ in range(layers)
            ]
        )
        self.variable_update = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden)
                )
                for _ in range(layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, graph: MilpGraph) -> torch.Tensor:
        v, c = (
            self.variable_input(graph.variables),
            self.constraint_input(graph.constraints),
        )
        rows, cols = graph.row_index, graph.col_index
        edge = torch.stack([graph.coefficients, graph.coefficients.abs()], dim=1)
        row_degree = torch.bincount(rows, minlength=len(c)).clamp(min=1).unsqueeze(1)
        col_degree = torch.bincount(cols, minlength=len(v)).clamp(min=1).unsqueeze(1)
        for vc, cv, cu, vu in zip(
            self.to_constraint,
            self.to_variable,
            self.constraint_update,
            self.variable_update,
        ):
            messages = vc(torch.cat([v[cols], edge], dim=1))
            aggregate = torch.zeros_like(c).index_add(0, rows, messages) / row_degree
            c = cu(torch.cat([c, aggregate], dim=1))
            messages = cv(torch.cat([c[rows], edge], dim=1))
            aggregate = torch.zeros_like(v).index_add(0, cols, messages) / col_degree
            v = vu(torch.cat([v, aggregate], dim=1))
        return self.head(v[: graph.n_x]).squeeze(1)

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        torch.save(
            {
                "format": "onts-neighborhood-v1",
                "hidden": self.hidden,
                "layers": self.layers,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> NeighborhoodPolicy:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != "onts-neighborhood-v1":
            raise ValueError("Unsupported neighborhood checkpoint")
        model = cls(checkpoint["hidden"], checkpoint["layers"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.training_metadata = checkpoint.get("metadata", {})
        model.eval()
        return model
