"""Outcome-ranked, semantic ONTS neighborhoods with sparse graph attention.

Inspired by recent contrastive LNS, preference learning and semantic MILP
representations. This is an ONTS adaptation, not a reproduction of those papers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .milp import ONTSMilp
from .neighborhood import GraphBuilder

OPERATORS = ("random", "utility", "run", "energy")
ROW_FAMILIES = (
    "startup_lower",
    "startup_on",
    "startup_previous",
    "startup_count",
    "min_periodo_job",
    "max_periodo_job",
    "minimum_run",
    "maximum_run",
    "power",
    "energy",
)


@dataclass
class SearchState:
    last_mask: torch.Tensor
    changed: torch.Tensor
    steps: int = 0
    stagnation: int = 0
    last_gain: float = 0.0
    last_runtime: float = 0.0

    @classmethod
    def empty(cls, J: int, T: int):
        return cls(
            torch.zeros(J, T, dtype=torch.bool), torch.zeros(J, T, dtype=torch.bool)
        )

    def advance(self, mask, previous, current, gain, runtime):
        self.last_mask = mask.detach().cpu().clone()
        self.changed = previous.detach().cpu().bool() != current.detach().cpu().bool()
        self.steps += 1
        self.stagnation = 0 if gain > 1e-6 else self.stagnation + 1
        self.last_gain, self.last_runtime = float(gain), float(runtime)


@dataclass
class Candidate:
    mask: torch.Tensor
    operator: str
    features: torch.Tensor


def make_candidates(
    problem: ONTSMilp,
    incumbent: torch.Tensor,
    state: SearchState,
    rng: np.random.Generator,
    *,
    fraction: float = 0.35,
    count: int = 8,
    block_size: int = 4,
) -> list[Candidate]:
    """Build bounded, diverse masks; run moves preserve complete temporal units.

    A unit larger than the requested cell budget is skipped, not truncated.
    Only x variables are masked; phi and SoC remain free in every MILP repair.
    """
    if (
        not math.isfinite(fraction)
        or not 0 < fraction <= 1
        or count < 4
        or block_size < 1
    ):
        raise ValueError("Expected fraction in (0,1], count >= 4 and block_size >= 1")
    x = incumbent.detach().cpu().bool().numpy()
    if x.shape != (problem.J, problem.T):
        raise ValueError("Incumbent must have shape [J,T]")
    if (
        state.last_mask.shape != incumbent.shape
        or state.changed.shape != incumbent.shape
    ):
        raise ValueError("Search history shape differs from incumbent")
    J, T = x.shape
    n, limit = J * T, max(1, math.ceil(J * T * fraction))
    priority = problem.instance.priority.cpu().double().numpy()
    draw = problem.instance.uso_p.cpu().double().numpy()
    solar = problem.instance.recurso_p.cpu().double().numpy()
    utility = priority / np.maximum(draw, 0.01)
    usage = draw @ x
    soc = problem.assignment(incumbent)[2 * n :]
    pressure = (
        usage
        / np.maximum(solar + problem.config.bat_usage * problem.config.v_bat, 1e-6)
        + 1
        - soc
    )
    legal = problem.upper[:n].reshape(J, T).astype(bool)
    masks = []
    seen = set()
    runs = []
    for row in x:
        changes = np.flatnonzero(np.diff(np.r_[False, row, False]))
        runs.append(list(zip(changes[::2], changes[1::2])))
    for index in range(count):
        operator = OPERATORS[index % len(OPERATORS)]
        noise = rng.gumbel(size=(J, T))
        if operator == "random":
            scores = noise
        elif operator == "energy":
            scores = np.broadcast_to(pressure, (J, T)).copy() + 0.2 * noise
        else:
            scores = utility[:, None] * (~x) + noise
        scores[~legal] = -np.inf
        mask = np.zeros_like(x)
        # Sort run/relocation units, rather than rank isolated binary decisions.
        units = []
        if operator == "run":
            for j in range(J):
                for start, end in runs[j]:
                    width = max(end - start, int(problem.instance.min_cpu_time[j]))
                    lo, hi = max(0, start - 1), min(T, end + width + 1)
                    units.append((float(scores[j, start:end].mean()), j, lo, hi))
        for flat in np.argsort(-scores.ravel(), kind="stable"):
            j, t = divmod(int(flat), T)
            if not legal[j, t]:
                continue
            width = (
                max(block_size, int(problem.instance.min_cpu_time[j]))
                if operator == "run"
                else min(block_size, T)
            )
            lo = max(0, min(t - width // 2, T - width))
            units.append((float(scores[j, t]), j, lo, min(T, lo + width)))
        for _, j, lo, hi in sorted(units, key=lambda u: -u[0]):
            if operator == "run":
                # Include every active segment touched by a relocation region.
                # Close transitively if expanding reaches a neighboring run.
                previous = None
                while previous != (lo, hi):
                    previous = (lo, hi)
                    for start, end in runs[j]:
                        if start < hi and end > lo:
                            lo, hi = (
                                min(lo, max(0, start - 1)),
                                max(hi, min(T, end + 1)),
                            )
            unit = legal[j, lo:hi] & ~mask[j, lo:hi]
            added = int(unit.sum())
            if added and int(mask.sum()) + added <= limit:
                mask[j, lo:hi] |= legal[j, lo:hi]
            if mask.sum() == limit:
                break
        # If no whole run fits, release only inactive cells for insertion;
        # never silently split a run in this operator's fallback.
        if not mask.any():
            available = np.flatnonzero(
                (legal & ~x if operator == "run" else legal).ravel()
            )
            if len(available):
                mask.ravel()[
                    rng.choice(
                        available, size=min(limit, len(available)), replace=False
                    )
                ] = True
        # Distinct random proposals when tied rankings yield duplicate masks.
        key = mask.tobytes()
        if key in seen and operator == "random":
            available = np.flatnonzero(legal.ravel())
            for _ in range(4):
                mask[:] = False
                if len(available):
                    mask.ravel()[
                        rng.choice(
                            available, size=min(limit, len(available)), replace=False
                        )
                    ] = True
                key = mask.tobytes()
                if key not in seen:
                    break
        seen.add(key)
        chosen = max(int(mask.sum()), 1)
        onehot = [float(operator == name) for name in OPERATORS]
        features = [
            *onehot,
            mask.sum() / n,
            (mask & x).sum() / chosen,
            np.any(mask, axis=1).mean(),
            np.any(mask, axis=0).mean(),
            (mask & state.last_mask.numpy()).sum() / chosen,
            float(pressure[np.any(mask, axis=0)].mean()) if mask.any() else 0.0,
        ]
        masks.append(
            Candidate(
                torch.from_numpy(mask),
                operator,
                torch.tensor(features, dtype=torch.float32),
            )
        )
    return masks


class RankingGraphBuilder(GraphBuilder):
    def __init__(self, problem: ONTSMilp):
        super().__init__(problem)
        self.row_types = torch.tensor(
            [ROW_FAMILIES.index(name.split(":", 1)[0]) for name in problem.row_names],
            dtype=torch.long,
        )

    def build_state(self, incumbent, state, *, remaining_fraction=1.0, repair_time=1.0):
        if not math.isfinite(repair_time) or repair_time <= 0:
            raise ValueError("repair_time must be positive and finite")
        graph = self.build(incumbent)
        p = self.problem
        extras = torch.zeros(p.n_vars, 4)
        x = incumbent.detach().cpu().bool()
        starts = x & ~torch.cat(
            [torch.zeros(p.J, 1, dtype=torch.bool), x[:, :-1]], dim=1
        )
        extras[: p.n_x, 0] = state.last_mask.flatten()
        extras[: p.n_x, 1] = state.changed.flatten()
        extras[: p.n_x, 2] = (
            starts.cumsum(1) / p.instance.max_statup.cpu().clamp(min=1)[:, None]
        ).flatten()
        ages = torch.zeros_like(x, dtype=torch.float32)
        for t in range(p.T):
            ages[:, t] = x[:, t] * (1 + (ages[:, t - 1] if t else 0))
        extras[: p.n_x, 3] = (
            ages / p.instance.max_cpu_time.cpu().clamp(min=1)[:, None]
        ).flatten()
        scale = max(float(p.instance.priority.sum()) * p.T, 1.0)
        context = torch.tensor(
            [
                np.clip(remaining_fraction, 0, 1),
                state.stagnation / (1 + state.stagnation),
                state.steps / (1 + state.steps),
                state.last_gain / scale,
                math.log1p(state.last_runtime / repair_time),
                math.log1p(repair_time),
            ],
            dtype=torch.float32,
        )
        return graph, extras, self.row_types, context


class SparseAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.message = nn.Linear(hidden + 2, hidden)
        self.attention = nn.Linear(2 * hidden + 2, 1)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden)
        )

    def forward(self, source, target, source_index, target_index, edge):
        logits = self.attention(
            torch.cat([source[source_index], target[target_index], edge], 1)
        ).squeeze(1)
        maximum = logits.new_full((len(target),), -torch.inf)
        maximum.scatter_reduce_(
            0, target_index, logits, reduce="amax", include_self=True
        )
        weights = torch.exp(logits - maximum[target_index])
        denominator = (
            weights.new_zeros(len(target))
            .index_add(0, target_index, weights)
            .clamp(min=1e-9)
        )
        weights = weights / denominator[target_index]
        messages = (
            self.message(torch.cat([source[source_index], edge], 1)) * weights[:, None]
        )
        aggregate = target.new_zeros(target.shape).index_add(0, target_index, messages)
        return target + self.update(torch.cat([target, aggregate], 1))


class NeighborhoodRanker(nn.Module):
    """One sparse encoding, many candidate-mask scores; no repair at scoring time."""

    def __init__(self, hidden=32, layers=2, *, semantics=True, memory=True):
        super().__init__()
        if hidden < 1 or layers < 1:
            raise ValueError("hidden and layers must be positive")
        self.hidden, self.layers, self.semantics, self.memory = (
            hidden,
            layers,
            semantics,
            memory,
        )
        self.variable_input = nn.Linear(16, hidden)
        self.constraint_input = nn.Linear(8, hidden)
        self.category = nn.Embedding(len(ROW_FAMILIES), hidden)
        self.vc = nn.ModuleList([SparseAttention(hidden) for _ in range(layers)])
        self.cv = nn.ModuleList([SparseAttention(hidden) for _ in range(layers)])
        self.global_input = nn.Sequential(nn.Linear(2 * hidden + 6, hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 10, hidden), nn.GELU(), nn.Linear(hidden, 2)
        )

    def forward(self, state_graph, candidates: list[Candidate]):
        if not candidates:
            raise ValueError("Provide at least one candidate")
        graph, extras, categories, context = state_graph
        device = next(self.parameters()).device
        graph = graph.to(device)
        extras, categories, context = (
            extras.to(device),
            categories.to(device),
            context.to(device),
        )
        extras = extras.clone()
        context = context.clone()
        if not self.memory:
            extras[:, :2] = 0
            context[1:5] = 0
        v = F.gelu(self.variable_input(torch.cat([graph.variables, extras], 1)))
        c = self.constraint_input(graph.constraints)
        if self.semantics:
            c = c + self.category(categories)
        c = F.gelu(c)
        edge = torch.stack([graph.coefficients, graph.coefficients.abs()], 1)
        for vc, cv in zip(self.vc, self.cv):
            c = vc(v, c, graph.col_index, graph.row_index, edge)
            v = cv(c, v, graph.row_index, graph.col_index, edge)
        global_state = self.global_input(
            torch.cat([v[: graph.n_x].mean(0), c.mean(0), context])
        )
        masks = torch.stack([candidate.mask.flatten() for candidate in candidates]).to(
            device=device, dtype=v.dtype
        )
        if masks.shape[1] != graph.n_x:
            raise ValueError("Candidate mask shape differs from execution variables")
        pooled = masks @ v[: graph.n_x] / masks.sum(1, keepdim=True).clamp(min=1)
        features = torch.stack([candidate.features for candidate in candidates]).to(
            device
        )
        if not self.memory:
            features = features.clone()
            features[:, 8] = 0
        output = self.head(
            torch.cat([pooled, global_state.expand(len(candidates), -1), features], 1)
        )
        return output[:, 0], F.softplus(output[:, 1])

    def save(self, path: str | Path, *, metadata=None):
        torch.save(
            {
                "format": "onts-ranker-v1",
                "hidden": self.hidden,
                "layers": self.layers,
                "semantics": self.semantics,
                "memory": self.memory,
                "row_families": ROW_FAMILIES,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device="cpu"):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if (
            checkpoint.get("format") != "onts-ranker-v1"
            or tuple(checkpoint.get("row_families", ())) != ROW_FAMILIES
        ):
            raise ValueError("Unsupported ranker checkpoint")
        policy = cls(
            checkpoint["hidden"],
            checkpoint["layers"],
            semantics=checkpoint["semantics"],
            memory=checkpoint["memory"],
        ).to(device)
        policy.load_state_dict(checkpoint["state_dict"])
        policy.training_metadata = checkpoint.get("metadata", {})
        policy.eval()
        return policy
