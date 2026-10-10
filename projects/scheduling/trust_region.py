"""Predict-and-search for ONTS: contrastive GNN marginals and a trust region.

Follows the ConPaS recipe (Huang et al., ICML 2024) on our bipartite encoder:
predict per-cell probabilities from the instance alone, select k1 likely-on and
k0 likely-off cells, and solve the full MILP with the trust-region row
sum_{S1}(1-x) + sum_{S0} x <= delta. Training contrasts high-objective
schedules against low-objective and infeasible ones. This is the mechanism family
used by SatGNN; it is an ONTS adaptation, not a reproduction of either paper.
Trust-region solves are restricted: their bounds are never reported as global.
"""

from __future__ import annotations

import math
from copy import copy
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import LinearConstraint
from scipy.sparse import csr_matrix
from torch.nn import functional as F

from .hybrid import HybridResult
from .job_profile import battery_trace, solve_profile
from .local_search import weighted_local_search
from .milp import ONTSMilp
from .neighborhood import GraphBuilder, MilpGraph, NeighborhoodPolicy


def static_graph(problem: ONTSMilp, builder: GraphBuilder | None = None) -> MilpGraph:
    """Instance-only features: the empty schedule and its battery trajectory."""
    builder = builder or GraphBuilder(problem)
    values = np.zeros(problem.n_vars)
    values[2 * problem.n_x :] = battery_trace(problem, np.zeros(problem.T))
    return builder.build_values(values)


class SolutionPredictor(NeighborhoodPolicy):
    """Same encoder as the neighborhood policy; logits mean P(x_jt = 1)."""

    def save(self, path: str | Path, *, metadata: dict | None = None) -> None:
        torch.save(
            {
                "format": "onts-predictor-v1",
                "hidden": self.hidden,
                "layers": self.layers,
                "state_dict": self.state_dict(),
                "metadata": metadata or {},
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> SolutionPredictor:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format") != "onts-predictor-v1":
            raise ValueError("Unsupported predictor checkpoint")
        model = cls(checkpoint["hidden"], checkpoint["layers"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.training_metadata = checkpoint.get("metadata", {})
        model.eval()
        return model


def collect_solutions(problem: ONTSMilp, rng, *, budget: float = 6.0):
    """Diverse validated schedules: exact anytime, local search and profile moves.

    Returns a list of (schedule bool [J,T], objective). Costs belong to label
    generation and are reported separately from search time.
    """
    pool: dict[bytes, tuple[torch.Tensor, float]] = {}

    def add(schedule, value):
        if schedule is not None:
            key = schedule.numpy().tobytes()
            pool.setdefault(key, (schedule.detach().cpu().bool().clone(), float(value)))

    share = budget / 3
    for limit in (share * 0.1, share * 0.3, share * 0.6):
        result = problem.solve(time_limit=max(limit, 0.05))
        add(result.schedule, result.objective or 0.0)
    weighted_local_search(
        problem,
        None,
        rng,
        deadline=perf_counter() + share,
        on_improvement=add,
    )
    best = max(pool.values(), key=lambda item: item[1], default=None)
    if best is not None:
        result = solve_profile(
            problem.instance,
            time_limit=share,
            initial_schedule=best[0],
            seed=int(rng.integers(1 << 30)),
            config=problem.config,
        )
        add(result.schedule, result.objective)
    return list(pool.values())


def contrastive_samples(problem, solutions, rng, *, positives=3, corrupt=4):
    """Top schedules are positives; the rest plus infeasible corruptions negative."""
    ranked = sorted(solutions, key=lambda item: -item[1])
    pos = [s for s, _ in ranked[:positives]]
    neg = [s for s, _ in ranked[positives:]]
    legal = np.flatnonzero(problem.upper[: problem.n_x] > 0)
    for s in pos:
        for _ in range(corrupt):
            y = s.numpy().copy().ravel()
            flips = rng.choice(legal, size=min(len(legal), max(2, len(legal) // 20)))
            y[flips] = ~y[flips]
            y = torch.from_numpy(y.reshape(problem.J, problem.T))
            if not problem.config.validate(problem.instance, y).feasible:
                neg.append(y)
    return pos, neg


def contrastive_loss(logits, legal, positives, negatives, *, temperature=0.5):
    """ConPaS InfoNCE on mean per-cell log-likelihood over legal cells."""
    log_p, log_q = F.logsigmoid(logits), F.logsigmoid(-logits)

    def score(schedule):
        y = schedule.reshape(-1).to(logits.dtype)
        return ((y * log_p + (1 - y) * log_q) * legal).sum() / legal.sum()

    pos = torch.stack([score(s) for s in positives]) / temperature
    if not negatives:
        return -pos.mean()
    neg = torch.stack([score(s) for s in negatives]) / temperature
    return torch.stack(
        [-(p - torch.logsumexp(torch.cat([p[None], neg]), 0)) for p in pos]
    ).mean()


def train_predictor(
    instances,
    *,
    config=None,
    epochs=30,
    hidden=32,
    layers=2,
    lr=2e-3,
    collect_budget=6.0,
    seed=0,
    device="cpu",
):
    started = perf_counter()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    data = []
    for instance in instances:
        problem = ONTSMilp(instance, config)
        solutions = collect_solutions(problem, rng, budget=collect_budget)
        if len(solutions) < 2:
            continue
        pos, neg = contrastive_samples(problem, solutions, rng)
        legal = torch.from_numpy(
            (problem.upper[: problem.n_x] > 0).astype(np.float32)
        ).to(device)
        data.append((static_graph(problem).to(device), legal, pos, neg))
    labeling = perf_counter() - started
    model = SolutionPredictor(hidden, layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in range(epochs if data else 0):
        total = 0.0
        for index in rng.permutation(len(data)):
            graph, legal, pos, neg = data[index]
            logits = model(graph)
            loss = contrastive_loss(logits, legal, pos, neg)
            # Small supervised anchor to the best schedule stabilizes scale.
            best = pos[0].reshape(-1).to(logits.dtype).to(device)
            loss = (
                loss
                + 0.1
                * (
                    F.binary_cross_entropy_with_logits(logits, best, reduction="none")
                    * legal
                ).sum()
                / legal.sum()
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()
        losses.append(total / len(data))
    model.eval()
    return model, {
        "trained": bool(data),
        "training_instances": len(data),
        "loss": losses,
        "labeling_runtime": labeling,
        "total_runtime": perf_counter() - started,
        "seed": seed,
    }


def trust_region_row(problem, probabilities, *, k1, k0, delta):
    """Row and selected sets for sum_{S1}(1-x) + sum_{S0} x <= delta."""
    legal = np.flatnonzero(problem.upper[: problem.n_x] > 0)
    order = legal[np.argsort(-probabilities[legal], kind="stable")]
    k1, k0 = min(k1, len(order)), min(k0, max(0, len(order) - k1))
    ones, zeros = order[:k1], order[len(order) - k0 :] if k0 else order[:0]
    coefficients = np.zeros(problem.n_vars)
    coefficients[ones] = -1
    coefficients[zeros] = 1
    row = LinearConstraint(csr_matrix(coefficients.reshape(1, -1)), -np.inf, delta - k1)
    return row, ones, zeros


def solve_predict_search(
    instance,
    predictor: SolutionPredictor,
    *,
    time_limit=30.0,
    k1_fraction=0.1,
    k0_fraction=0.4,
    delta_fraction=0.05,
    config=None,
    seed=0,
    region_share=0.5,
    polish=False,
):
    """Trust-region MILP around predicted marginals; widen when exhausted.

    Each region solve gets region_share of the remaining time. A region proven
    optimal widens delta (x2) and re-solves with the incumbent; a region without
    a solution halves k0/k1; if all fail, the unrestricted model gets the rest.
    polish=True hands remaining time to whole-job profile search. All trust-region
    solves are restricted and report no global bound.
    """
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite time limit required")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, time_limit * 0.05)
    problem = ONTSMilp(instance, config)
    with torch.no_grad():
        probabilities = (
            torch.sigmoid(predictor(static_graph(problem))).cpu().double().numpy()
        )
    n_legal = int((problem.upper[: problem.n_x] > 0).sum())
    k1 = math.ceil(n_legal * k1_fraction)
    k0 = math.ceil(n_legal * k0_fraction)
    delta = max(1, math.ceil(n_legal * delta_fraction))
    history = [dict(time=perf_counter() - started, event="prediction")]
    incumbent, objective, first, iterations = None, None, None, 0
    while perf_counter() < deadline and (k1 or k0):
        row, _, _ = trust_region_row(problem, probabilities, k1=k1, k0=k0, delta=delta)
        sub = copy(problem)
        sub.constraints = [problem.constraints, row]
        result = sub.solve(
            time_limit=max(1e-3, region_share * (deadline - perf_counter())),
            incumbent=incumbent,
        )
        iterations += 1
        history.append(
            dict(
                time=perf_counter() - started,
                event="trust_region",
                objective=result.objective,
                status=result.status,
                restricted=True,
                k1=k1,
                k0=k0,
                delta=delta,
            )
        )
        if result.schedule is None:
            k1, k0 = k1 // 2, k0 // 2
            continue
        if first is None:
            first = perf_counter() - started
        incumbent, objective = result.schedule, result.objective
        if result.status != "optimal" or delta >= k1 + k0:
            break
        delta = min(k1 + k0, delta * 2)
    if incumbent is None and perf_counter() < deadline:
        # Every trust region failed; fall back to the unrestricted model.
        result = problem.solve(time_limit=max(1e-3, deadline - perf_counter()))
        history.append(
            dict(
                time=perf_counter() - started,
                event="fallback_exact",
                objective=result.objective,
                status=result.status,
            )
        )
        if result.schedule is not None:
            incumbent, objective = result.schedule, result.objective
            first = perf_counter() - started
    if polish and incumbent is not None and perf_counter() < deadline:
        polished = solve_profile(
            instance,
            time_limit=max(1e-3, deadline - perf_counter()),
            initial_schedule=incumbent,
            seed=seed,
            config=config,
        )
        offset = perf_counter() - started - polished.runtime
        history.extend(
            {**h, "time": h["time"] + offset, "phase": "polish"}
            for h in polished.history
        )
        incumbent, objective = polished.schedule, polished.objective
    if (
        incumbent is not None
        and not problem.config.validate(instance, incumbent).feasible
    ):
        raise RuntimeError("Predict-and-search returned an infeasible schedule")
    return HybridResult(
        "feasible" if incumbent is not None else "unknown",
        incumbent,
        objective,
        None,
        None,
        perf_counter() - started,
        first,
        iterations,
        history,
    )
