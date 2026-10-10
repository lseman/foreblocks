"""Validated ranked-neighborhood search and adaptive portfolio controls."""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

from .hybrid import HybridResult
from .milp import ONTSConfig, ONTSMilp
from .ranking import (
    OPERATORS,
    NeighborhoodRanker,
    RankingGraphBuilder,
    SearchState,
    make_candidates,
)


def solve_ranked(
    instance,
    *,
    policy: NeighborhoodRanker | None = None,
    mode="ranked",
    time_limit=30.0,
    repair_time=1.0,
    init_fraction=0.3,
    fraction=0.35,
    candidate_count=8,
    block_size=4,
    exploration=0.15,
    seed=0,
    config: ONTSConfig | None = None,
    initial_schedule=None,
    max_iterations=1000,
):
    if mode not in ("ranked", "uniform", "bandit", "adaptive"):
        raise ValueError("Unknown ranked search mode")
    if mode in ("ranked", "adaptive") and policy is None:
        raise ValueError("Ranked/adaptive search requires a ranker")
    if policy is not None and not isinstance(policy, NeighborhoodRanker):
        raise ValueError("Expected NeighborhoodRanker")
    if not all(math.isfinite(v) and v > 0 for v in (time_limit, repair_time)):
        raise ValueError("Time limits must be positive and finite")
    if not 0 < init_fraction <= 1 or not 0 < fraction <= 1 or not 0 <= exploration <= 1:
        raise ValueError("Invalid fractions/exploration")
    if candidate_count < 4 or block_size < 1 or max_iterations < 0:
        raise ValueError("Invalid candidate count, block size or iteration limit")
    started = perf_counter()
    reserve = min(0.25, time_limit * 0.05)
    deadline = started + time_limit - reserve

    def remaining():
        return deadline - perf_counter()

    p = ONTSMilp(instance, config)
    rng = np.random.default_rng(seed)
    history, bound, first, iterations = [], None, None, 0
    incumbent, objective, status = None, None, "unknown"

    def record(event, result, **extra):
        history.append(
            {
                "time": perf_counter() - started,
                "event": event,
                "objective": result.objective,
                "status": result.status,
                "restricted": result.restricted,
                "solve_runtime": result.runtime,
                **extra,
            }
        )

    if initial_schedule is not None:
        report = p.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("Initial schedule is infeasible")
        incumbent, objective = (
            initial_schedule.detach().cpu().bool().clone(),
            report.objective,
        )
        first = perf_counter() - started
        history.append(
            {"time": first, "event": "initial_schedule", "objective": objective}
        )
    elif remaining() > 0:
        result = p.solve(time_limit=min(remaining(), init_fraction * time_limit))
        record("initial_exact", result)
        incumbent, objective, bound, status = (
            result.schedule,
            result.objective,
            result.upper_bound,
            result.status,
        )
        if incumbent is not None:
            first = perf_counter() - started
        elif remaining() > 0 and status != "infeasible":
            result = p.solve(time_limit=remaining())
            record("initial_retry", result)
            incumbent, objective, status = (
                result.schedule,
                result.objective,
                result.status,
            )
            if result.upper_bound is not None:
                bound = (
                    min(bound, result.upper_bound)
                    if bound is not None
                    else result.upper_bound
                )
            if incumbent is not None:
                first = perf_counter() - started
    state = SearchState.empty(p.J, p.T)
    builder = (
        RankingGraphBuilder(p)
        if mode in ("ranked", "adaptive") and incumbent is not None and remaining() > 0
        else None
    )
    pulls, rewards = np.zeros(len(OPERATORS)), np.zeros(len(OPERATORS))
    scale = max(float(instance.priority.sum()) * p.T, 1.0)
    while incumbent is not None and remaining() > 0 and iterations < max_iterations:
        if bound is not None and bound - objective <= 1e-6:
            break
        selection_started = perf_counter()
        candidates = make_candidates(
            p,
            incumbent,
            state,
            rng,
            fraction=fraction,
            count=candidate_count,
            block_size=block_size,
        )
        scores = None
        if builder is not None:
            was_training = policy.training
            policy.eval()
            try:
                with torch.no_grad():
                    scores, _ = policy(
                        builder.build_state(
                            incumbent,
                            state,
                            remaining_fraction=max(remaining(), 0) / time_limit,
                            repair_time=repair_time,
                        ),
                        candidates,
                    )
                    scores = scores.detach().cpu().numpy()
            finally:
                policy.train(was_training)
            if not np.isfinite(scores).all():
                raise ValueError("Ranker returned nonfinite scores")
        if mode in ("bandit", "adaptive"):
            # UCB on bounded realized gain per wall time, not a best-arm guarantee.
            value = rewards / np.maximum(pulls, 1) + np.sqrt(
                2 * np.log(2 + pulls.sum()) / np.maximum(pulls, 1)
            )
            value[pulls == 0] = np.inf
            available = [
                i
                for i in range(len(OPERATORS))
                if any(c.operator == OPERATORS[i] for c in candidates)
            ]
            arm = max(available, key=lambda i: value[i])
            choices = [
                i for i, c in enumerate(candidates) if c.operator == OPERATORS[arm]
            ]
        else:
            choices = list(range(len(candidates)))
        if scores is None or rng.random() < exploration:
            selected = int(rng.choice(choices))
        else:
            selected = max(choices, key=lambda i: float(scores[i]))
        candidate = candidates[selected]
        selection_runtime = perf_counter() - selection_started
        if remaining() <= 0:
            break
        previous, old_objective = incumbent, objective
        result = p.solve(
            time_limit=min(remaining(), repair_time),
            incumbent=incumbent,
            free_mask=candidate.mask,
        )
        incumbent, objective = result.schedule, result.objective
        gain = objective - old_objective
        record(
            "repair",
            result,
            operator=candidate.operator,
            released_cells=int(candidate.mask.sum()),
            selection_runtime=selection_runtime,
            candidate_count=len(candidates),
            score=float(scores[selected]) if scores is not None else None,
        )
        state.advance(candidate.mask, previous, incumbent, gain, result.runtime)
        arm = OPERATORS.index(candidate.operator)
        pulls[arm] += 1
        rewards[arm] += min(
            1.0,
            max(0.0, gain / scale)
            * repair_time
            / max(result.runtime + selection_runtime, 1e-6),
        )
        iterations += 1
    if incumbent is not None:
        if not p.config.validate(instance, incumbent).feasible:
            raise RuntimeError("Ranked search returned an infeasible incumbent")
        status = (
            "optimal" if bound is not None and bound - objective <= 1e-6 else "feasible"
        )
    gap = (
        max(0.0, bound - objective) / max(abs(objective), 1e-9)
        if bound is not None and objective is not None
        else None
    )
    return HybridResult(
        status,
        incumbent,
        objective,
        bound,
        gap,
        perf_counter() - started,
        first,
        iterations,
        history,
    )
