"""Portable classical ONTS controls: LP disagreement, local branching and VNS.

`adaptive_local_branch` adapts the Hamming radius from repair outcomes, as in
Fischetti-Lodi local branching: a proven-exhausted ball grows, a timed-out
ball without improvement shrinks. `portfolio_plus` adds whole-job profile
moves and weighted local search to the round-robin portfolio.

Mathematical adaptations on the existing HiGHS backend, not SCIP/Balans replicas.
All construction, LP relaxation and repair costs belong to the search deadline.
"""

from __future__ import annotations

import math
import warnings
from copy import copy
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

from .hybrid import HybridResult
from .job_profile import reprofile, select_jobs
from .local_search import weighted_local_search
from .milp import ONTSMilp, SolveResult
from .ranking import SearchState, make_candidates


def lp_relaxation(problem, *, time_limit):
    """Return a root LP optimum only; a timed-out primal is not an upper bound."""
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite LP time limit required")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Unrecognized options detected:.*",
            category=RuntimeWarning,
        )
        result = milp(
            c=problem.c,
            integrality=np.zeros(problem.n_vars),
            bounds=Bounds(problem.lower, problem.upper),
            constraints=problem.constraints,
            options={"time_limit": time_limit, "threads": 1, "random_seed": 0},
        )
    if result.status != 0 or result.x is None or not np.isfinite(result.x).all():
        return None
    return np.asarray(result.x[: problem.n_x]).reshape(problem.J, problem.T)


def lp_disagreement_mask(problem, incumbent, relaxation, *, fraction=0.2, rins=False):
    if not 0 < fraction <= 1:
        raise ValueError("Invalid neighborhood fraction")
    values = np.asarray(relaxation, dtype=float)
    if values.shape != (problem.J, problem.T) or not np.isfinite(values).all():
        raise ValueError("Finite LP execution assignment required")
    distance = np.abs(values - incumbent.detach().cpu().numpy()).ravel()
    legal = problem.upper[: problem.n_x] > 0
    if rins:
        free = legal & (distance > 1e-6)
    else:
        free = np.zeros(problem.n_x, dtype=bool)
        order = np.flatnonzero(legal)
        # Stable tie ordering; random/temporal controls supply diversification.
        order = order[np.argsort(-distance[order], kind="stable")]
        free[order[: math.ceil(problem.n_x * fraction)]] = True
    return torch.from_numpy(free.reshape(problem.J, problem.T))


def local_branch(problem, incumbent, *, radius, time_limit):
    """Optimize a Hamming ball of x decisions while leaving startup/SoC free."""
    if not isinstance(radius, int) or radius < 0:
        raise ValueError("Nonnegative integer Hamming radius required")
    problem.assignment(incumbent)
    x = incumbent.detach().cpu().numpy().ravel().astype(float)
    coefficients = np.zeros(problem.n_vars)
    coefficients[: problem.n_x] = 1 - 2 * x
    row = LinearConstraint(
        csr_matrix(coefficients.reshape(1, -1)), -np.inf, radius - x.sum()
    )
    subproblem = copy(problem)
    subproblem.constraints = [problem.constraints, row]
    result = subproblem.solve(incumbent=incumbent, time_limit=time_limit)
    result.restricted = True
    result.upper_bound = None
    result.relative_gap = None
    if (
        result.schedule is not None
        and int((result.schedule != incumbent.bool()).sum()) > radius
    ):
        raise RuntimeError("Local-branch solution violates Hamming radius")
    return result


def energy_exchange_mask(problem, incumbent, rng, *, fraction=0.2):
    """Release a shared time region and every active run it intersects.

    All jobs can exchange power inside the region; a touched run is never split.
    Regions whose closure exceeds the cell budget are skipped, not truncated.
    """
    if not 0 < fraction <= 1:
        raise ValueError("Invalid neighborhood fraction")
    x = incumbent.detach().cpu().bool().numpy()
    legal = (problem.upper[: problem.n_x] > 0).reshape(problem.J, problem.T)
    usage = (x * problem.instance.uso_p.cpu().numpy()[:, None]).sum(axis=0)
    solar = problem.instance.recurso_p.cpu().numpy()
    pressure = usage / (solar + problem.config.bat_usage * problem.config.v_bat + 1e-8)
    weights = pressure - pressure.min() + 0.05
    centers = rng.choice(
        problem.T, size=min(12, problem.T), replace=False, p=weights / weights.sum()
    )
    budget = math.ceil(problem.n_x * fraction)
    width = max(1, int(problem.T * fraction / 2))
    runs = []
    for row in x:
        ends = np.flatnonzero(np.diff(np.r_[False, row, False]))
        runs.append(list(zip(ends[::2], ends[1::2])))
    for center in centers:
        mask = np.zeros_like(x)
        start = max(0, int(center) - width // 2)
        end = min(problem.T, start + width)
        mask[:, start:end] = legal[:, start:end]
        for j, segments in enumerate(runs):
            for lo, hi in segments:
                if mask[j, lo:hi].any():
                    mask[j, max(0, lo - 1) : min(problem.T, hi + 1)] |= legal[
                        j, max(0, lo - 1) : min(problem.T, hi + 1)
                    ]
        # Adjacent expansion may touch another active run; close transitively.
        changed = True
        while changed:
            old = mask.sum()
            for j, segments in enumerate(runs):
                for lo, hi in segments:
                    if mask[j, lo:hi].any():
                        mask[j, lo:hi] = True
            changed = mask.sum() != old
        if 0 < mask.sum() <= budget:
            return torch.from_numpy(mask)
    # Safe fallback can always release inactive cells without cutting a run.
    mask = np.zeros_like(x)
    cells = np.flatnonzero(legal & ~x)
    mask.ravel()[rng.permutation(cells)[:budget]] = True
    return torch.from_numpy(mask)


def adapt_radius(radius, *, gain, status, n_x):
    """Grow a proven-exhausted ball, shrink a timed-out one, keep after a gain."""
    if gain > 1e-6:
        return radius
    if status == "optimal":
        return min(n_x, max(radius + 1, math.ceil(radius * 1.5)))
    return max(2, (2 * radius) // 3)


def solve_classical(
    instance,
    *,
    mode="portfolio",
    time_limit=30.0,
    repair_time=1.0,
    fraction=0.2,
    init_fraction=0.3,
    seed=0,
    config=None,
    initial_schedule=None,
    max_iterations=1000,
):
    if mode not in (
        "lp_guided",
        "rins",
        "local_branch",
        "adaptive_local_branch",
        "portfolio",
        "portfolio_plus",
    ):
        raise ValueError("Unknown classical mode")
    if not all(math.isfinite(v) and v > 0 for v in (time_limit, repair_time)):
        raise ValueError("Positive finite time limits required")
    if not 0 < fraction <= 1 or not 0 < init_fraction <= 1 or max_iterations < 0:
        raise ValueError("Invalid fractions/iteration limit")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, time_limit * 0.05)

    def remaining():
        return deadline - perf_counter()

    problem = ONTSMilp(instance, config)
    history = []
    bound = None
    first = None
    iterations = 0
    status = "unknown"
    incumbent = None
    objective = None

    def record(event, result, **extra):
        history.append(
            dict(
                time=perf_counter() - started,
                event=event,
                objective=result.objective,
                status=result.status,
                restricted=result.restricted,
                solve_runtime=result.runtime,
                **extra,
            )
        )

    if initial_schedule is not None:
        report = problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("Infeasible initial schedule")
        incumbent = initial_schedule.detach().cpu().bool().clone()
        objective = report.objective
        first = perf_counter() - started
        history.append(dict(time=first, event="initial_schedule", objective=objective))
    elif remaining() > 0:
        result = problem.solve(time_limit=min(remaining(), time_limit * init_fraction))
        record("initial_exact", result)
        incumbent, objective, bound, status = (
            result.schedule,
            result.objective,
            result.upper_bound,
            result.status,
        )
        if incumbent is None and remaining() > 0 and status != "infeasible":
            result = problem.solve(time_limit=remaining())
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
    relaxation = None
    if (
        mode not in ("local_branch", "adaptive_local_branch")
        and incumbent is not None
        and remaining() > 0
    ):
        lp_started = perf_counter()
        relaxation = lp_relaxation(
            problem, time_limit=min(remaining(), repair_time * 0.5)
        )
        history.append(
            dict(
                time=perf_counter() - started,
                event="root_lp",
                objective=objective,
                optimal=relaxation is not None,
                solve_runtime=perf_counter() - lp_started,
            )
        )
    rng = np.random.default_rng(seed)
    state = SearchState.empty(problem.J, problem.T)
    stagnation = 0
    portfolio = mode in ("portfolio", "portfolio_plus")
    operators = (
        ("lp_guided", "energy_exchange", "run", "local_branch")
        if mode == "portfolio"
        else (
            "lp_guided",
            "energy_exchange",
            "run",
            "adaptive_local_branch",
            "profile",
            "local_search",
        )
    )
    no_mask = ("local_branch", "adaptive_local_branch", "profile", "local_search")
    lb_radius = max(2, math.ceil(problem.n_x * fraction / 2))
    while incumbent is not None and remaining() > 0 and iterations < max_iterations:
        if bound is not None and bound - objective <= 1e-6:
            break
        proposal_started = perf_counter()
        level = min(3, stagnation // 2)
        size = min(0.65, fraction * (1.5**level)) if portfolio else fraction
        operator = operators[iterations % len(operators)] if portfolio else mode
        if operator in ("lp_guided", "rins") and relaxation is not None:
            mask = lp_disagreement_mask(
                problem, incumbent, relaxation, fraction=size, rins=operator == "rins"
            )
        elif operator == "energy_exchange":
            mask = energy_exchange_mask(problem, incumbent, rng, fraction=size)
        elif operator not in no_mask:
            candidates = make_candidates(
                problem, incumbent, state, rng, fraction=size, count=8
            )
            choices = [c for c in candidates if c.operator == "run"] or candidates
            mask = choices[int(rng.integers(len(choices)))].mask
            if operator in ("lp_guided", "rins"):
                operator = "run_fallback"
        radius = max(1, math.ceil(problem.n_x * size))
        selection_runtime = perf_counter() - proposal_started
        if remaining() <= 0:
            break
        previous = incumbent
        old_objective = objective
        cap = min(
            remaining(),
            repair_time * (1.5**level) if portfolio else repair_time,
        )
        used_radius = lb_radius if operator == "adaptive_local_branch" else radius
        if operator in ("local_branch", "adaptive_local_branch"):
            result = local_branch(
                problem, incumbent, radius=used_radius, time_limit=cap
            )
        elif operator == "profile":
            move_started = perf_counter()
            jobs = select_jobs(
                problem,
                incumbent,
                rng,
                1 + min(2, level),
                rule=("pressure", "utility", "random")[iterations % 3],
            )
            schedule, value, _ = reprofile(
                problem, incumbent, jobs, rng, deadline=move_started + cap
            )
            result = SolveResult(
                "heuristic",
                "",
                schedule,
                value,
                None,
                None,
                perf_counter() - move_started,
                None,
                True,
            )
        elif operator == "local_search":
            move_started = perf_counter()
            schedule, value, _ = weighted_local_search(
                problem,
                incumbent.numpy(),
                rng,
                deadline=move_started + cap,
            )
            if schedule is None or value <= objective:
                schedule, value = incumbent, objective
            result = SolveResult(
                "heuristic",
                "",
                schedule,
                value,
                None,
                None,
                perf_counter() - move_started,
                None,
                True,
            )
        else:
            result = problem.solve(incumbent=incumbent, free_mask=mask, time_limit=cap)
        incumbent, objective = result.schedule, result.objective
        gain = objective - old_objective
        used = (incumbent != previous) if operator in no_mask else mask
        record(
            "repair",
            result,
            operator=operator,
            fraction=size,
            radius=used_radius if "local_branch" in operator else None,
            released_cells=None if operator in no_mask else int(mask.sum()),
            selection_runtime=selection_runtime,
        )
        if operator == "adaptive_local_branch":
            lb_radius = adapt_radius(
                lb_radius, gain=gain, status=result.status, n_x=problem.n_x
            )
        state.advance(used, previous, incumbent, gain, result.runtime)
        stagnation = 0 if gain > 1e-6 else stagnation + 1
        iterations += 1
    if incumbent is not None:
        if not problem.config.validate(instance, incumbent).feasible:
            raise RuntimeError("Invalid classical output")
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
