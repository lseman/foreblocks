"""Whole-job profile moves: exact per-job DP priced by energy, then validation.

The per-job DP enumerates activation profiles under startup counts, windows,
minimum/maximum period and run length, as in ONTS column-generation pricing.
Using it as a large-neighborhood move is our adaptation, not a reproduction of
branch-and-price: battery coupling is handled by per-time energy prices that are
raised on violated windows, and every candidate is replayed by the validator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from scipy.optimize import linprog

from .hybrid import HybridResult
from .milp import ONTSMilp


@dataclass(frozen=True)
class JobRules:
    min_starts: int
    max_starts: int
    min_run: int
    max_run: int
    min_gap: int
    max_gap: int
    win_min: int
    win_max: int


def job_rules(problem: ONTSMilp, j: int) -> JobRules:
    i = problem.instance
    return JobRules(
        *(
            int(getattr(i, name)[j])
            for name in (
                "min_statup",
                "max_statup",
                "min_cpu_time",
                "max_cpu_time",
                "min_periodo_job",
                "max_periodo_job",
                "win_min",
                "win_max",
            )
        )
    )


def _range_max(values: np.ndarray):
    """Sparse table for O(1) vectorized range-maximum queries over values."""
    T = len(values)
    table = [values]
    width = 1
    while 2 * width <= T:
        prev = table[-1]
        level = np.full(T, -math.inf)
        level[: T - 2 * width + 1] = np.maximum(
            prev[: T - 2 * width + 1], prev[width : T - width + 1]
        )
        table.append(level)
        width *= 2
    return np.stack(table)


def best_profile(
    reward: np.ndarray, rules: JobRules, *, truncated_final_run: bool = True
) -> tuple[np.ndarray | None, float]:
    """Maximize sum(reward * row) over rows satisfying one job's constraints.

    reward[t] = -inf forbids slot t. Returns (None, -inf) if no profile exists.
    Maximum period P <= T means: first start <= P-1, consecutive starts at most
    P apart and last start >= T-P, which is equivalent to every full window of
    length P containing a start. Consecutive runs need an idle slot between them.
    Vectorized over start times; ties prefer shorter runs, then stopping, then
    the earliest next start.
    """
    reward = np.asarray(reward, dtype=float)
    T = len(reward)
    allowed = np.isfinite(reward)
    allowed[: max(0, rules.win_min)] = False
    allowed[min(T, rules.win_max) :] = False
    # nxt[t]: first forbidden slot at or after t.
    blocked = np.flatnonzero(~allowed)
    nxt = np.full(T + 1, T)
    position = np.searchsorted(blocked, np.arange(T + 1))
    has = position < len(blocked)
    nxt[has] = blocked[position[has]]
    prefix = np.concatenate([[0.0], np.cumsum(np.where(allowed, reward, 0.0))])
    periodic = rules.max_gap <= T
    # Non-final starts are at least max(min_gap, min_run + 1) apart.
    spacing = max(1, rules.min_gap, rules.min_run + 1)
    K = min(rules.max_starts, 1 + (T - 1) // spacing)
    if rules.min_starts > K:
        return None, -math.inf
    starts = np.arange(T)
    shortest = (
        np.minimum(rules.min_run, T - starts)
        if truncated_final_run
        else np.full(T, rules.min_run)
    )
    longest = np.minimum(np.minimum(rules.max_run, T - starts), nxt[:T] - starts)
    log2 = np.zeros(T + 2, dtype=int)
    log2[2:] = np.floor(np.log2(np.arange(2, T + 2))).astype(int)
    last_ok = (starts >= T - rules.max_gap) if periodic else np.ones(T, bool)
    value = np.full((K + 2, T), -math.inf)
    run_length = np.zeros((K + 2, T), dtype=int)
    max_length = int(min(rules.max_run, T))
    for k in range(K, 0, -1):
        table = _range_max(value[k + 1]) if k < K else None
        terminal = np.where(last_ok & (k >= rules.min_starts), 0.0, -math.inf)
        for length in range(1, max_length + 1):
            ok = allowed & (length >= shortest) & (length <= longest)
            if not ok.any():
                continue
            end = np.minimum(starts + length, T)
            total = prefix[end] - prefix[starts]
            cont = terminal.copy()
            if table is not None:
                lo = np.maximum(end + 1, starts + rules.min_gap)
                hi = (
                    np.minimum(starts + rules.max_gap, T - 1)
                    if periodic
                    else np.full(T, T - 1)
                )
                valid = ok & (lo <= hi)
                if valid.any():
                    a, b = lo[valid], hi[valid]
                    level = log2[b - a + 1]
                    nextv = np.maximum(
                        table[level, a], table[level, b - (1 << level) + 1]
                    )
                    cont[valid] = np.maximum(cont[valid], nextv)
            total = np.where(ok, total + cont, -math.inf)
            better = total > value[k]
            value[k, better] = total[better]
            run_length[k, better] = length
    best, first = -math.inf, -1
    if rules.min_starts == 0 and not periodic:
        best = 0.0
    last_first = min(T - 1, rules.max_gap - 1) if periodic else T - 1
    candidates = value[1, : last_first + 1]
    if len(candidates) and candidates.max() > best:
        first = int(np.argmax(candidates))
        best = float(candidates[first])
    if not math.isfinite(best):
        return None, -math.inf
    row = np.zeros(T, dtype=bool)
    k, s = 1, first
    while s >= 0:
        length = int(run_length[k, s])
        row[s : s + length] = True
        end = s + length
        stop = 0.0 if (last_ok[s] and k >= rules.min_starts) else -math.inf
        following = -1
        if k < K:
            lo = max(end + 1, s + rules.min_gap)
            hi = min(s + rules.max_gap, T - 1) if periodic else T - 1
            if lo <= hi:
                segment = value[k + 1, lo : hi + 1]
                if segment.max() > stop:
                    following = lo + int(np.argmax(segment))
        k, s = k + 1, following
    return row, best


def battery_trace(problem: ONTSMilp, usage: np.ndarray) -> np.ndarray:
    """Closed form of soc[t] = min(1, soc[t-1] + d[t]) used by the validator."""
    c = problem.config
    k = c.ef / (c.q * c.v_bat * 60)
    solar = problem.instance.recurso_p.cpu().double().numpy()
    cumulative = np.cumsum(k * (solar - usage))
    return cumulative + np.minimum(c.soc_inicial, 1 - np.maximum.accumulate(cumulative))


def curtailed_power(problem: ONTSMilp, usage: np.ndarray) -> np.ndarray:
    """Per-slot power discarded by the upper battery bound, in watts."""
    c = problem.config
    k = c.ef / (c.q * c.v_bat * 60)
    soc = battery_trace(problem, usage)
    previous = np.r_[c.soc_inicial, soc[:-1]]
    solar = problem.instance.recurso_p.cpu().double().numpy()
    return np.maximum(0.0, previous + k * (solar - usage) - 1.0) / k


def lp_energy_prices(problem: ONTSMilp, *, time_limit: float = 1.0):
    """Marginal LP objective cost of one watt drawn at each time, or None.

    Combines duals of the instantaneous power row and the SoC energy row; usage
    at t appears only in those two rows. LP duals are guidance, not bounds.
    """
    A = problem.A.tocsr()
    finite_hi = np.isfinite(problem.row_upper)
    finite_lo = np.isfinite(problem.row_lower)
    import scipy.sparse as sp

    A_ub = sp.vstack([A[finite_hi], -A[finite_lo]]).tocsr()
    b_ub = np.concatenate([problem.row_upper[finite_hi], -problem.row_lower[finite_lo]])
    result = linprog(
        problem.c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=np.stack([problem.lower, problem.upper], axis=1),
        method="highs",
        options={"time_limit": float(time_limit)},
    )
    if result.status != 0:
        return None
    marginals = -np.asarray(result.ineqlin.marginals[: int(finite_hi.sum())])
    by_row = np.zeros(len(problem.row_upper))
    by_row[np.flatnonzero(finite_hi)] = marginals
    c = problem.config
    k = c.ef / (c.q * c.v_bat * 60)
    prices = np.zeros(problem.T)
    for row, name in enumerate(problem.row_names):
        family, _, t = name.partition(":")
        if family == "power":
            prices[int(t)] += by_row[row]
        elif family == "energy":
            prices[int(t)] += k * by_row[row]
    return np.maximum(prices, 0.0)


def _violation_windows(problem: ONTSMilp, soc: np.ndarray) -> list[tuple[int, int]]:
    """For each battery deficit, the slots since the battery was last full."""
    lower = problem.config.limite_inferior
    windows, start = [], 0
    for t, level in enumerate(soc):
        if level >= 1 - 1e-12:
            start = t + 1
        elif level < lower - 1e-9:
            windows.append((start, t + 1))
    return windows


def reprofile(
    problem: ONTSMilp,
    incumbent: torch.Tensor,
    jobs,
    rng: np.random.Generator,
    *,
    base_prices: np.ndarray | None = None,
    iterations: int = 8,
    deadline: float | None = None,
    surplus_weight: float = 0.0,
    surplus_mode: str = "saturation",
    accept_equal: bool = False,
):
    """Remove `jobs`, reinsert each by priced DP and keep the best valid result.

    Prices start at base_prices; deficit windows raise them geometrically until
    the battery replay is feasible, then bisection lowers them toward the
    cheapest feasible price. Returns (schedule, objective, info); the schedule
    is the incumbent itself when no validated improvement is found.

    surplus_weight > 0 discounts slots where the battery is full without the
    job (that solar would be curtailed): reward is reduced by
    surplus_weight * priority * draw/max_draw only in unsaturated slots.
    With surplus_mode="curtailment", the charged draw is max(0, draw - spill),
    where spill is the partial schedule's curtailed power in watts. This local
    surrogate does not account for changes in later spill after insertion.
    accept_equal keeps an equal-objective schedule with more total battery
    charge (lexicographic objective, then reserve); the objective never falls.
    """
    if surplus_mode not in ("saturation", "curtailment"):
        raise ValueError("surplus_mode must be 'saturation' or 'curtailment'")
    if not math.isfinite(surplus_weight) or surplus_weight < 0:
        raise ValueError("surplus_weight must be finite and nonnegative")
    p, inst = problem, problem.instance
    x = incumbent.detach().cpu().bool().numpy().copy()
    priority = inst.priority.cpu().double().numpy()
    draw = inst.uso_p.cpu().double().numpy()
    capacity = (
        inst.recurso_p.cpu().double().numpy() + p.config.bat_usage * p.config.v_bat
    )
    base = (
        np.zeros(p.T) if base_prices is None else np.asarray(base_prices, float).copy()
    )
    rho = float(np.median(priority / np.maximum(draw, 1e-6)))
    jobs = [int(j) for j in jobs]
    start_objective = p.config.validate(inst, incumbent).objective
    best, best_objective = incumbent.detach().cpu().bool().clone(), start_objective
    best_reserve = float(battery_trace(p, draw @ x).sum())
    start_reserve = best_reserve
    max_draw = max(float(draw.max()), 1e-9)
    infeasible_prices, feasible_prices = None, None
    prices = base.copy()
    evaluated = 0
    for step in range(iterations):
        if deadline is not None and perf_counter() > deadline:
            break
        y = x.copy()
        y[jobs] = False
        order = sorted(jobs, key=lambda j: -priority[j] / max(draw[j], 1e-6))
        if step:
            rng.shuffle(order)
        complete = True
        for j in order:
            usage = draw @ y
            spare = capacity - usage
            reward = priority[j] - prices * draw[j]
            if surplus_weight:
                if surplus_mode == "curtailment":
                    charged_draw = np.maximum(0.0, draw[j] - curtailed_power(p, usage))
                else:
                    charged_draw = (battery_trace(p, usage) < 1 - 1e-9) * draw[j]
                reward = reward - surplus_weight * priority[j] * charged_draw / max_draw
            reward[draw[j] > spare + 1e-9] = -math.inf
            row, _ = best_profile(
                reward,
                job_rules(p, j),
                truncated_final_run=p.config.allow_truncated_final_run,
            )
            if row is None:
                complete = False
                break
            y[j] = row
        if not complete:
            # A job lost every power-feasible profile; prices cannot help.
            break
        evaluated += 1
        # DP enforces each job's own rules and the reward mask enforces power,
        # so only the battery needs checking here; accepted schedules are still
        # passed through the full validator below.
        soc = battery_trace(p, draw @ y)
        battery_ok = soc.min() >= p.config.limite_inferior - 1e-6
        if battery_ok:
            feasible_prices = prices.copy()
            objective = float(priority @ y.sum(axis=1))
            reserve = float(soc.sum())
            improves = objective > best_objective + 1e-9
            if (
                accept_equal
                and not improves
                and objective >= best_objective - 1e-9
                and reserve > best_reserve + 1e-9
            ):
                improves = True
            if improves:
                report = p.config.validate(inst, torch.from_numpy(y))
                if not report.feasible:
                    raise RuntimeError(f"Profile move invalid: {report.violations}")
                best, best_objective = torch.from_numpy(y), report.objective
                best_reserve = reserve
            if infeasible_prices is None:
                prices = (prices + base) / 2 if np.any(prices > base) else prices
                if np.allclose(prices, feasible_prices):
                    break
            else:
                prices = (infeasible_prices + feasible_prices) / 2
        else:
            infeasible_prices = prices.copy()
            if feasible_prices is not None:
                prices = (infeasible_prices + feasible_prices) / 2
            else:
                for lo, hi in _violation_windows(p, soc):
                    prices[lo:hi] += rho * 0.5 * (1.6**step)
    info = dict(
        jobs=jobs,
        evaluations=evaluated,
        gain=best_objective - start_objective,
        reserve_gain=best_reserve - start_reserve,
        changed_cells=int(torch.count_nonzero(best != incumbent)),
    )
    return best, best_objective, info


def select_jobs(problem, incumbent, rng, count, *, rule="pressure"):
    """Choose jobs to re-profile: random, utility or battery-pressure."""
    J = problem.J
    count = max(1, min(count, J))
    inst = problem.instance
    x = incumbent.detach().cpu().bool().numpy()
    if rule == "random":
        return rng.choice(J, size=count, replace=False).tolist()
    priority = inst.priority.cpu().double().numpy()
    draw = inst.uso_p.cpu().double().numpy()
    if rule == "utility":
        weight = priority / np.maximum(draw, 1e-6)
    else:
        soc = battery_trace(problem, draw @ x)
        tight = 1.0 - (soc - problem.config.limite_inferior)
        weight = (x * tight[None, :]).sum(axis=1) * draw + 1e-3
    weight = weight - weight.min() + 1e-3
    return rng.choice(J, size=count, replace=False, p=weight / weight.sum()).tolist()


def solve_profile(
    instance,
    *,
    time_limit=30.0,
    init_fraction=0.3,
    seed=0,
    config=None,
    initial_schedule=None,
    prices="zero",
    max_jobs=3,
    fallback_time=0.0,
    max_iterations=100000,
):
    """Anytime search using only whole-job DP moves, with optional MILP fallback.

    prices: "zero" (default; selected on development instances) or "lp" (root
    LP duals, which over-priced energy in development runs). With fallback_time > 0, a move that
    finds no DP improvement repairs the same released jobs exactly by MILP.
    """
    if prices not in ("lp", "zero"):
        raise ValueError("prices must be 'lp' or 'zero'")
    if not math.isfinite(time_limit) or time_limit <= 0 or max_jobs < 1:
        raise ValueError("Invalid time limit or job count")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, time_limit * 0.05)

    def remaining():
        return deadline - perf_counter()

    problem = ONTSMilp(instance, config)
    rng = np.random.default_rng(seed)
    history, bound, first, status = [], None, None, "unknown"
    incumbent, objective = None, None
    if initial_schedule is not None:
        report = problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("Infeasible initial schedule")
        incumbent = initial_schedule.detach().cpu().bool().clone()
        objective = report.objective
    else:
        result = problem.solve(time_limit=max(1e-3, remaining() * init_fraction))
        incumbent, objective, bound = (
            result.schedule,
            result.objective,
            result.upper_bound,
        )
        status = result.status
        history.append(
            dict(
                time=perf_counter() - started,
                event="initial_exact",
                objective=objective,
            )
        )
    if incumbent is not None:
        first = perf_counter() - started
    base = None
    if incumbent is not None and prices == "lp" and remaining() > 0:
        base = lp_energy_prices(
            problem, time_limit=max(1e-3, min(1.0, remaining() / 4))
        )
        history.append(
            dict(
                time=perf_counter() - started,
                event="lp_prices",
                objective=objective,
                available=base is not None,
            )
        )
    rules = ("pressure", "utility", "random")
    iterations, stagnation = 0, 0
    legal = (problem.upper[: problem.n_x] > 0).reshape(problem.J, problem.T)
    while incumbent is not None and remaining() > 0 and iterations < max_iterations:
        if bound is not None and bound - objective <= 1e-6:
            break
        count = 1 + min(max_jobs - 1, stagnation // 4)
        rule = rules[iterations % len(rules)]
        jobs = select_jobs(problem, incumbent, rng, count, rule=rule)
        old = objective
        candidate, value, info = reprofile(
            problem, incumbent, jobs, rng, base_prices=base, deadline=deadline
        )
        operator = "profile"
        if value <= old + 1e-9 and fallback_time > 0 and remaining() > 0:
            mask = np.zeros_like(legal)
            mask[jobs] = legal[jobs]
            result = problem.solve(
                incumbent=incumbent,
                free_mask=torch.from_numpy(mask),
                time_limit=min(remaining(), fallback_time),
            )
            candidate, value, operator = result.schedule, result.objective, "job_milp"
        incumbent, objective = candidate, value
        stagnation = 0 if objective > old + 1e-9 else stagnation + 1
        history.append(
            dict(
                time=perf_counter() - started,
                event="repair",
                operator=operator,
                rule=rule,
                objective=objective,
                restricted=True,
                **info,
            )
        )
        iterations += 1
    if incumbent is not None:
        if not problem.config.validate(instance, incumbent).feasible:
            raise RuntimeError("Profile search returned an infeasible schedule")
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
