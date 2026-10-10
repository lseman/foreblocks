"""LP-free weighted local search for ONTS (Feasibility Jump / Local-MIP style).

Adapted to ONTS structure rather than generic MIP rows: moves act on runs
(insert, delete, extend, shrink, shift) and single cells. Constraint weights grow
on local minima as in Feasibility Jump. After a feasible solution, an objective
target slightly above the best value acts as an extra weighted constraint, the
analogue of Local-MIP's breakthrough move; moves that keep feasibility and raise
the objective play the role of its lift move. Not a reproduction of either.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import pairwise
from time import perf_counter

import numpy as np
import torch

from .hybrid import HybridResult
from .job_profile import battery_trace
from .milp import ONTSMilp

FAMILIES = 5  # startup count, minimum gap, maximum gap, minimum run, maximum run


class ScheduleState:
    """Violation amounts by job family and by time for power and battery."""

    def __init__(self, problem: ONTSMilp):
        self.p = problem
        inst = problem.instance
        self.J, self.T = problem.J, problem.T

        def arr(name, kind=int):
            return getattr(inst, name).cpu().numpy().astype(kind)

        self.priority = arr("priority", float)
        self.draw = arr("uso_p", float)
        self.smin, self.smax = arr("min_statup"), arr("max_statup")
        self.rmin, self.rmax = arr("min_cpu_time"), arr("max_cpu_time")
        self.gmin, self.gmax = arr("min_periodo_job"), arr("max_periodo_job")
        self.legal = (problem.upper[: problem.n_x] > 0).reshape(self.J, self.T)
        self.capacity = (
            inst.recurso_p.cpu().double().numpy()
            + problem.config.bat_usage * problem.config.v_bat
        )
        self.lower = problem.config.limite_inferior
        self.truncated = problem.config.allow_truncated_final_run

    def runs(self, row):
        edges = np.flatnonzero(np.diff(np.r_[False, row, False]))
        return list(zip(edges[::2].tolist(), edges[1::2].tolist()))

    def job_violation(self, j, row):
        T = self.T
        runs = self.runs(row)
        starts = [s for s, _ in runs]
        n = len(starts)
        out = np.zeros(FAMILIES)
        out[0] = max(0, self.smin[j] - n) + max(0, n - self.smax[j])
        out[1] = sum(max(0, self.gmin[j] - (b - a)) for a, b in pairwise(starts))
        P = self.gmax[j]
        if P <= T:
            phi = np.zeros(T + 1)
            phi[np.array(starts, dtype=int) + 1] = 1
            window = np.cumsum(phi)
            counts = window[P:] - window[: T - P + 1]
            out[2] = float((counts == 0).sum())
        for s, e in runs:
            minimum = min(self.rmin[j], T - s) if self.truncated else self.rmin[j]
            out[3] += max(0, minimum - (e - s))
            out[4] += max(0, (e - s) - self.rmax[j])
        return out

    def global_violation(self, usage):
        power = np.maximum(0.0, usage - self.capacity)
        soc = battery_trace(self.p, usage)
        battery = np.maximum(0.0, self.lower - soc) * 100.0
        return power, battery


def _moves(state: ScheduleState, j, row, rng, inserts=4, flips=3):
    """Structured neighbor rows of job j; illegal cells are never switched on."""
    T, legal = state.T, state.legal[j]
    out = []
    for s, e in state.runs(row):
        for lo, hi in (
            (s - 1, e),
            (s, e + 1),
            (s + 1, e),
            (s, e - 1),
            (s - 1, e - 1),
            (s + 1, e + 1),
        ):
            if 0 <= lo < hi <= T and legal[lo:hi].all():
                new = row.copy()
                new[s:e] = False
                new[lo:hi] = True
                out.append(new)
        new = row.copy()
        new[s:e] = False
        out.append(new)
    starts = np.flatnonzero(legal)
    if len(starts):
        length = int(max(1, state.rmin[j]))
        for s in rng.choice(starts, size=min(inserts, len(starts)), replace=False):
            hi = min(T, int(s) + length)
            if legal[s:hi].all() and not row[max(0, s - 1) : hi + 1].any():
                new = row.copy()
                new[s:hi] = True
                out.append(new)
        for t in rng.choice(starts, size=min(flips, len(starts)), replace=False):
            new = row.copy()
            new[t] = not new[t]
            out.append(new)
    return out


def weighted_local_search(
    problem: ONTSMilp,
    start: np.ndarray | None,
    rng: np.random.Generator,
    *,
    deadline: float,
    max_steps: int = 10**7,
    restart_after: int = 400,
    on_improvement=None,
    polish_after_feasible: float | None = None,
):
    """Return (best feasible schedule or None, its objective, steps).

    polish_after_feasible: if set, stop this many seconds after the first
    feasible schedule (at least as long as finding it took), so callers that
    only need a start do not spend their whole allowance here.
    """
    S = ScheduleState(problem)
    J, T = S.J, S.T
    x = (
        np.zeros((J, T), dtype=bool)
        if start is None
        else np.asarray(start, dtype=bool).copy()
    )
    x &= S.legal
    jw = np.ones((J, FAMILIES))
    pw, bw = np.ones(T), np.ones(T)
    obj_w = 1.0
    scale = max(float(S.priority.max()), 1e-9)
    jv = np.stack([S.job_violation(j, x[j]) for j in range(J)])
    usage = S.draw @ x
    power, battery = S.global_violation(usage)
    objective = float(S.priority @ x.sum(axis=1))
    best, best_obj = None, -math.inf
    target = None
    tabu = [deque(maxlen=8) for _ in range(J)]
    since = 0
    steps = 0

    def penalty(jvj, power_, battery_, obj):
        total = float((jw * jvj).sum()) + float(pw @ power_ + bw @ battery_)
        if target is not None:
            total += obj_w * max(0.0, target - obj) / scale
        return total

    def feasible_now():
        return not jv.any() and not power.any() and not battery.any()

    search_started = perf_counter()

    def accept_best():
        nonlocal best, best_obj, target, since, deadline
        report = problem.config.validate(problem.instance, torch.from_numpy(x))
        if report.feasible and report.objective > best_obj + 1e-9:
            if best is None and polish_after_feasible is not None:
                found = perf_counter() - search_started
                deadline = min(
                    deadline, perf_counter() + max(polish_after_feasible, found)
                )
            best, best_obj = x.copy(), report.objective
            target = best_obj + 0.5 * float(S.priority.min())
            since = 0
            if on_improvement is not None:
                on_improvement(torch.from_numpy(best.copy()), best_obj)

    if feasible_now():
        accept_best()
    current = penalty(jv, power, battery, objective)
    while steps < max_steps and perf_counter() < deadline:
        steps += 1
        since += 1
        # Focus on jobs with violations, jobs active where power/battery fail,
        # or any job when feasible (lift/breakthrough).
        violated_jobs = np.flatnonzero(jv.sum(axis=1) > 0)
        bad_t = np.flatnonzero((power > 0) | (battery > 0))
        if len(violated_jobs) and rng.random() < 0.6:
            j = int(rng.choice(violated_jobs))
        elif len(bad_t):
            t = int(rng.choice(bad_t))
            window = slice(max(0, t - 6), t + 1)
            active = np.flatnonzero(x[:, window].any(axis=1))
            j = int(rng.choice(active)) if len(active) else int(rng.integers(J))
        else:
            j = int(rng.integers(J))
        best_move, best_score = None, current - 1e-12
        base_jv = jv.copy()
        for new in _moves(S, j, x[j], rng):
            key = new.tobytes()
            delta = new.astype(float) - x[j]
            u = usage + S.draw[j] * delta
            pw_new, bat_new = S.global_violation(u)
            jvj = base_jv.copy()
            jvj[j] = S.job_violation(j, new)
            obj = objective + S.priority[j] * float(delta.sum())
            score = penalty(jvj, pw_new, bat_new, obj)
            aspiration = (
                not jvj.any()
                and not pw_new.any()
                and not bat_new.any()
                and obj > best_obj + 1e-9
            )
            if key in tabu[j] and not aspiration:
                continue
            if score < best_score or aspiration:
                best_move, best_score = (new, jvj[j], u, pw_new, bat_new, obj), score
                if aspiration:
                    break
        if best_move is None:
            # Local minimum: raise weights of violated constraints (FJ rule).
            jw += jv > 0
            pw += power > 0
            bw += battery > 0
            if feasible_now() and target is not None:
                obj_w += 1.0
            current = penalty(jv, power, battery, objective)
            moves = _moves(S, j, x[j], rng)
            if not moves:
                continue
            new = moves[int(rng.integers(len(moves)))]
            delta = new.astype(float) - x[j]
            usage = usage + S.draw[j] * delta
            objective += S.priority[j] * float(delta.sum())
            tabu[j].append(x[j].tobytes())
            x[j] = new
            jv[j] = S.job_violation(j, new)
            power, battery = S.global_violation(usage)
        else:
            new, jvj, usage, power, battery, objective = best_move
            tabu[j].append(x[j].tobytes())
            x[j] = new
            jv[j] = jvj
        current = penalty(jv, power, battery, objective)
        if feasible_now() and objective > best_obj + 1e-9:
            accept_best()
        if best is not None and since > restart_after:
            # Restart near the best solution with a small random perturbation.
            x = best.copy()
            for jj in rng.choice(J, size=min(2, J), replace=False):
                moves = _moves(S, int(jj), x[jj], rng)
                if moves:
                    x[jj] = moves[int(rng.integers(len(moves)))]
            jv = np.stack([S.job_violation(jj, x[jj]) for jj in range(J)])
            usage = S.draw @ x
            power, battery = S.global_violation(usage)
            objective = float(S.priority @ x.sum(axis=1))
            jw[:] = 1
            pw[:] = 1
            bw[:] = 1
            obj_w = 1.0
            since = 0
            current = penalty(jv, power, battery, objective)
    return (
        None if best is None else torch.from_numpy(best),
        best_obj if best is not None else None,
        steps,
    )


def solve_local_search(
    instance,
    *,
    time_limit=30.0,
    seed=0,
    config=None,
    initial_schedule=None,
):
    """Cold-start (no MILP) or polishing local search under a wall deadline."""
    if not math.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Positive finite time limit required")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, time_limit * 0.05)
    problem = ONTSMilp(instance, config)
    rng = np.random.default_rng(seed)
    history, first = [], None
    start = None
    if initial_schedule is not None:
        report = problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("Infeasible initial schedule")
        start = initial_schedule.detach().cpu().bool().numpy()

    def improved(schedule, value):
        nonlocal first
        now = perf_counter() - started
        if first is None:
            first = now
        history.append(dict(time=now, event="improvement", objective=value))

    schedule, objective, steps = weighted_local_search(
        problem, start, rng, deadline=deadline, on_improvement=improved
    )
    if (
        schedule is not None
        and not problem.config.validate(instance, schedule).feasible
    ):
        raise RuntimeError("Local search returned an infeasible schedule")
    return HybridResult(
        "feasible" if schedule is not None else "unknown",
        schedule,
        objective,
        None,
        None,
        perf_counter() - started,
        first,
        steps,
        history,
    )
