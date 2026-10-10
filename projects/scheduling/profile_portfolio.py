"""Cheap-first portfolio: whole-job profile moves plus uniform MILP repair.

Profile moves cost milliseconds and need no solver; MILP repairs of a uniformly
chosen candidate mask cost a solver call but can change many jobs jointly.
Controllers:
  stagnation: profile moves until `patience` consecutive failures, then one repair.
  ucb: UCB over the two operators on normalized gain per second.
Job selection for profile moves:
  rules: cycle battery-pressure / utility / random rules.
  adaptive: UCB over jobs on realized gain when released; the remaining jobs of a
  multi-job move use the pressure rule. Learned online, per run; no training.
Without an initial schedule, weighted local search runs first (default) because
exact HiGHS starts timed out on bundled benchmark instances; exact solving and a
retry remain as fallbacks.

repair_strategy="whole_jobs" releases ceil(fraction * J) complete legal rows,
cycling pressure, utility and random job selection. The default "uniform" uses
the original candidate masks. "time_window" releases ceil(fraction * T)
consecutive slots across jobs, cycling pressure, spill and uniform placement.
surplus_mode is passed to the profile operator.

Selective repair gates are opt-in: repair_start_fraction delays calls relative
to this invocation's allowance; repair_budget_fraction targets a cumulative
repair-time quota; repair_min_remaining reserves a DP tail. Preparation and
solver return overhead can overshoot the quota, which is not a hard timer.
"""

from __future__ import annotations

import math
from time import perf_counter

import numpy as np
import torch

from .hybrid import HybridResult
from .job_profile import battery_trace, curtailed_power, reprofile, select_jobs
from .local_search import weighted_local_search
from .milp import ONTSMilp
from .ranking import SearchState, make_candidates

RULES = ("pressure", "utility", "random")


class JobBandit:
    def __init__(self, J, rng):
        self.pulls, self.gains, self.rng = np.zeros(J), np.zeros(J), rng

    def choose(self, problem, incumbent, count):
        total = self.pulls.sum()
        value = self.gains / np.maximum(self.pulls, 1) + np.sqrt(
            2 * np.log(2 + total) / np.maximum(self.pulls, 1)
        )
        value[self.pulls == 0] = np.inf
        value = value + 1e-9 * self.rng.random(len(value))
        first = int(np.argmax(value))
        jobs = [first]
        if count > 1:
            extra = select_jobs(problem, incumbent, self.rng, count, rule="pressure")
            jobs += [j for j in extra if j != first][: count - 1]
        return jobs

    def update(self, jobs, reward):
        for j in jobs:
            self.pulls[j] += 1
            self.gains[j] += reward


def solve_profile_portfolio(
    instance,
    *,
    controller="stagnation",
    job_selection="adaptive",
    patience=30,
    max_jobs=3,
    time_limit=30.0,
    repair_time=1.0,
    fraction=0.2,
    candidate_count=8,
    init="local_search",
    init_fraction=0.3,
    polish_after_feasible=None,
    adaptive_fraction=False,
    surplus_weight=0.0,
    accept_equal=False,
    seed=0,
    config=None,
    initial_schedule=None,
    max_iterations=10**6,
    enable_repairs=True,
    enable_profiles=True,
    observe_solver_incumbents=False,
    repair_strategy="uniform",
    surplus_mode="saturation",
    repair_start_fraction=0.0,
    repair_budget_fraction=1.0,
    repair_min_remaining=0.0,
):
    if controller not in ("stagnation", "ucb"):
        raise ValueError("controller must be 'stagnation' or 'ucb'")
    if not enable_profiles and not enable_repairs:
        raise ValueError("At least one improvement operator must be enabled")
    if job_selection not in ("rules", "adaptive"):
        raise ValueError("job_selection must be 'rules' or 'adaptive'")
    if init not in ("exact", "local_search"):
        raise ValueError("init must be 'exact' or 'local_search'")
    if repair_strategy not in ("uniform", "whole_jobs", "time_window"):
        raise ValueError("repair_strategy must be 'uniform', 'whole_jobs' or 'time_window'")
    if surplus_mode not in ("saturation", "curtailment"):
        raise ValueError("surplus_mode must be 'saturation' or 'curtailment'")
    if not math.isfinite(surplus_weight) or surplus_weight < 0:
        raise ValueError("surplus_weight must be finite and nonnegative")
    if not all(math.isfinite(v) and v > 0 for v in (time_limit, repair_time)):
        raise ValueError("Positive finite time limits required")
    if patience < 1 or max_jobs < 1 or not 0 < fraction <= 1:
        raise ValueError("Invalid patience, job count or fraction")
    if not (math.isfinite(repair_start_fraction) and 0 <= repair_start_fraction <= 1
            and math.isfinite(repair_budget_fraction) and 0 <= repair_budget_fraction <= 1
            and math.isfinite(repair_min_remaining) and repair_min_remaining >= 0):
        raise ValueError("Invalid repair scheduling fractions or remaining-time reserve")
    if not enable_profiles and (repair_start_fraction or repair_budget_fraction < 1
                                or repair_min_remaining):
        raise ValueError("Selective repairs require profile moves as a fallback")
    started = perf_counter()
    deadline = started + time_limit - min(0.25, time_limit * 0.05)

    def remaining():
        return deadline - perf_counter()

    problem = ONTSMilp(instance, config)
    rng = np.random.default_rng(seed)
    history, bound, first, status = [], None, None, "unknown"
    incumbent, objective = None, None
    first_source = None

    def initialized_improvement(schedule, value):
        nonlocal first, first_source
        now = perf_counter() - started
        if first is None:
            first, first_source = now, "local_search_validation"
        history.append(dict(time=now, event="initialization_incumbent", objective=value,
                            source="local_search_validation"))

    def initialize_exact(limit):
        nonlocal first, first_source
        solve_started = perf_counter() - started
        result = problem.solve(time_limit=max(1e-3, limit),
                               observe_incumbents=observe_solver_incumbents)
        for event in result.incumbent_history:
            observed = dict(event, time=solve_started + event["time"],
                            event="initialization_incumbent")
            history.append(observed)
            if first is None:
                first, first_source = observed["time"], observed["source"]
        if result.schedule is not None and first is None:
            first, first_source = perf_counter() - started, "solver_return"
        return result
    if initial_schedule is not None:
        report = problem.config.validate(instance, initial_schedule)
        if not report.feasible:
            raise ValueError("Infeasible initial schedule")
        incumbent = initial_schedule.detach().cpu().bool().clone()
        objective = report.objective
        first, first_source = 0.0, "provided_schedule"
        history.append(dict(time=0.0, event="initial_schedule", objective=objective))
    else:
        if init == "local_search":
            incumbent, objective, _ = weighted_local_search(
                problem,
                None,
                rng,
                deadline=perf_counter() + remaining() * init_fraction,
                polish_after_feasible=polish_after_feasible,
                on_improvement=initialized_improvement,
            )
            history.append(
                dict(
                    time=perf_counter() - started,
                    event="initial_local_search",
                    objective=objective,
                )
            )
        if incumbent is None and remaining() > 0:
            result = initialize_exact(remaining() * init_fraction)
            incumbent, objective, bound, status = (
                result.schedule,
                result.objective,
                result.upper_bound,
                result.status,
            )
            history.append(
                dict(
                    time=perf_counter() - started,
                    event="initial_exact",
                    objective=objective,
                )
            )
        if incumbent is None and init == "exact" and status != "infeasible":
            # Exact start timed out: try solver-free search, then retry exactly.
            incumbent, objective, _ = weighted_local_search(
                problem,
                None,
                rng,
                deadline=perf_counter() + remaining() / 2,
                polish_after_feasible=polish_after_feasible,
                on_improvement=initialized_improvement,
            )
            history.append(
                dict(
                    time=perf_counter() - started,
                    event="fallback_local_search",
                    objective=objective,
                )
            )
        if incumbent is None and remaining() > 0 and status != "infeasible":
            result = initialize_exact(remaining())
            incumbent, objective, status = (
                result.schedule,
                result.objective,
                result.status,
            )
            if result.upper_bound is not None:
                bound = (
                    result.upper_bound
                    if bound is None
                    else min(bound, result.upper_bound)
                )
            history.append(
                dict(
                    time=perf_counter() - started,
                    event="initial_retry",
                    objective=objective,
                )
            )
    initialization_complete_at = perf_counter() - started if incumbent is not None else None
    state = SearchState.empty(problem.J, problem.T)
    jobs_bandit = JobBandit(problem.J, rng)
    arms = ("profile", "repair")
    pulls, rewards = np.zeros(2), np.zeros(2)
    scale = max(float(instance.priority.sum()) * problem.T, 1.0)
    failures = 0
    iterations = 0
    repair_seconds = 0.0
    while incumbent is not None and remaining() > 0 and iterations < max_iterations:
        if bound is not None and bound - objective <= 1e-6:
            break
        if not enable_profiles:
            arm = 1
        elif not enable_repairs:
            arm = 0
        elif controller == "stagnation":
            arm = 1 if failures >= patience else 0
        else:
            value = rewards / np.maximum(pulls, 1) + np.sqrt(
                2 * np.log(2 + pulls.sum()) / np.maximum(pulls, 1)
            )
            value[pulls == 0] = np.inf
            arm = int(np.argmax(value))
        # Fractions refer to the solver's allowance, including model construction.
        # In shared-start studies the caller has already deducted initialization.
        repair_allowance = repair_budget_fraction * time_limit - repair_seconds
        if arm == 1 and enable_profiles and (
            perf_counter() - started < repair_start_fraction * time_limit
            or repair_allowance <= 1e-3
            or remaining() <= repair_min_remaining
        ):
            arm = 0
        move_started = perf_counter()
        old = objective
        info = {}
        if arms[arm] == "profile":
            count = 1 + min(max_jobs - 1, failures // max(1, patience // max_jobs))
            if job_selection == "adaptive":
                jobs = jobs_bandit.choose(problem, incumbent, count)
                rule = "adaptive"
            else:
                rule = RULES[iterations % len(RULES)]
                jobs = select_jobs(problem, incumbent, rng, count, rule=rule)
            schedule, value_, info = reprofile(
                problem,
                incumbent,
                jobs,
                rng,
                deadline=deadline,
                surplus_weight=surplus_weight,
                surplus_mode=surplus_mode,
                accept_equal=accept_equal,
            )
            mask = None
            info["rule"] = rule
        else:
            if repair_strategy == "whole_jobs":
                # Release complete legal rows so runs and startup spacing can change jointly.
                rule = RULES[int(pulls[1]) % len(RULES)]
                jobs = select_jobs(problem, incumbent, rng,
                                   max(1, math.ceil(problem.J * fraction)), rule=rule)
                legal = (problem.upper[:problem.n_x] > 0).reshape(problem.J, problem.T)
                free = np.zeros_like(legal)
                free[jobs] = legal[jobs]
                mask = torch.from_numpy(free)
                candidate_operator = "whole_jobs_" + rule
            elif repair_strategy == "time_window":
                # Jointly change competing jobs over one contiguous energy interval.
                width = max(1, math.ceil(problem.T * fraction))
                rule = ("pressure", "surplus", "random")[int(pulls[1]) % 3]
                usage = problem.instance.uso_p.cpu().double().numpy() @ incumbent.cpu().numpy()
                if rule == "pressure":
                    weights = 1.0 - battery_trace(problem, usage)
                elif rule == "surplus":
                    weights = curtailed_power(problem, usage)
                else:
                    weights = np.ones(problem.T)
                prefix = np.r_[0., np.cumsum(np.maximum(weights, 0.))]
                windows = prefix[width:] - prefix[:-width]
                windows = windows + max(float(windows.max()), 1.) * .05
                lo = int(rng.choice(len(windows), p=windows / windows.sum()))
                legal = (problem.upper[:problem.n_x] > 0).reshape(problem.J, problem.T)
                free = np.zeros_like(legal)
                free[:, lo:lo + width] = legal[:, lo:lo + width]
                mask = torch.from_numpy(free)
                candidate_operator = "time_window_" + rule
            else:
                candidates = make_candidates(
                    problem, incumbent, state, rng, fraction=fraction, count=candidate_count
                )
                candidate = candidates[int(rng.integers(len(candidates)))]
                mask = candidate.mask
                candidate_operator = candidate.operator
            # Mask construction is charged to the repair quota as well. It may
            # exhaust that quota before the solver is called.
            available = min(remaining() - repair_min_remaining,
                            repair_allowance - (perf_counter() - move_started))
            if available <= 1e-3:
                elapsed = perf_counter() - move_started
                repair_seconds += elapsed
                history.append(dict(time=perf_counter() - started, event="repair_skipped",
                                    operator="repair", objective=objective,
                                    solve_runtime=elapsed, reason="remaining_allowance"))
                iterations += 1
                continue
            solve_cap = min(available, repair_time)
            solver_started = perf_counter() - started
            result = problem.solve(
                incumbent=incumbent,
                free_mask=mask,
                time_limit=solve_cap,
            )
            schedule, value_ = result.schedule, result.objective
            info = dict(
                candidate_operator=candidate_operator,
                status=result.status,
                fraction=fraction,
                released_cells=int(mask.sum()),
                released_jobs=int(mask.any(dim=1).sum()),
                repair_time_limit=solve_cap,
                solver_started=solver_started,
                repair_seconds_before=repair_seconds,
            )
            if adaptive_fraction and value_ <= objective + 1e-9:
                # Timed out: smaller neighborhoods; proven exhausted: larger.
                if result.status == "optimal":
                    fraction = min(0.6, fraction * 1.3)
                else:
                    fraction = max(0.02, fraction * 0.7)
        elapsed = perf_counter() - move_started
        if arm == 1:
            repair_seconds += elapsed
        previous = incumbent
        incumbent, objective = schedule, value_
        gain = objective - old
        reward = min(1.0, max(0.0, gain / scale) / max(elapsed, 1e-4))
        pulls[arm] += 1
        rewards[arm] += reward
        if arms[arm] == "profile":
            jobs_bandit.update(info["jobs"], max(0.0, gain) / scale * 1e3)
        state.advance(
            mask if mask is not None else (previous != incumbent),
            previous,
            incumbent,
            gain,
            elapsed,
        )
        if gain > 1e-9:
            failures = 0
        elif arms[arm] == "repair":
            failures = patience // 2
        else:
            failures += 1
        history.append(
            dict(
                time=perf_counter() - started,
                event="repair",
                operator=arms[arm],
                objective=objective,
                restricted=True,
                solve_runtime=elapsed,
                **{k: v for k, v in info.items() if k != "gain"},
            )
        )
        iterations += 1
    if incumbent is not None:
        if not problem.config.validate(instance, incumbent).feasible:
            raise RuntimeError("Profile portfolio returned an infeasible schedule")
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
        initialization_complete_at=initialization_complete_at,
        first_feasible_source=first_source,
    )
