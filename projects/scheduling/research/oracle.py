"""Independent tiny-instance oracle based on runs and gap endpoints."""

from __future__ import annotations

from dataclasses import asdict
from itertools import pairwise, product
from time import perf_counter

import numpy as np
import torch

from ..environments.onts_env import ONTSInstance, onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp


def oracle_check(data: dict, rows: list[list[int]], config: ONTSConfig):
    """Return feasibility, objective and maximal energy trajectory.

    Unlike the production checker, periodicity is tested with sentinel gaps
    and run lengths are recovered from change points, not while loops.
    """
    T, J = len(data["recurso_p"]), len(data["priority"])
    valid = True
    for j, row in enumerate(rows):
        boundaries = [
            t
            for t in range(T + 1)
            if (row[t] if t < T else 0) != (row[t - 1] if t else 0)
        ]
        runs = list(zip(boundaries[::2], boundaries[1::2]))
        starts = [start for start, _ in runs]
        valid &= data["min_statup"][j] <= len(starts) <= data["max_statup"][j]
        valid &= all(
            data["win_min"][j] <= start and end <= data["win_max"][j]
            for start, end in runs
        )
        valid &= all(b - a >= data["min_periodo_job"][j] for a, b in pairwise(starts))
        period = data["max_periodo_job"][j]
        if period <= T:
            # A full period-size startup-free window exists iff a sentinel gap
            # exceeds period (right sentinel T; left sentinel -1).
            valid &= max(b - a for a, b in pairwise([-1, *starts, T])) <= period
        for start, end in runs:
            minimum = data["min_cpu_time"][j]
            if config.allow_truncated_final_run:
                minimum = min(minimum, T - start)
            valid &= minimum <= end - start <= data["max_cpu_time"][j]
    energy, trace = config.soc_inicial, []
    objective = sum(data["priority"][j] * sum(rows[j]) for j in range(J))
    for t in range(T):
        usage = sum(data["uso_p"][j] * rows[j][t] for j in range(J))
        valid &= usage <= data["recurso_p"][t] + config.bat_usage * config.v_bat + 1e-6
        energy = min(
            1,
            energy
            + (data["recurso_p"][t] - usage)
            * config.ef
            / (config.q * config.v_bat * 60),
        )
        trace.append(energy)
        valid &= energy >= config.limite_inferior - 1e-6
    return bool(valid), float(objective), trace


def audit_instance(instance: ONTSInstance, config: ONTSConfig, *, max_bits: int = 16):
    """Enumerate ALL schedules; compare oracle, checker, matrix and optimum."""
    if max_bits < instance.J * instance.T:
        raise ValueError("Exhaustive audit exceeds configured bit limit")
    started = perf_counter()
    data = onts_instance_to_json_data(instance)
    p = ONTSMilp(instance, config)
    feasible = 0
    best = None
    disagreements = []
    for bits in product((0, 1), repeat=p.n_x):
        x = np.asarray(bits).reshape(instance.J, instance.T)
        expected, objective, trace = oracle_check(data, x.tolist(), config)
        previous = np.concatenate([np.zeros((p.J, 1)), x[:, :-1]], axis=1)
        assignment = np.concatenate([x.ravel(), (x > previous).ravel(), trace])
        activity = p.A @ assignment
        matrix_feasible = bool(
            np.all(assignment >= p.lower - 1e-6)
            and np.all(assignment <= p.upper + 1e-6)
            and np.all(activity >= p.row_lower - 1e-6)
            and np.all(activity <= p.row_upper + 1e-6)
        )
        report = config.validate(instance, torch.from_numpy(x))
        if (
            expected != report.feasible
            or expected != matrix_feasible
            or abs(objective - report.objective) > 1e-6
        ):
            disagreements.append(
                {
                    "schedule": x.tolist(),
                    "oracle": expected,
                    "validator": report.feasible,
                    "matrix": matrix_feasible,
                    "violations": list(report.violations),
                }
            )
        if expected:
            feasible += 1
            best = max(best, objective) if best is not None else objective
    solved = p.solve(time_limit=5)
    optimum_matches = (best is None and solved.status == "infeasible") or (
        best is not None
        and solved.status == "optimal"
        and abs(solved.objective - best) <= 1e-6
    )
    return {
        "jobs": p.J,
        "horizon": p.T,
        "config": asdict(config),
        "schedules": 2**p.n_x,
        "feasible_schedules": feasible,
        "oracle_optimum": best,
        "solver_objective": solved.objective,
        "solver_status": solved.status,
        "optimum_matches": optimum_matches,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements[:5],
        "runtime": perf_counter() - started,
    }


def random_tiny_instance(seed: int):
    rng = np.random.default_rng(seed)
    J, T = ((1, 6), (2, 4), (3, 3))[seed % 3]

    def ints(lo, hi):
        return rng.integers(lo, hi + 1, size=J)

    minimum = ints(1, T + 1)
    maximum = minimum + ints(0, 2)
    period_min = ints(1, T + 2)
    period_max = period_min + ints(0, T)
    starts_min = ints(0, 2)
    starts_max = starts_min + ints(0, 3)
    wmin = ints(0, T)
    wmax = np.array([rng.integers(v, T + 3) for v in wmin])
    solar = rng.choice([0.0, 1.0, 5.0, 50.0], size=T)
    instance = ONTSInstance(
        priority=torch.tensor(rng.integers(-2, 6, size=J)).double(),
        uso_p=torch.tensor(rng.choice([0.0, 2.0, 10.0, 25.0], size=J)),
        min_statup=torch.tensor(starts_min),
        max_statup=torch.tensor(starts_max),
        min_cpu_time=torch.tensor(minimum),
        max_cpu_time=torch.tensor(maximum),
        min_periodo_job=torch.tensor(period_min),
        max_periodo_job=torch.tensor(period_max),
        win_min=torch.tensor(wmin),
        win_max=torch.tensor(wmax),
        recurso_p=torch.tensor(solar),
    )
    config = ONTSConfig(
        soc_inicial=float(rng.choice([0.0, 0.01, 0.7, 1.0])),
        q=float(rng.choice([0.05, 5.0])),
        bat_usage=float(rng.choice([0.0, 5.0])),
        allow_truncated_final_run=bool(seed % 2),
    )
    return instance, config


def run_exhaustive_audit(cases: int = 120, seed: int = 7000):
    results = [audit_instance(*random_tiny_instance(seed + i)) for i in range(cases)]
    return {
        "format": "onts-exhaustive-audit-v1",
        "cases": cases,
        "seed": seed,
        "schedules": sum(r["schedules"] for r in results),
        "disagreement_count": sum(r["disagreement_count"] for r in results),
        "optimum_mismatches": sum(not r["optimum_matches"] for r in results),
        "all_passed": all(
            not r["disagreement_count"] and r["optimum_matches"] for r in results
        ),
        "results": results,
    }
