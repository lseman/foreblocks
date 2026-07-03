"""Generate feasible ONTS training instances.

The generator samples the parameter ranges used in the original ONTS scripts,
checks feasibility with SciPy's MILP solver, and writes JSON instances that can
be loaded by ``ONTSEnv``.

Usage:
    python instances/generate_onts_instances.py --count 100 --out instances/generated
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

INSTANCE_FORMAT = "onts-instance-v1"
INSTANCE_FIELDS = (
    "priority",
    "uso_p",
    "min_statup",
    "max_statup",
    "min_cpu_time",
    "max_cpu_time",
    "min_periodo_job",
    "max_periodo_job",
    "win_min",
    "win_max",
    "recurso_p",
)


@dataclass
class FeasibilityResult:
    feasible: bool
    status: int
    message: str
    solver: str = ""


def generate_candidate(*, J: int, T: int, rng: random.Random, name: str) -> dict[str, Any]:
    """Generate one random ONTS candidate using the requested ranges."""
    min_statup = [rng.randint(1, math.ceil(T / 45)) for _ in range(J)]
    max_statup = [rng.randint(min_statup[j], math.ceil(T / 15)) for j in range(J)]
    min_cpu_time = [rng.randint(1, math.ceil(T / 10)) for _ in range(J)]
    max_cpu_time = [
        rng.randint(min_cpu_time[j], math.ceil(T / 4)) for j in range(J)
    ]
    min_periodo_job = [
        rng.randint(min_cpu_time[j], math.ceil(T / 4)) for j in range(J)
    ]
    max_periodo_job = [rng.randint(min_periodo_job[j], T) for j in range(J)]
    win_min = [rng.randint(0, math.ceil(T / 5)) for _ in range(J)]
    win_max = [rng.randint(T - math.ceil(T / 5), T) for _ in range(J)]

    return {
        "format": INSTANCE_FORMAT,
        "name": name,
        "jobs": J,
        "tamanho": T,
        "recurso_p": [round(rng.uniform(10, 15), 6) for _ in range(T)],
        "priority": [round(rng.uniform(1, 3), 6) for _ in range(J)],
        "uso_p": [round(rng.uniform(0.3, 2.5), 6) for _ in range(J)],
        "min_statup": min_statup,
        "max_statup": max_statup,
        "min_cpu_time": min_cpu_time,
        "max_cpu_time": max_cpu_time,
        "min_periodo_job": min_periodo_job,
        "max_periodo_job": max_periodo_job,
        "win_min": win_min,
        "win_max": win_max,
        "generator": "random-onts-v1",
    }


def _feasibility_milp(
    data: dict[str, Any],
    *,
    soc_inicial: float,
    limite_inferior: float,
    ef: float,
    v_bat: float,
    q: float,
    bat_usage: float,
    include_valid_inequalities: bool,
    time_limit: float,
) -> FeasibilityResult:
    J = int(data["jobs"])
    T = int(data["tamanho"])
    uso_p = data["uso_p"]
    min_statup = data["min_statup"]
    max_statup = data["max_statup"]
    min_cpu_time = data["min_cpu_time"]
    max_cpu_time = data["max_cpu_time"]
    min_periodo_job = data["min_periodo_job"]
    max_periodo_job = data["max_periodo_job"]
    win_min = data["win_min"]
    win_max = data["win_max"]
    recurso_p = data["recurso_p"]

    if any(len(data[field]) != (T if field == "recurso_p" else J) for field in INSTANCE_FIELDS):
        return FeasibilityResult(False, -2, "invalid instance dimensions")

    n_x = J * T
    n_vars = 2 * n_x

    def x(j: int, t: int) -> int:
        return j * T + t

    def phi(j: int, t: int) -> int:
        return n_x + j * T + t

    rows: list[dict[int, float]] = []
    lb: list[float] = []
    ub: list[float] = []

    def add(coeffs: dict[int, float], lower: float, upper: float) -> None:
        rows.append(coeffs)
        lb.append(lower)
        ub.append(upper)

    for j in range(J):
        for t in range(T):
            prev = x(j, t - 1) if t > 0 else None
            coeff = {phi(j, t): 1.0, x(j, t): -1.0}
            if prev is not None:
                coeff[prev] = 1.0
            add(coeff, 0.0, math.inf)
            add({phi(j, t): 1.0, x(j, t): -1.0}, -math.inf, 0.0)
            coeff = {phi(j, t): 1.0, x(j, t): 1.0}
            if prev is not None:
                coeff[prev] = 1.0
            add(coeff, -math.inf, 2.0)

        add({phi(j, t): 1.0 for t in range(T)}, min_statup[j], max_statup[j])
        for t in range(0, min(win_min[j], T)):
            add({x(j, t): 1.0}, 0.0, 0.0)
        for t in range(max(0, win_max[j]), T):
            add({x(j, t): 1.0}, 0.0, 0.0)

        for t in range(0, T - min_periodo_job[j] + 1):
            add(
                {phi(j, tt): 1.0 for tt in range(t, t + min_periodo_job[j])},
                -math.inf,
                1.0,
            )
        for t in range(0, T - max_periodo_job[j] + 1):
            add(
                {phi(j, tt): 1.0 for tt in range(t, t + max_periodo_job[j])},
                1.0,
                math.inf,
            )
        for t in range(0, T - min_cpu_time[j] + 1):
            coeff = {x(j, tt): 1.0 for tt in range(t, t + min_cpu_time[j])}
            coeff[phi(j, t)] = coeff.get(phi(j, t), 0.0) - min_cpu_time[j]
            add(coeff, 0.0, math.inf)
        for t in range(0, T - max_cpu_time[j]):
            add(
                {x(j, tt): 1.0 for tt in range(t, t + max_cpu_time[j] + 1)},
                -math.inf,
                max_cpu_time[j],
            )
        for t in range(T - min_cpu_time[j] + 1, T):
            coeff = {x(j, tt): 1.0 for tt in range(t, T)}
            coeff[phi(j, t)] = coeff.get(phi(j, t), 0.0) - (T - t)
            add(coeff, 0.0, math.inf)

        if include_valid_inequalities:
            for t in range(T):
                add(
                    {
                        phi(j, tt): 1.0
                        for tt in range(t, min(T, t + min_cpu_time[j] + 1))
                    },
                    -math.inf,
                    1.0,
                )
            coeff = {x(j, t): 1.0 for t in range(T)}
            for t in range(T):
                coeff[phi(j, t)] = coeff.get(phi(j, t), 0.0) - max_cpu_time[j]
            add(coeff, -math.inf, 0.0)
            for t in range(0, T - max_cpu_time[j]):
                coeff = {x(j, tt): 1.0 for tt in range(t, t + max_cpu_time[j])}
                lo = max(0, t - max_cpu_time[j] + 1)
                hi = min(T, t + max_cpu_time[j])
                for tt in range(lo, hi):
                    coeff[phi(j, tt)] = coeff.get(phi(j, tt), 0.0) - max_cpu_time[j]
                add(coeff, -math.inf, 0.0)
            for t in range(0, T - min_periodo_job[j] + 1):
                add(
                    {x(j, tt): 1.0 for tt in range(t, t + min_periodo_job[j])},
                    -math.inf,
                    min_periodo_job[j],
                )
            if max_cpu_time[j] < (max_periodo_job[j] - min_cpu_time[j]):
                for t in range(0, T - max_cpu_time[j]):
                    add({phi(j, t): 1.0, x(j, t + max_cpu_time[j]): 1.0}, -math.inf, 1.0)

    for t in range(T):
        add(
            {x(j, t): float(uso_p[j]) for j in range(J)},
            -math.inf,
            float(recurso_p[t] + bat_usage * v_bat),
        )

    soc_coef = ef / (q * v_bat * 60.0)
    cumulative_solar = 0.0
    cumulative_usage_coeffs: dict[int, float] = {}
    for t in range(T):
        cumulative_solar += float(recurso_p[t])
        for j in range(J):
            cumulative_usage_coeffs[x(j, t)] = (
                cumulative_usage_coeffs.get(x(j, t), 0.0)
                - soc_coef * float(uso_p[j])
            )
        base = soc_inicial + soc_coef * cumulative_solar
        add(dict(cumulative_usage_coeffs), limite_inferior - base, 1.0 - base)

    pulp_result = _solve_with_pulp(rows, lb, ub, n_vars, time_limit)
    if pulp_result.solver:
        return pulp_result

    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import lil_matrix
    except Exception as exc:
        return FeasibilityResult(False, -1, f"no MILP solver available: {exc}", "")

    A = lil_matrix((len(rows), n_vars), dtype=float)
    for i, row in enumerate(rows):
        for col, value in row.items():
            A[i, col] = value

    result = milp(
        c=np.zeros(n_vars),
        integrality=np.ones(n_vars),
        bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
        constraints=LinearConstraint(A.tocsr(), np.array(lb), np.array(ub)),
        options={"time_limit": time_limit, "mip_rel_gap": 0.0, "disp": False},
    )
    return FeasibilityResult(
        bool(result.success),
        int(result.status),
        str(result.message),
        "scipy.optimize.milp",
    )


def _solve_with_pulp(
    rows: list[dict[int, float]],
    lb: list[float],
    ub: list[float],
    n_vars: int,
    time_limit: float,
) -> FeasibilityResult:
    try:
        import pulp
    except Exception as exc:
        return FeasibilityResult(False, -10_000, f"pulp unavailable: {exc}", "")

    prob = pulp.LpProblem("onts_feasibility", pulp.LpMinimize)
    var = [
        pulp.LpVariable(f"z_{idx}", lowBound=0, upBound=1, cat=pulp.LpBinary)
        for idx in range(n_vars)
    ]
    prob += 0
    for row_id, coeffs in enumerate(rows):
        expr = pulp.lpSum(value * var[col] for col, value in coeffs.items())
        if math.isfinite(lb[row_id]):
            prob += expr >= lb[row_id], f"lb_{row_id}"
        if math.isfinite(ub[row_id]):
            prob += expr <= ub[row_id], f"ub_{row_id}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit)
    status = prob.solve(solver)
    status_name = pulp.LpStatus.get(status, str(status))
    return FeasibilityResult(
        status_name == "Optimal",
        int(status),
        status_name,
        "pulp.PULP_CBC_CMD",
    )


def generate_feasible_instances(
    *,
    count: int,
    out_dir: Path,
    J: int,
    T: int,
    seed: int,
    max_attempts: int,
    time_limit: float,
    include_valid_inequalities: bool,
) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    written = 0
    attempts = 0
    while written < count and attempts < max_attempts:
        attempts += 1
        name = f"gen_{J}_{T}_{seed}_{attempts:05d}"
        data = generate_candidate(J=J, T=T, rng=rng, name=name)
        result = _feasibility_milp(
            data,
            soc_inicial=0.7,
            limite_inferior=0.0,
            ef=0.9,
            v_bat=3.6,
            q=5.0,
            bat_usage=5.0,
            include_valid_inequalities=include_valid_inequalities,
            time_limit=time_limit,
        )
        if not result.feasible:
            print(f"reject {name}: {result.message}")
            continue
        data["feasibility"] = {
            "solver": result.solver,
            "status": result.status,
            "message": result.message,
            "valid_inequalities": include_valid_inequalities,
        }
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        written += 1
        print(f"wrote {path}")
    return written, attempts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("instances/generated"))
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--horizon", type=int, default=97)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--time-limit", type=float, default=5.0)
    parser.add_argument(
        "--no-valid-inequalities",
        action="store_true",
        help="Check feasibility without the valid inequalities from original.py",
    )
    args = parser.parse_args()
    written, attempts = generate_feasible_instances(
        count=args.count,
        out_dir=args.out,
        J=args.jobs,
        T=args.horizon,
        seed=args.seed,
        max_attempts=args.max_attempts,
        time_limit=args.time_limit,
        include_valid_inequalities=not args.no_valid_inequalities,
    )
    print(f"generated={written} attempts={attempts} out={args.out}")
    if written < args.count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
