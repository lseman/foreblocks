"""Anytime incumbent histories on the bundled instances (descriptive).

Runs each method once per instance without a warm start and stores every
incumbent improvement as (time, objective). Outputs are checked by the oracle.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from projects.scheduling.environments.onts_env import (
    load_onts_instance,
    onts_instance_to_json_data,
)
from projects.scheduling.hybrid import solve_hybrid
from projects.scheduling.local_search import solve_local_search
from projects.scheduling.milp import ONTSConfig, ONTSMilp
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.ranked_search import solve_ranked
from projects.scheduling.research.oracle import oracle_check

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("v2", "v1", "heuristic", "uniform", "local_search", "exact")
V1 = dict(controller="stagnation", patience=10, job_selection="rules", fraction=0.2)
V2 = dict(V1, polish_after_feasible=0.5, surplus_weight=0.05)


def run(task):
    path, method, budget, repair, seed = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    instance = load_onts_instance(path)
    if method == "exact":
        result = ONTSMilp(instance).solve(time_limit=budget)
        schedule = result.schedule
        objective = result.objective
        history = [(result.runtime, objective)] if schedule is not None else []
        bound = result.upper_bound
    else:
        if method in ("v1", "v2"):
            result = solve_profile_portfolio(
                instance,
                time_limit=budget,
                repair_time=repair,
                seed=seed,
                **(V2 if method == "v2" else V1),
            )
        elif method == "heuristic":
            result = solve_hybrid(
                instance,
                selector="heuristic",
                time_limit=budget,
                repair_time=repair,
                neighborhood_fraction=0.5,
                seed=seed,
            )
        elif method == "uniform":
            result = solve_ranked(
                instance,
                mode="uniform",
                time_limit=budget,
                repair_time=repair,
                fraction=0.2,
                seed=seed,
            )
        else:
            result = solve_local_search(instance, time_limit=budget, seed=seed)
        schedule, objective, bound = result.schedule, result.objective, None
        best, history = -np.inf, []
        for event in result.history:
            value = event.get("objective")
            if value is not None and value > best + 1e-9:
                best = value
                history.append((event["time"], value))
    if schedule is not None:
        valid, checked, _ = oracle_check(
            onts_instance_to_json_data(instance), schedule.int().tolist(), ONTSConfig()
        )
        if not valid or not np.isclose(checked, objective, atol=1e-4):
            raise RuntimeError(f"Validation failed: {path.name} {method}")
    return dict(
        instance=path.name,
        method=method,
        J=instance.J,
        T=instance.T,
        # Same 0.5 s solver-overrun tolerance as the confirmatory studies.
        history=[
            (float(min(t, budget)), float(v)) for t, v in history if t <= budget + 0.5
        ],
        upper_bound=bound,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/results/anytime-bundled"
    )
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--repair-time", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=list(METHODS))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "instances/examples").glob("*.json"))
    tasks = [
        (p, m, args.budget, args.repair_time, args.seed)
        for p in paths
        for m in args.methods
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    path = args.output / "runs.json"
    if path.exists() and set(args.methods) != set(METHODS):
        # Replace only the re-run methods in an existing result file.
        kept = [
            r for r in json.loads(path.read_text()) if r["method"] not in args.methods
        ]
        rows = kept + rows
    path.write_text(json.dumps(rows, indent=1) + "\n")
    print(f"wrote {len(rows)} runs")


if __name__ == "__main__":
    main()
