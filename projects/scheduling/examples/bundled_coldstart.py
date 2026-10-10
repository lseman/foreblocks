"""Cold-start benchmark on the bundled ONTS instances (descriptive, exploratory).

No witness is given: each method must find its own first schedule. The bundled
set is a fixed benchmark, not a random sample, and six of its instances were
inspected during development, so results are descriptive, not inferential.
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
from projects.scheduling.research.protocol import objectives_at_budget

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("exact", "heuristic", "uniform", "local_search", "profile_portfolio")


def run(task):
    path, method, budget, repair = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    instance = load_onts_instance(path)
    row = dict(instance=path.name, method=method, J=instance.J, T=instance.T)
    if method == "exact":
        result = ONTSMilp(instance).solve(time_limit=budget)
        payload = dict(
            schedule=None
            if result.schedule is None
            else result.schedule.int().tolist(),
            objective=result.objective,
            upper_bound=result.upper_bound,
            runtime=result.runtime,
            history=[],
            time_to_first_feasible=result.runtime
            if result.schedule is not None
            else None,
        )
        credited = result.objective if result.runtime <= budget * 1.05 else None
    else:
        if method == "heuristic":
            result = solve_hybrid(
                instance,
                selector="heuristic",
                time_limit=budget,
                repair_time=repair,
                neighborhood_fraction=0.5,
            )
        elif method == "uniform":
            result = solve_ranked(
                instance,
                mode="uniform",
                time_limit=budget,
                repair_time=repair,
                fraction=0.2,
            )
        elif method == "local_search":
            result = solve_local_search(instance, time_limit=budget)
        else:
            result = solve_profile_portfolio(
                instance,
                time_limit=budget,
                repair_time=repair,
                patience=10,
                job_selection="rules",
                fraction=0.2,
            )
        payload = result.to_dict()
        credited = objectives_at_budget(payload, budget)
    if payload["schedule"] is not None:
        valid, objective, _ = oracle_check(
            onts_instance_to_json_data(instance), payload["schedule"], ONTSConfig()
        )
        if not valid or not np.isclose(objective, payload["objective"], atol=1e-4):
            raise RuntimeError(f"Validation failed: {path.name} {method}")
    row.update(
        objective=credited,
        upper_bound=payload.get("upper_bound"),
        runtime=payload["runtime"],
        time_to_first_feasible=payload["time_to_first_feasible"],
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "research/results/bundled-coldstart"
    )
    parser.add_argument("--budget", type=float, default=30.0)
    parser.add_argument("--repair-time", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted((ROOT / "instances/examples").glob("*.json"))
    tasks = [(p, m, args.budget, args.repair_time) for p in paths for m in METHODS]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    by = {}
    for r in rows:
        by.setdefault(r["instance"], {})[r["method"]] = r
    bounds = {
        i: m["exact"]["upper_bound"]
        for i, m in by.items()
        if m["exact"]["upper_bound"] is not None
    }
    lines = [
        f"# Bundled cold-start benchmark ({len(by)} instances, {args.budget:g}s, "
        "one thread, no witness)",
        "",
        "Gap is relative to the best objective found by any method on that instance "
        "(not a proven optimum); infeasible-by-deadline counts as 100%.",
        "",
        "| Method | Feasible | Mean gap to best found | Best-found count | Median time to first feasible |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        gaps, feasible, best_count, ttf = [], 0, 0, []
        for _, methods in by.items():
            best = max((r["objective"] or 0.0) for r in methods.values())
            value = methods[method]["objective"]
            if value is not None:
                feasible += 1
                ttf.append(methods[method]["time_to_first_feasible"] or np.nan)
            gaps.append(
                100.0 if not value or best <= 0 else 100 * (best - value) / best
            )
            best_count += value is not None and value >= best - 1e-6
        lines.append(
            f"| {method} | {feasible}/{len(by)} | {np.mean(gaps):.2f}% | {best_count} | "
            f"{np.nanmedian(ttf) if ttf else float('nan'):.2f}s |"
        )
    lines += ["", f"Instances with a finite exact upper bound: {len(bounds)}."]
    (args.output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
