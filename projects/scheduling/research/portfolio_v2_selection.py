"""Select portfolio v2 options on development instances only.

Criterion fixed before running: maximize the mean normalized score (objective /
best found by any configuration on that instance; infeasible = 0) on bundled-like
development instances, subject to no more than a 0.25% mean loss versus v1 on the
planted development instances. Ties prefer the earlier (simpler) configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..profile_portfolio import solve_profile_portfolio
from .heuristics_dev import RESULTS, instances
from .protocol import objectives_at_budget
from .realistic import realistic_instance

V1 = dict(controller="stagnation", patience=10, job_selection="rules")
CONFIGS = {
    "v1": {},
    "early": dict(polish_after_feasible=0.5),
    "early+adaptive": dict(polish_after_feasible=0.5, adaptive_fraction=True),
    "early+equal": dict(polish_after_feasible=0.5, accept_equal=True),
    "early+tiebreak": dict(polish_after_feasible=0.5, surplus_weight=1e-6),
    "early+surplus": dict(polish_after_feasible=0.5, surplus_weight=0.05),
    "all-tiebreak": dict(
        polish_after_feasible=0.5,
        adaptive_fraction=True,
        accept_equal=True,
        surplus_weight=1e-6,
    ),
    "all-surplus": dict(
        polish_after_feasible=0.5,
        adaptive_fraction=True,
        accept_equal=True,
        surplus_weight=0.05,
    ),
}
REALISTIC_SEED = 52_000_000


def run(task):
    kind, item, name, seed, budget, repair = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    kwargs = dict(
        **V1, **CONFIGS[name], time_limit=budget, repair_time=repair, seed=seed
    )
    if kind == "planted":
        kwargs["initial_schedule"] = item["witness"]
    result = solve_profile_portfolio(item["instance"], fraction=0.2, **kwargs)
    return dict(
        kind=kind,
        instance_id=item["id"],
        config=name,
        seed=seed,
        objective=objectives_at_budget(result.to_dict(), budget),
        first=result.time_to_first_feasible,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=RESULTS / "portfolio-v2-selection"
    )
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    planted = instances("iid", 3, False, 100) + instances("ood", 2, True, 200)
    realistic = [
        dict(
            id=f"realistic:{REALISTIC_SEED + i}",
            instance=realistic_instance(REALISTIC_SEED + i),
        )
        for i in range(12)
    ]
    tasks = [
        ("planted", item, name, seed, 3.0, 0.3)
        for item in planted
        for seed in (0, 1, 2)
        for name in CONFIGS
    ] + [
        ("realistic", item, name, seed, 20.0, 1.0)
        for item in realistic
        for seed in (0, 1)
        for name in CONFIGS
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    scores = {name: {} for name in CONFIGS}
    for kind in ("planted", "realistic"):
        by = {}
        for r in rows:
            if r["kind"] == kind:
                by.setdefault(r["instance_id"], {}).setdefault(r["config"], []).append(
                    r["objective"] or 0.0
                )
        for name in CONFIGS:
            values = []
            for methods in by.values():
                mine = np.mean(methods[name])
                if kind == "planted":
                    base = np.mean(methods["v1"])
                    values.append(100 * (mine - base) / max(abs(base), 1e-9))
                else:
                    best = max(np.mean(v) for v in methods.values())
                    values.append(mine / best if best > 0 else 1.0)
            scores[name][kind] = float(np.mean(values))
        if kind == "realistic":
            for name in CONFIGS:
                firsts = [
                    r["first"]
                    for r in rows
                    if r["kind"] == kind and r["config"] == name and r["first"]
                ]
                scores[name]["median_first_feasible"] = float(np.median(firsts))
    eligible = [n for n in CONFIGS if scores[n]["planted"] >= -0.25]
    selected = max(eligible, key=lambda n: scores[n]["realistic"])
    (args.output / "selection.json").write_text(
        json.dumps(
            dict(selected=selected, config=CONFIGS[selected], scores=scores), indent=2
        )
        + "\n"
    )
    for name, value in scores.items():
        print(
            f"{name:16s} planted vs v1 {value['planted']:+6.2f}%  realistic "
            f"{100 * value['realistic']:6.2f}%  first {value['median_first_feasible']:.2f}s"
        )
    print("selected", selected, CONFIGS[selected])


if __name__ == "__main__":
    main()
