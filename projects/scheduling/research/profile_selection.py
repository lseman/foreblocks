"""Select the profile-portfolio configuration on development instances only.

Uses the heuristics_dev instance set (never the confirmatory test set). The
criterion, fixed before running: highest mean of the IID and OOD mean relative
objective change versus `heuristic`; ties prefer the earlier configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..hybrid import solve_hybrid
from ..profile_portfolio import solve_profile_portfolio
from .heuristics_dev import RESULTS, instances
from .protocol import objectives_at_budget

CONFIGS = [
    dict(controller="stagnation", patience=p, job_selection=s)
    for p in (10, 30, 100)
    for s in ("rules", "adaptive")
] + [
    dict(controller="ucb", patience=30, job_selection=s) for s in ("rules", "adaptive")
]


def name(config):
    return f"{config['controller']}-p{config['patience']}-{config['job_selection']}"


def run(task):
    item, config, seed, budget, repair = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    common = dict(
        initial_schedule=item["witness"],
        time_limit=budget,
        repair_time=repair,
        seed=seed,
    )
    if config is None:
        result = solve_hybrid(
            item["instance"],
            selector="heuristic",
            initial_exact=False,
            neighborhood_fraction=0.5,
            **common,
        )
    else:
        result = solve_profile_portfolio(
            item["instance"], fraction=0.2, **config, **common
        )
    return dict(
        instance_id=item["id"],
        method="heuristic" if config is None else name(config),
        seed=seed,
        objective=objectives_at_budget(result.to_dict(), budget),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "profile-selection")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    test = instances("iid", 3, False, 100) + instances("ood", 2, True, 200)
    tasks = [
        (item, config, seed, 3.0, 0.3)
        for item in test
        for seed in (0, 1, 2)
        for config in [None, *CONFIGS]
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    by = {}
    for row in rows:
        by.setdefault(row["instance_id"], {}).setdefault(row["method"], []).append(
            row["objective"] or 0.0
        )
    scores = {}
    for config in CONFIGS:
        split_means = []
        for split in ("iid", "ood"):
            rel = [
                (np.mean(m[name(config)]) - np.mean(m["heuristic"]))
                / max(abs(np.mean(m["heuristic"])), 1e-9)
                for i, m in by.items()
                if i.startswith(split)
            ]
            split_means.append(100 * float(np.mean(rel)))
        scores[name(config)] = dict(iid=split_means[0], ood=split_means[1])
        scores[name(config)]["criterion"] = float(np.mean(split_means))
    selected = max(CONFIGS, key=lambda c: scores[name(c)]["criterion"])
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    (args.output / "selection.json").write_text(
        json.dumps(dict(selected=selected, scores=scores), indent=2) + "\n"
    )
    for key, value in scores.items():
        print(f"{key:28s} iid {value['iid']:+6.2f}%  ood {value['ood']:+6.2f}%")
    print("selected", selected)


if __name__ == "__main__":
    main()
