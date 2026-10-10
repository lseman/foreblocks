"""Development comparison of new ONTS heuristics (exploratory, not confirmatory).

Fresh planted instances from a seed range disjoint from earlier studies; every
instance hash is checked against saved study manifests. Budgets mirror the
confirmatory protocol: 3 s, 0.3 s repair cap, common witness start. Results
guide algorithm selection only; a new frozen protocol is needed for claims.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..classical import solve_classical
from ..hybrid import solve_hybrid
from ..job_profile import solve_profile
from ..local_search import solve_local_search
from ..ranked_search import solve_ranked
from ..trust_region import SolutionPredictor, solve_predict_search, train_predictor
from .datasets import instance_hash, witness_instance
from .protocol import objectives_at_budget

RESULTS = Path(__file__).parent / "results"
BASE_SEED = 31_000_000
METHODS = (
    "heuristic",
    "uniform",
    "classical_portfolio",
    "local_branch",
    "adaptive_local_branch",
    "profile",
    "profile_fallback",
    "local_search",
    "portfolio_plus",
    "predict_search",
    "predict_search_polish",
)


def known_hashes() -> set[str]:
    found = set()
    for path in RESULTS.rglob("*.json*"):
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError):
            continue
        found.update(
            token.strip('"')
            for token in text.replace(",", " ").split()
            if len(token.strip('"')) == 64
        )
    return found


def instances(split: str, per_regime: int, large: bool, offset: int):
    out = []
    for r, regime in enumerate(("balanced", "energy", "periodic")):
        for i in range(per_regime):
            seed = BASE_SEED + offset + 1000 * r + i
            instance, witness = witness_instance(seed=seed, regime=regime, large=large)
            out.append(
                dict(id=f"{split}:{regime}:{seed}", instance=instance, witness=witness)
            )
    return out


def run(task):
    item, method, predictor_path, budget, repair, seed = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    instance, witness = item["instance"], item["witness"]
    warm = dict(initial_schedule=witness, time_limit=budget, seed=seed)
    if method == "heuristic":
        result = solve_hybrid(
            instance,
            selector="heuristic",
            initial_exact=False,
            neighborhood_fraction=0.5,
            repair_time=repair,
            **warm,
        )
    elif method == "uniform":
        result = solve_ranked(
            instance, mode="uniform", fraction=0.2, repair_time=repair, **warm
        )
    elif method in (
        "classical_portfolio",
        "portfolio_plus",
        "local_branch",
        "adaptive_local_branch",
    ):
        mode = {"classical_portfolio": "portfolio"}.get(method, method)
        result = solve_classical(
            instance, mode=mode, fraction=0.2, repair_time=repair, **warm
        )
    elif method in ("profile", "profile_fallback"):
        result = solve_profile(
            instance,
            fallback_time=repair if method == "profile_fallback" else 0.0,
            **warm,
        )
    elif method == "local_search":
        result = solve_local_search(instance, **warm)
    else:
        predictor = SolutionPredictor.load(predictor_path)
        # Predict-and-search is a cold-start method by design (no witness).
        result = solve_predict_search(
            instance,
            predictor,
            time_limit=budget,
            polish=method == "predict_search_polish",
            seed=seed,
        )
    payload = result.to_dict()
    return dict(
        instance_id=item["id"],
        method=method,
        seed=seed,
        objective=objectives_at_budget(payload, budget),
        returned_objective=payload["objective"],
        runtime=payload["runtime"],
        iterations=payload["iterations"],
        time_to_first_feasible=payload["time_to_first_feasible"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS / "heuristics-dev")
    parser.add_argument("--budget", type=float, default=3.0)
    parser.add_argument("--repair-time", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    train = instances("train", 4, False, 0)
    test = instances("iid", 3, False, 100) + instances("ood", 2, True, 200)
    seen = known_hashes()
    hashes = [instance_hash(i["instance"]) for i in train + test]
    if seen.intersection(hashes) or len(set(hashes)) != len(hashes):
        raise SystemExit("Instance overlap with earlier studies or within this one")
    predictor_path = args.output / "predictor.pt"
    model, meta = train_predictor(
        [i["instance"] for i in train], epochs=40, collect_budget=3.0, seed=0
    )
    model.save(
        predictor_path, metadata={**meta, "training_hashes": hashes[: len(train)]}
    )
    tasks = [
        (item, method, predictor_path, args.budget, args.repair_time, seed)
        for item in test
        for seed in args.seeds
        for method in METHODS
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))
    (args.output / "runs.json").write_text(json.dumps(rows, indent=2) + "\n")
    summarize(rows, args.output, meta)


def summarize(rows, output, meta):
    by = {}
    for row in rows:
        by.setdefault(row["instance_id"], {}).setdefault(row["method"], []).append(
            row["objective"] or 0.0
        )
    lines = [
        "# Development heuristic comparison (exploratory)",
        "",
        f"Predictor: {meta['training_instances']} training instances, "
        f"labeling {meta['labeling_runtime']:.1f}s. Relative change is versus "
        "`heuristic` per instance, then averaged; W/L counts instances.",
        "",
        "| Method | Split | Mean rel. vs heuristic | W/L | Mean objective |",
        "|---|---|---:|---:|---:|",
    ]
    for method in METHODS:
        for split in ("iid", "ood"):
            rel, wins, losses, objs = [], 0, 0, []
            for iid, methods in by.items():
                if not iid.startswith(split):
                    continue
                base = np.mean(methods["heuristic"])
                value = np.mean(methods[method])
                objs.append(value)
                rel.append((value - base) / max(abs(base), 1e-9))
                wins += value > base + 1e-6
                losses += value < base - 1e-6
            lines.append(
                f"| {method} | {split} | {100 * np.mean(rel):+.2f}% | {wins}/{losses} "
                f"| {np.mean(objs):.1f} |"
            )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
