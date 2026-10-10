"""Fixed-size paired confirmation of the frozen profile portfolio.

Protocol: research/PROFILE_CONFIRMATORY.md. No tuning or optional stopping.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import load_onts_instance, onts_instance_to_json_data
from ..hybrid import solve_hybrid
from ..job_profile import solve_profile
from ..milp import ONTSConfig
from ..profile_portfolio import solve_profile_portfolio
from ..ranked_search import solve_ranked
from .confirmatory import infer, save
from .datasets import build_suite, source_hashes
from .oracle import oracle_check
from .protocol import append_record, objectives_at_budget, read_jsonl
from .statistics import holm_adjust

METHODS = ("portfolio", "heuristic", "uniform", "profile")
PRIMARY = ("heuristic", "uniform")
SEEDS = (101, 211, 307)
BUDGET, REPAIR = 3.0, 0.3
FROZEN = dict(
    controller="stagnation",
    patience=10,
    job_selection="rules",
    max_jobs=3,
    fraction=0.2,
    candidate_count=8,
)


def run(task):
    entry, seed, method = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    row = dict(
        run_id=f"{entry['id']}|{seed}|{method}",
        instance_id=entry["id"],
        instance_hash=entry["hash"],
        split=entry["split"],
        method=method,
        search_seed=seed,
        budget=BUDGET,
    )
    try:
        instance = load_onts_instance(entry["path"])
        common = dict(
            initial_schedule=torch.tensor(entry["witness"]),
            time_limit=BUDGET,
            seed=seed,
        )
        if method == "portfolio":
            result = solve_profile_portfolio(
                instance, repair_time=REPAIR, **FROZEN, **common
            )
        elif method == "heuristic":
            result = solve_hybrid(
                instance,
                selector="heuristic",
                initial_exact=False,
                neighborhood_fraction=0.5,
                repair_time=REPAIR,
                **common,
            )
        elif method == "uniform":
            result = solve_ranked(
                instance,
                mode="uniform",
                fraction=0.2,
                candidate_count=8,
                repair_time=REPAIR,
                **common,
            )
        else:
            result = solve_profile(instance, **common)
        payload = result.to_dict()
        valid, objective, _ = oracle_check(
            onts_instance_to_json_data(instance), payload["schedule"], ONTSConfig()
        )
        if not valid or not np.isclose(objective, payload["objective"], atol=1e-4):
            raise RuntimeError("Independent output validation failed")
        credited = objectives_at_budget(payload, BUDGET)
        row.update(
            objective=credited,
            returned_objective=payload["objective"],
            feasible=credited is not None,
            score=credited if credited is not None else 0.0,
            validation_passed=True,
            runtime=payload["runtime"],
            iterations=payload["iterations"],
            overrun=max(0.0, payload["runtime"] - BUDGET),
        )
    except Exception as exc:  # Failures are retained with score zero.
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            score=0.0,
            feasible=False,
            validation_passed=False,
        )
    return row


def analyze(rows, entries):
    indexed = {(r["instance_id"], r["search_seed"], r["method"]): r for r in rows}
    expected = {(e["id"], s, m) for e in entries for s in SEEDS for m in METHODS}
    if set(indexed) != expected or len(indexed) != len(rows):
        raise ValueError("Complete unique paired records required before inference")

    def deltas(baseline):
        out = []
        for e in entries:
            a = np.mean([indexed[e["id"], s, "portfolio"]["score"] for s in SEEDS])
            b = np.mean([indexed[e["id"], s, baseline]["score"] for s in SEEDS])
            out.append((e["split"], float(100 * (a - b) / max(abs(b), 1e-9))))
        return out

    comparisons = []
    for baseline in METHODS[1:]:
        detail = deltas(baseline)
        values = [d for _, d in detail]
        result = dict(
            baseline=baseline, primary=baseline in PRIMARY, **infer(values, family=2)
        )
        result["split_descriptive"] = {
            split: float(np.mean([d for s, d in detail if s == split]))
            for split in ("test", "ood")
        }
        comparisons.append(result)
    primary = [c for c in comparisons if c["primary"]]
    for c, p in zip(primary, holm_adjust([c["two_sided_p"] for c in primary])):
        c["holm_p"] = p
        c["statistical_superiority"] = p < 0.05 and c["simultaneous_ci95"][0] > 0
        c["practical_superiority_1_percent"] = (
            c["statistical_superiority"] and c["simultaneous_ci95"][0] > 1
        )
    all_valid = all(r.get("validation_passed") and r.get("feasible") for r in rows)
    return dict(
        completed_runs=len(rows),
        comparisons=comparisons,
        all_outputs_valid_by_deadline=all_valid,
        max_overrun=max(r.get("overrun", 0.0) for r in rows),
        claim_supported=all_valid
        and all(c["practical_superiority_1_percent"] for c in primary),
    )


def report(analysis, out):
    lines = [
        "# Profile portfolio confirmatory results",
        "",
        f"Completed runs: {analysis['completed_runs']}; all outputs valid by deadline: "
        f"{analysis['all_outputs_valid_by_deadline']}; max overrun "
        f"{analysis['max_overrun']:.3f}s; practical claim supported: "
        f"{analysis['claim_supported']}.",
        "",
        "| Baseline | Primary | Mean % | Simultaneous 95% CI | Holm p | W/L/T | IID % | OOD % |",
        "|---|---|---:|---|---:|---:|---:|---:|",
    ]
    for c in analysis["comparisons"]:
        lo, hi = c["simultaneous_ci95"]
        holm = f"{c['holm_p']:.5f}" if "holm_p" in c else "—"
        lines.append(
            f"| {c['baseline']} | {c['primary']} | {c['mean_percent']:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {holm} | {c['wins']}/{c['losses']}/{c['ties']} | "
            f"{c['split_descriptive']['test']:+.3f} | {c['split_descriptive']['ood']:+.3f} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        plan = json.loads((out / "plan.json").read_text())
        if source_hashes() != plan["source_hashes"]:
            raise RuntimeError("Frozen source changed; do not resume this study")
        manifest = json.loads((out / "data/manifest.json").read_text())
    else:
        manifest = build_suite(
            out / "data",
            seed=47_000_000,
            train_per_regime=0,
            validation_per_regime=0,
            test_per_regime=10,
            ood_per_regime=10,
        )
        old = set()
        for path in out.parent.glob("*/data/manifest.json"):
            if path != out / "data/manifest.json":
                old.update(e["hash"] for e in json.loads(path.read_text())["entries"])
        if old & {e["hash"] for e in manifest["entries"]}:
            raise RuntimeError("New study overlaps an existing study")
        plan = dict(
            format="onts-profile-confirmatory-v1",
            source_hashes=source_hashes(),
            frozen=FROZEN,
            methods=METHODS,
            primary=PRIMARY,
            seeds=SEEDS,
            budget=BUDGET,
            repair_time=REPAIR,
            protocol="research/PROFILE_CONFIRMATORY.md",
        )
        save(out / "plan.json", plan)
    entries = [e for e in manifest["entries"] if e["split"] in ("test", "ood")]
    if len(entries) != 60:
        raise RuntimeError(f"Expected 60 held-out instances, found {len(entries)}")
    done = {r["run_id"] for r in read_jsonl(out / "runs.jsonl")}
    tasks = [
        (e, s, m)
        for e in entries
        for s in SEEDS
        for m in METHODS
        if f"{e['id']}|{s}|{m}" not in done
    ]
    order = np.random.default_rng(8231).permutation(len(tasks))
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run, [tasks[int(i)] for i in order]):
            append_record(out / "runs.jsonl", row)
    rows = read_jsonl(out / "runs.jsonl")
    analysis = analyze(rows, entries)
    save(out / "analysis.json", analysis)
    report(analysis, out)


if __name__ == "__main__":
    main()
