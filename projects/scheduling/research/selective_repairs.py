"""Frozen exploratory comparison of delayed and budget-aware MILP repairs."""
from __future__ import annotations

import argparse
import hashlib
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import operator_attribution as attribution
from .datasets import source_hashes
from .portfolio_ablation import frozen_plan, save
from .protocol import append_record, read_jsonl

PROTOCOL = Path(__file__).with_name("SELECTIVE_REPAIRS.md")
VARIANTS = {
    "reference": {},
    "dp_only": {"enable_repairs": False},
    "delayed": {"repair_start_fraction": 1 / 3},
    "budget_aware": {"repair_budget_fraction": .25, "repair_min_remaining": 2.},
}
COMPARISONS = [("delayed", "reference"), ("budget_aware", "reference")]


def make_plan(**kwargs):
    plan = attribution.make_plan(**kwargs)
    variants = {name: dict(settings) for name, settings in VARIANTS.items()}
    variants["budget_aware"]["repair_min_remaining"] = 2 * plan["repair_time"]
    plan.update(format="onts-selective-repairs-pilot-v1", variants=variants,
                comparisons=[list(pair) for pair in COMPARISONS],
                # Keep the per-call cap at 1 second at both pilot and full budgets.
                order_seed=981008,
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                scope="fixed exploratory short-budget pilot; no default promotion")
    return plan


def policy_diagnostics(rows, plan):
    """Audit recorded operator intervals independently of the scheduling branch."""
    result = {}
    for method, settings in plan["variants"].items():
        calls = skipped = 0
        spent_total = max_overshoot = 0.
        for row in rows:
            if row["method"] != method:
                continue
            spent = 0.
            allowance = row["improvement_budget"]
            offset = row["initialization_runtime"]
            quota = settings.get("repair_budget_fraction", 1.) * allowance
            for event in row["history"]:
                if event.get("stage") != "improvement" or event.get("operator") != "repair":
                    continue
                elapsed = event["solve_runtime"]
                start = event["time"] - offset - elapsed
                assert start + 1e-7 >= settings.get("repair_start_fraction", 0.) * allowance
                assert spent < quota + 1e-7
                assert method != "dp_only"
                if event["event"] == "repair_skipped":
                    skipped += 1
                else:
                    calls += 1
                    reserve = settings.get("repair_min_remaining", 0.)
                    deadline = allowance - min(.25, .05 * allowance)
                    assert event["repair_time_limit"] <= plan["repair_time"] + 1e-7
                    assert event["solver_started"] + event["repair_time_limit"] <= deadline - reserve + .001
                    assert event["repair_seconds_before"] == spent
                    assert event["repair_time_limit"] <= quota - spent + 1e-7
                spent += elapsed
            spent_total += spent
            max_overshoot = max(max_overshoot, spent - quota)
        result[method] = dict(repair_calls=calls, skipped_calls=skipped,
                              repair_seconds=spent_total,
                              max_quota_overshoot_seconds=max_overshoot)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1009, 1103])
    parser.add_argument("--budget", type=float, default=10.)
    parser.add_argument("--initialization-budget", type=float, default=3.)
    parser.add_argument("--base", type=int, default=69_000_000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ONTS_SOLVER_THREADS"):
        os.environ[name] = "1"
    torch.set_num_threads(1)
    plan = make_plan(count=args.count, seeds=args.seeds, budget=args.budget,
                     initialization_budget=args.initialization_budget,
                     workers=args.workers, base=args.base)
    if not (args.output / "plan.json").exists():
        old = attribution.previous_instance_hashes(Path(__file__).parent / "results")
        if set(plan["instance_hashes"]) & old:
            raise ValueError("New study overlaps an existing instance manifest")
    plan = frozen_plan(args.output, plan)
    (args.output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    if args.plan_only:
        return
    previous = read_jsonl(args.output / "runs.jsonl")
    done = {r["run_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Duplicate run IDs")
    tasks = [(i, s, plan, str(args.output.resolve()))
             for i in range(plan["count"]) for s in plan["seeds"]
             if any(f"{i}|{s}|{m}" not in done for m in plan["variants"])]
    order = np.random.default_rng(plan["order_seed"]).permutation(len(tasks))
    print(f"Frozen {len(tasks)} pairs, four methods, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(attribution.run_pair, tasks[int(k)]): tasks[int(k)] for k in order}
        for future in as_completed(futures):
            for row in future.result():
                append_record(args.output / "runs.jsonl", row)
            print(f"Completed {futures[future][0:2]}", flush=True)
    if source_hashes() != plan["source_hashes"] or attribution.input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Sources or inputs changed during execution")
    rows = read_jsonl(args.output / "runs.jsonl")
    result = attribution.analyze(rows, plan)
    result["policy_diagnostics"] = policy_diagnostics(rows, plan)
    save(args.output / "analysis.json", result)
    content = attribution.report(result, plan).replace(
        "# Frozen operator attribution results", "# Selective repair pilot results").replace(
        "Intervals cover four comparisons per metric.", "Intervals cover two comparisons per metric.")
    content += (f"\nExploratory {plan['budget']:g}-second pilot: "
                "no default promotion or 30-second confirmation.\n")
    (args.output / "report.md").write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
