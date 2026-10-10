"""Fixed-size DP-only noninferiority and repair-mask replacement measurement."""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from . import operator_attribution as attribution
from .confirmatory import infer
from .datasets import source_hashes
from .portfolio_ablation import frozen_plan, save
from .protocol import append_record, read_jsonl
from .statistics import holm_adjust

PROTOCOL = Path(__file__).with_name("REPAIR_FOLLOWUP.md")
VARIANTS = {"reference": {}, "dp_only": {"enable_repairs": False},
            "time_window": {"repair_strategy": "time_window"}}
COMPARISONS = [("reference", "dp_only"), ("time_window", "reference")]


def noninferiority(values, *, margin, family=3):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError("Positive finite noninferiority margin required")
    x = np.asarray(values, dtype=float)
    result = infer(x, family=family)
    mean, se = float(x.mean()), float(stats.sem(x))
    result["one_sided_p"] = float(stats.t.sf((mean + margin) / se, len(x) - 1)) if se else (
        0. if mean > -margin else 1.)
    result.update(method="dp_only", baseline="reference", margin_pp=margin,
                  null="mean(DP only - reference) <= -margin", instance_deltas=x.tolist())
    return result


def make_plan(*, count=80, seeds=(809, 907), budget=30., initialization_budget=9.,
              workers=8, base=67_000_000):
    plan = attribution.make_plan(count=count, seeds=seeds, budget=budget,
                                 initialization_budget=initialization_budget,
                                 workers=workers, base=base)
    plan.update(format="onts-repair-followup-v1", variants=VARIANTS,
                comparisons=[list(pair) for pair in COMPARISONS], order_seed=971031,
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                noninferiority_margin_pp=.5, primary_family_size=3,
                planning=dict(previous_cases=40, previous_sd_pp=[.5346036130098665, .40135999652336396],
                              fixed_cases=80, pooled_primary=False),
                checkpoints=[1., 3., 10., 30.])
    return plan


def analyze(rows, plan):
    result = attribution.analyze(rows, plan)
    family = len(plan["comparisons"]) + 1
    for c in result["comparisons"]:
        c["score_pp"].update(infer(c["score_pp"]["instance_deltas"], family=family))
    reference_delta = next(c["score_pp"]["instance_deltas"] for c in result["comparisons"]
                           if c["method"] == "reference" and c["baseline"] == "dp_only")
    ni = noninferiority(-np.asarray(reference_delta), margin=plan["noninferiority_margin_pp"], family=family)
    pvalues = [c["score_pp"]["two_sided_p"] for c in result["comparisons"]] + [ni["one_sided_p"]]
    adjusted = holm_adjust(pvalues)
    for c, pvalue in zip(result["comparisons"], adjusted[:-1]):
        v = c["score_pp"]
        v["holm_p"] = pvalue
        v["superiority_established"] = pvalue < .05 and v["simultaneous_ci95"][0] > 0
        v["practical_threshold_met"] = v["superiority_established"] and v["simultaneous_ci95"][0] > plan["practical_threshold_pp"]
    ni["holm_p"] = adjusted[-1]
    ni["noninferiority_established"] = (ni["holm_p"] < .05 and
        ni["simultaneous_ci95"][0] > -ni["margin_pp"] and result["all_unsolved"] < plan["count"])
    result["noninferiority"] = ni
    result["primary_family_size"] = family
    refs = result["references"]
    for method in plan["variants"]:
        selected = [r for r in rows if r["method"] == method]
        result["methods"][method]["checkpoint_scores_pp"] = {
            str(t): float(np.mean([100 * max([0.] + [e["objective"] for e in r["history"]
                    if e.get("objective") is not None and e["time"] <= min(t, plan["budget"])]) / refs[r["instance"]]
                    if refs[r["instance"]] else 0. for r in selected])) for t in plan["checkpoints"]}
        result["methods"][method]["returned_after_deadline"] = sum(r["runtime"] > plan["budget"] for r in selected)
    return result


def report(result, plan):
    text = attribution.report(result, plan).replace("# Frozen operator attribution results",
        "# Focused DP-only and repair-mask follow-up").replace(
        "Intervals cover four comparisons per metric.",
        "Endpoint family: two superiority tests and one noninferiority test. AUC family: two contrasts.")
    ni = result["noninferiority"]; lo, hi = ni["simultaneous_ci95"]
    text += (f"\n## Prespecified DP-only noninferiority\n\n"
             f"DP only minus reference: {ni['mean_percent']:+.3f} pp; simultaneous 95% CI "
             f"[{lo:+.3f}, {hi:+.3f}]; margin {ni['margin_pp']:.3f} pp; "
             f"one-sided Holm p {ni['holm_p']:.4g}. Established: {ni['noninferiority_established']}.\n\n"
             "This bounds an average normalized score tradeoff on the tested generator; it does not "
             "establish equivalence, a per-instance guarantee, or solver-free initialization.\n\n"
             "## Descriptive quality checkpoints\n\n"
             "| Method | 1 s | 3 s | 10 s | 30 s | Returned after deadline |\n"
             "|---|---:|---:|---:|---:|---:|\n")
    for m, v in result["methods"].items():
        scores = v["checkpoint_scores_pp"]
        text += f"| {m} | " + " | ".join(f"{scores[str(t)]:.3f}" for t in plan["checkpoints"])
        text += f" | {v['returned_after_deadline']} |\n"
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--seeds", type=int, nargs="+", default=[809, 907])
    parser.add_argument("--budget", type=float, default=30.)
    parser.add_argument("--initialization-budget", type=float, default=9.)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base", type=int, default=67_000_000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ONTS_SOLVER_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    plan = make_plan(count=args.count, seeds=args.seeds, budget=args.budget,
                     initialization_budget=args.initialization_budget, workers=args.workers, base=args.base)
    if not (args.output / "plan.json").exists():
        old = attribution.previous_instance_hashes(Path(__file__).parent / "results")
        if set(plan["instance_hashes"]) & old:
            raise ValueError("Fresh follow-up overlaps an existing instance manifest")
    plan = frozen_plan(args.output, plan)
    (args.output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    if args.plan_only:
        print("Frozen focused plan, source snapshots and input hashes", flush=True)
        return
    previous = read_jsonl(args.output / "runs.jsonl")
    done = {r["run_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Duplicate saved run IDs")
    tasks = [(i, s, plan, str(args.output.resolve())) for i in range(plan["count"])
             for s in plan["seeds"] if any(f"{i}|{s}|{m}" not in done for m in plan["variants"])]
    order = np.random.default_rng(plan["order_seed"]).permutation(len(tasks))
    print(f"Frozen {len(tasks)} pairs, three pipelines each, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(attribution.run_pair, tasks[int(k)]): tasks[int(k)] for k in order}
        for future in as_completed(futures):
            for row in future.result():
                append_record(args.output / "runs.jsonl", row)
            task = futures[future]
            print(f"Completed instance {task[0]}, seed {task[1]}", flush=True)
    if source_hashes() != plan["source_hashes"] or attribution.input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Frozen sources or inputs changed during execution")
    result = analyze(read_jsonl(args.output / "runs.jsonl"), plan)
    save(args.output / "analysis.json", result)
    content = report(result, plan)
    (args.output / "report.md").write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
