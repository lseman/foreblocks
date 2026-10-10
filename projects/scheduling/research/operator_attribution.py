"""Frozen shared-initialization comparison of DP, MILP and local-search repair."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ..local_search import solve_local_search
from ..profile_portfolio import solve_profile_portfolio
from .confirmatory import infer
from .datasets import instance_hash, source_hashes
from .portfolio_ablation import COMMON, frozen_plan, save, verify
from .protocol import append_record, objectives_at_budget, read_jsonl
from .realistic import BUNDLED, realistic_instance
from .statistics import bootstrap_mean, holm_adjust, normalized_auc

VARIANTS = {
    "reference": {},
    "dp_only": {"enable_repairs": False},
    "milp_only": {"enable_profiles": False},
    "local_search": None,
    "time_window": {"repair_strategy": "time_window"},
}
COMPARISONS = [("reference", "milp_only"), ("reference", "local_search"),
               ("reference", "dp_only"), ("time_window", "reference")]
PROTOCOL = Path(__file__).with_name("OPERATOR_ATTRIBUTION.md")


def input_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(BUNDLED.glob("*.json"))}


def previous_instance_hashes(root):
    """Read identity metadata only, without loading archived run histories."""
    found = set()
    for name in ("plan.json", "manifest.json"):
        for path in root.rglob(name):
            data = json.loads(path.read_text())
            values = data.get("instance_hashes", [])
            found.update(values.values() if isinstance(values, dict) else values)
            found.update(entry["hash"] for entry in data.get("entries", [])
                         if "hash" in entry)
    return found


def make_plan(*, count=40, seeds=(601, 701), budget=30., initialization_budget=9.,
              workers=8, base=65_000_000):
    if count < 2 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least two cases and unique paired seeds required")
    if not (math.isfinite(budget) and math.isfinite(initialization_budget)
            and 0 < initialization_budget < budget and workers >= 1):
        raise ValueError("Require positive initialization allowance below total budget")
    hashes = [instance_hash(realistic_instance(base + i)) for i in range(count)]
    if len(set(hashes)) != count:
        raise ValueError("Duplicate generated instances")
    return dict(format="onts-operator-attribution-v1", count=count, seeds=list(seeds),
                budget=budget, initialization_budget=initialization_budget,
                workers=workers, base=base, repair_time=min(1., budget / 10),
                common=COMMON, variants=VARIANTS,
                comparisons=[list(pair) for pair in COMPARISONS], order_seed=971017,
                source_hashes=source_hashes(), bundled_input_hashes=input_hashes(),
                instance_hashes=hashes,
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                scope="fresh generator draws; common initializer charged to budget; failed starts retained",
                practical_threshold_pp=1.)


def run_pair(task):
    index, seed, plan, output_string = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    if source_hashes() != plan["source_hashes"] or input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Sources or benchmark inputs changed after freeze")
    instance = realistic_instance(plan["base"] + index)
    if instance_hash(instance) != plan["instance_hashes"][index]:
        raise RuntimeError("Instance differs from frozen hash")
    output = Path(output_string)
    cache = output / "initializers" / f"{index}-{seed}.json"
    if cache.exists():
        prefix = json.loads(cache.read_text())
    else:
        result = solve_profile_portfolio(
            instance, time_limit=plan["initialization_budget"], max_iterations=0,
            seed=seed, repair_time=plan["repair_time"],
            **dict(plan["common"], init_fraction=1.))
        if result.status in ("error", "validation_failed"):
            raise RuntimeError(f"Initializer solver status: {result.status}")
        value = verify(instance, result.schedule)
        usable = result.schedule is not None and result.runtime <= plan["initialization_budget"]
        prefix = dict(schedule=result.schedule.int().tolist() if usable else None,
                      objective=value if usable else None, runtime=result.runtime,
                      history=result.history if usable else [],
                      first_feasible_source=result.first_feasible_source,
                      instance_hash=instance_hash(instance), usable=usable)
        cache.parent.mkdir(exist_ok=True)
        save(cache, prefix)
    start = torch.tensor(prefix["schedule"], dtype=torch.bool) if prefix["usable"] else None
    if start is not None:
        verify(instance, start)
    fingerprint = hashlib.sha256(json.dumps(prefix["schedule"]).encode()).hexdigest()
    remaining = plan["budget"] - prefix["runtime"]
    methods = list(plan["variants"])
    np.random.default_rng(plan["order_seed"] + index * 1000 + seed).shuffle(methods)
    done = {r["run_id"] for r in read_jsonl(output / "runs.jsonl")}
    rows = []
    for method in methods:
        run_id = f"{index}|{seed}|{method}"
        if run_id in done:
            continue
        row = dict(run_id=run_id, instance=index, seed=seed, method=method,
                   instance_hash=instance_hash(instance), shared_start_hash=fingerprint,
                   initial_objective=prefix["objective"], initialization_runtime=prefix["runtime"],
                   improvement_budget=max(0., remaining), objective=None,
                   returned_objective=None, returned_schedule=None, history=[],
                   feasible=False, validation_passed=True, error=None,
                   status="shared_initialization_failed", runtime=prefix["runtime"])
        if start is not None:
            row["history"] = [dict(event, stage="initialization") for event in prefix["history"]]
            # Availability of the saved incumbent at the handoff is explicit for every arm.
            row["history"].append(dict(time=prefix["runtime"], event="shared_start",
                                       objective=prefix["objective"], stage="handoff"))
            try:
                if method == "local_search":
                    result = solve_local_search(instance, initial_schedule=start,
                                                time_limit=remaining, seed=seed)
                else:
                    result = solve_profile_portfolio(
                        instance, initial_schedule=start, time_limit=remaining, seed=seed,
                        repair_time=plan["repair_time"],
                        **dict(plan["common"], **plan["variants"][method]))
                returned_value = verify(instance, result.schedule)
                row["history"] += [dict(event, time=prefix["runtime"] + event["time"],
                                         stage="improvement") for event in result.history]
                bad_statuses = [e["status"] for e in result.history
                                if e.get("status") in ("error", "validation_failed", "infeasible", "unbounded")]
                if bad_statuses:
                    raise RuntimeError(f"Repair solver failure: {bad_statuses}")
                credited = objectives_at_budget(row, plan["budget"])
                row.update(objective=credited, feasible=credited is not None,
                           returned_objective=returned_value, returned_schedule=result.to_dict()["schedule"],
                           status=result.status, runtime=prefix["runtime"] + result.runtime)
            except Exception as exc:
                row.update(error=f"{type(exc).__name__}: {exc}", validation_passed=False)
        rows.append(row)
    return rows


def analyze(rows, plan):
    expected = {(i, s, m) for i in range(plan["count"])
                for s in plan["seeds"] for m in plan["variants"]}
    indexed = {(r["instance"], r["seed"], r["method"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Complete unique paired records required")
    if any(r["error"] or not r["validation_passed"] for r in rows):
        raise ValueError("Execution or validation errors block primary inference")
    for i in range(plan["count"]):
        for s in plan["seeds"]:
            for field in ("shared_start_hash", "initialization_runtime", "improvement_budget", "initial_objective"):
                if len({indexed[i, s, m][field] for m in plan["variants"]}) != 1:
                    raise ValueError("Paired starts and allowances must match")
    refs = {i: max(r["objective"] or 0. for r in rows if r["instance"] == i)
            for i in range(plan["count"])}
    score = {key: 100 * (r["objective"] or 0.) / refs[key[0]] if refs[key[0]] else 0.
             for key, r in indexed.items()}
    auc = {key: 100 * normalized_auc(r["history"], budget=plan["budget"], reference=refs[key[0]])
           if refs[key[0]] else 0. for key, r in indexed.items()}
    methods = {}
    for m in plan["variants"]:
        selected = [r for r in rows if r["method"] == m]
        keys = [(r["instance"], r["seed"], m) for r in selected]
        operators = {op: dict(calls=0, improvements=0, gain=0., seconds=0.)
                     for op in ("profile", "repair")}
        for r in selected:
            old = r["initial_objective"]
            for event in r["history"]:
                if event.get("stage") != "improvement":
                    continue
                value = event.get("objective")
                op = event.get("operator")
                if op in operators:
                    d = operators[op]
                    gain = max(0., value - old) if value is not None and old is not None else 0.
                    d["calls"] += 1
                    d["improvements"] += int(gain > 1e-9)
                    d["gain"] += gain
                    d["seconds"] += event.get("solve_runtime", 0.)
                if value is not None:
                    old = value
        solved_keys = [key for key in keys if refs[key[0]]]
        gains = [r["objective"] - r["initial_objective"] for r in selected if r["feasible"]]
        methods[m] = dict(runs=len(selected), feasible=sum(r["feasible"] for r in selected),
                          score_pp=float(np.mean([score[key] for key in keys])),
                          conditional_score_pp=float(np.mean([score[key] for key in solved_keys]))
                          if solved_keys else None,
                          auc_pp=float(np.mean([auc[key] for key in keys])),
                          mean_raw_gain_on_usable_starts=float(np.mean(gains)) if gains else None,
                          improved_runs=sum(gain > 1e-9 for gain in gains), operators=operators)
    comparisons = []
    for number, (a, b) in enumerate(plan["comparisons"]):
        comparison = dict(method=a, baseline=b)
        for name, metric in (("score_pp", score), ("auc_pp", auc)):
            delta = [float(np.mean([metric[i, s, a] - metric[i, s, b] for s in plan["seeds"]]))
                     for i in range(plan["count"])]
            statistic = infer(delta, family=len(plan["comparisons"]))
            statistic["instance_deltas"] = delta
            statistic["descriptive_bootstrap"] = bootstrap_mean(delta, seed=1207 + number)
            comparison[name] = statistic
        comparisons.append(comparison)
    for name in ("score_pp", "auc_pp"):
        adjusted = holm_adjust([c[name]["two_sided_p"] for c in comparisons])
        for c, pvalue in zip(comparisons, adjusted):
            c[name]["holm_p"] = pvalue
            c[name]["superiority_established"] = pvalue < .05 and c[name]["simultaneous_ci95"][0] > 0
            c[name]["practical_threshold_met"] = (c[name]["superiority_established"] and
                c[name]["simultaneous_ci95"][0] > plan["practical_threshold_pp"])
    return dict(instances=plan["count"], runs=len(rows), methods=methods, comparisons=comparisons,
                all_unsolved=sum(v <= 0 for v in refs.values()), references=refs, errors=0)


def report(result, plan):
    lines = ["# Frozen operator attribution results", "",
             f"{plan['count']} fresh instances; paired seeds {plan['seeds']}; {plan['budget']:g} s total allowance.",
             "Common initialization is charged to every pipeline; failed starts remain zero.", "",
             "| Method | Feasible | All-instance score (%) | Conditional score (%) | AUC (%) | Improved starts |",
             "|---|---:|---:|---:|---:|---:|"]
    for m, v in result["methods"].items():
        conditional = f"{v['conditional_score_pp']:.3f}" if v["conditional_score_pp"] is not None else "--"
        lines.append(f"| {m} | {v['feasible']}/{v['runs']} | {v['score_pp']:.3f} | {conditional} | {v['auc_pp']:.3f} | {v['improved_runs']} |")
    for name, title in (("score_pp", "Primary endpoint"), ("auc_pp", "Secondary AUC")):
        lines += ["", f"## {title}", "",
                  "| Comparison | Difference (pp) | Simultaneous 95% CI | Holm p | W/T/L | Established |",
                  "|---|---:|---|---:|---|---|"]
        for c in result["comparisons"]:
            v = c[name]; low, high = v["simultaneous_ci95"]
            lines.append(f"| {c['method']} − {c['baseline']} | {v['mean_percent']:+.3f} | [{low:+.3f}, {high:+.3f}] | {v['holm_p']:.4g} | {v['wins']}/{v['ties']}/{v['losses']} | {v['superiority_established']} |")
    lines += ["", "## Descriptive operator diagnostics", "",
              "| Method | Profile improvements/calls | Profile seconds | MILP improvements/calls | MILP seconds |",
              "|---|---:|---:|---:|---:|"]
    for m, v in result["methods"].items():
        p, r = v["operators"]["profile"], v["operators"]["repair"]
        lines.append(f"| {m} | {p['improvements']}/{p['calls']} | {p['seconds']:.1f} | {r['improvements']}/{r['calls']} | {r['seconds']:.1f} |")
    lines += ["", f"All-unsolved instances retained: {result['all_unsolved']}; execution errors: {result['errors']}.",
              "Intervals cover four comparisons per metric. Bootstrap/sign checks are descriptive.",
              "The study isolates improvement under shared initialization. Failed-start salvage,",
              "new mission distributions, published decomposition comparators and constellations remain untested."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[601, 701])
    parser.add_argument("--budget", type=float, default=30.)
    parser.add_argument("--initialization-budget", type=float, default=9.)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--base", type=int, default=65_000_000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ONTS_SOLVER_THREADS"):
        os.environ[key] = "1"
    torch.set_num_threads(1)
    plan = make_plan(count=args.count, seeds=args.seeds, budget=args.budget,
                     initialization_budget=args.initialization_budget,
                     workers=args.workers, base=args.base)
    if not (args.output / "plan.json").exists():
        old = previous_instance_hashes(Path(__file__).parent / "results")
        if set(plan["instance_hashes"]) & old:
            raise ValueError("Fresh study overlaps an existing instance manifest")
    plan = frozen_plan(args.output, plan)
    (args.output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    if args.plan_only:
        print("Frozen plan, source archive and input hashes", flush=True)
        return
    previous = read_jsonl(args.output / "runs.jsonl")
    done = {r["run_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Duplicate saved run IDs")
    tasks = [(i, s, plan, str(args.output.resolve())) for i in range(plan["count"])
             for s in plan["seeds"] if any(f"{i}|{s}|{m}" not in done for m in plan["variants"])]
    order = np.random.default_rng(plan["order_seed"]).permutation(len(tasks))
    print(f"Frozen {len(tasks)} pairs, five methods each, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_pair, tasks[int(k)]): tasks[int(k)] for k in order}
        for future in as_completed(futures):
            for row in future.result():
                append_record(args.output / "runs.jsonl", row)
            task = futures[future]
            print(f"Completed instance {task[0]}, seed {task[1]}", flush=True)
    if source_hashes() != plan["source_hashes"] or input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Sources or inputs changed during execution")
    result = analyze(read_jsonl(args.output / "runs.jsonl"), plan)
    save(args.output / "analysis.json", result)
    content = report(result, plan)
    (args.output / "report.md").write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
