"""Replicated factorial constellation comparison with validated callback credit."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ..constellation import generate_constellation, solve_constellation
from .constellation_oracle import data_hash, from_data, instance_data, replay
from .datasets import source_hashes
from .operator_attribution import input_hashes
from .portfolio_ablation import frozen_plan, save
from .protocol import append_record, read_jsonl
from .statistics import holm_adjust, normalized_auc

PROTOCOL = Path(__file__).with_name("CONSTELLATION_REPLICATION.md")
DESIGN = list(itertools.product((96, 144, 192), (5, 10, 15), (8, 12), (4, 8)))


def make_plan(output, *, replicates=3, seeds=(1301, 1409), budget=60., workers=12,
              base=73_000_000, design=None):
    design = DESIGN if design is None else design
    if (replicates < 2 or not seeds or len(set(seeds)) != len(seeds)
            or not math.isfinite(budget) or budget <= 0 or workers < 1 or not design):
        raise ValueError("Require replicated cells, unique seeds and positive budget/workers")
    output.mkdir(parents=True, exist_ok=True)
    directory = output / "instances"
    directory.mkdir(exist_ok=True)
    entries = []
    for cell, (T, S, Js, Jc) in enumerate(design):
        for replicate in range(replicates):
            i = cell * replicates + replicate
            data = instance_data(generate_constellation(S=S, Js=Js, Jc=Jc, T=T, seed=base + i))
            path = directory / f"{i}.json"
            if path.exists():
                if json.loads(path.read_text()) != data:
                    raise ValueError("Archived instance differs from frozen generator")
            else:
                save(path, data)
            entries.append(dict(instance=i, cell=cell, replicate=replicate, T=T, S=S,
                                Js=Js, Jc=Jc, generator_seed=base + i,
                                hash=data_hash(data), redraws=data["redraws"]))
        print(f"Prepared cell {cell + 1}/{len(design)}", flush=True)
    if len({entry["hash"] for entry in entries}) != len(entries):
        raise ValueError("Duplicate constellation instances")
    return dict(format="onts-constellation-replication-v1", budget=budget, workers=workers,
                seeds=list(seeds), replicates=replicates, design=[list(row) for row in design],
                entries=entries, count=len(entries), base=base, order_seed=981109,
                source_hashes=source_hashes(), bundled_input_hashes=input_hashes(),
                protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
                primary=["feasibility_pp", "score_pp"], solver_seed=0,
                scope="replication within constellation generator; curtailing battery; strict observed credit")


def expected_runs(plan):
    return [(entry["instance"], method, seed) for entry in plan["entries"]
            for method, seeds in (("portfolio", plan["seeds"]), ("exact", [0])) for seed in seeds]


def run(task):
    i, method, seed, plan, output_string = task
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    if source_hashes() != plan["source_hashes"] or input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Sources or inputs changed after freeze")
    output = Path(output_string)
    data = json.loads((output / "instances" / f"{i}.json").read_text())
    assert data_hash(data) == plan["entries"][i]["hash"]
    instance = from_data(data)
    row = dict(run_id=f"{i}|{method}|{seed}", instance=i, method=method, seed=seed,
               instance_hash=data_hash(data), history=[], objective=None, feasible=False,
               returned_objective=None, runtime=None, error=None, validation_passed=True)
    credited, returned = None, None

    def observe(schedule, value, time, source):
        nonlocal credited
        row["history"].append(dict(time=time, objective=value, source=source))
        if time <= plan["budget"] and (row["objective"] is None or value > row["objective"]):
            credited, row["objective"], row["feasible"] = schedule, value, True

    try:
        result = solve_constellation(instance, time_limit=plan["budget"], seed=seed,
                                     mode=method, observe_incumbents=True, on_incumbent=observe)
        if result["status"] == "error":
            raise RuntimeError("Constellation solver returned error status")
        returned = result["schedule"]
        row.update(returned_objective=result["objective"], runtime=result["runtime"],
                   status=result["status"], bound=result.get("bound"),
                   build_time=result.get("build_time"), iterations=result.get("iterations"),
                   repairs=result.get("repairs"))
        for name, schedule, value in (("credited", credited, row["objective"]),
                                       ("returned", returned, row["returned_objective"])):
            if schedule is not None:
                checked, bad = replay(data, schedule)
                if bad or not np.isclose(checked, value, rtol=1e-10, atol=1e-6):
                    raise RuntimeError(f"Independent {name} replay failed: {bad[:3]}")
    except Exception as exc:
        row.update(error=f"{type(exc).__name__}: {exc}", validation_passed=False)
    path = output / "schedules" / f"{i}-{method}-{seed}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, **{name: schedule for name, schedule in
                                (("credited", credited), ("returned", returned)) if schedule is not None})
    row["schedule_file"] = str(path.relative_to(output))
    row["schedule_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return row


def blocked_infer(values, cells, *, family=2):
    """Equal fixed-cell weights with within-cell variance and Satterthwaite df."""
    groups = [np.asarray([v for v, c in zip(values, cells) if c == cell], dtype=float)
              for cell in sorted(set(cells))]
    if any(len(group) < 2 for group in groups):
        raise ValueError("At least two independent instances per cell required")
    terms = np.array([group.var(ddof=1) / len(group) for group in groups]) / len(groups)**2
    mean = float(np.mean([group.mean() for group in groups]))
    variance = float(terms.sum())
    if variance:
        df = variance**2 / sum(term**2 / (len(group) - 1) for term, group in zip(terms, groups))
        se = math.sqrt(variance)
        p = float(2 * stats.t.sf(abs(mean / se), df))
        radius = float(stats.t.ppf(1 - .05 / (2 * family), df) * se)
    else:
        df, se, p, radius = None, 0., (1. if mean == 0 else 0.), 0.
    rng = np.random.default_rng(981009)
    bootstrap = np.zeros(10000)
    for group in groups:
        bootstrap += group[rng.integers(len(group), size=(10000, len(group)))].mean(axis=1) / len(groups)
    return dict(mean_pp=mean, standard_error=se, degrees_of_freedom=df,
                simultaneous_ci95=[mean-radius, mean+radius], two_sided_p=p,
                stratified_bootstrap_ci95=np.quantile(bootstrap, [.025, .975]).tolist(),
                instance_deltas=list(map(float, values)))


def analyze(rows, plan):
    indexed = {(r["instance"], r["method"], r["seed"]): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != set(expected_runs(plan)):
        raise ValueError("Complete unique replicated runs required")
    if any(r["error"] or not r["validation_passed"] for r in rows):
        raise ValueError("Execution or validation errors block primary inference")
    values = []
    for entry in plan["entries"]:
        i = entry["instance"]
        portfolio = [indexed[i, "portfolio", seed] for seed in plan["seeds"]]
        exact = [indexed[i, "exact", 0]]
        reference = max(r["objective"] or 0. for r in portfolio + exact)
        record = dict(instance=i, cell=entry["cell"], reference=reference)
        for method, selected in (("portfolio", portfolio), ("exact", exact)):
            record[method] = dict(
                feasibility_pp=100 * float(np.mean([r["feasible"] for r in selected])),
                score_pp=float(np.mean([100*(r["objective"] or 0.)/reference if reference else 0. for r in selected])),
                auc_pp=float(np.mean([100*normalized_auc(r["history"], budget=plan["budget"], reference=reference)
                                      if reference else 0. for r in selected])))
        values.append(record)
    comparisons = {}
    for metric in plan["primary"]:
        deltas = [r["portfolio"][metric] - r["exact"][metric] for r in values]
        comparisons[metric] = blocked_infer(deltas, [r["cell"] for r in values], family=len(plan["primary"]))
    adjusted = holm_adjust([comparisons[m]["two_sided_p"] for m in plan["primary"]])
    for metric, p in zip(plan["primary"], adjusted):
        comparisons[metric]["holm_p"] = p
        comparisons[metric]["superiority_established"] = p < .05 and comparisons[metric]["simultaneous_ci95"][0] > 0
    methods = {}
    for method in ("portfolio", "exact"):
        selected = [r for r in rows if r["method"] == method]
        methods[method] = dict(runs=len(selected), feasible=sum(r["feasible"] for r in selected),
            returned_after_deadline=sum(r["runtime"] > plan["budget"] for r in selected),
            **{metric: float(np.mean([v[method][metric] for v in values]))
               for metric in ("feasibility_pp", "score_pp", "auc_pp")})
    return dict(instances=len(values), runs=len(rows), methods=methods, comparisons=comparisons,
                all_unsolved=sum(v["reference"] == 0 for v in values), instance_results=values,
                errors=0, inference="fixed-cell paired means; within-cell variance; Satterthwaite df")


def report(result, plan):
    lines = ["# Replicated constellation results", "",
             f"{result['instances']} instances; {len(plan['design'])} cells × {plan['replicates']} independent draws; {plan['budget']:g} s total.",
             f"Portfolio seeds: {plan['seeds']}; exact solver once per instance with solver seed 0.",
             "Strict validated observation credit, including model construction; all instances retained.", "",
             "| Method | Feasible runs | Score (%) | AUC (%) | Returns after deadline |",
             "|---|---:|---:|---:|---:|"]
    for method, value in result["methods"].items():
        lines.append(f"| {method} | {value['feasible']}/{value['runs']} | {value['score_pp']:.3f} | {value['auc_pp']:.3f} | {value['returned_after_deadline']} |")
    lines += ["", "| Primary contrast: portfolio minus exact | Difference (pp) | Simultaneous 95% CI | Holm p | Established |",
              "|---|---:|---|---:|---|"]
    for metric, value in result["comparisons"].items():
        lo, hi = value["simultaneous_ci95"]
        lines.append(f"| {metric} | {value['mean_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {value['holm_p']:.4g} | {value['superiority_established']} |")
    lines += ["", f"All-unsolved cases retained: {result['all_unsolved']}.",
              "Inference averages portfolio seeds within instances, then equally weights fixed design cells.",
              "AUC and stratified bootstrap sensitivity are descriptive; normalization is best observed, not optimum.",
              "Scope: this generator and curtailing battery model; no published decomposition comparator or independent mission data."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1301, 1409])
    parser.add_argument("--budget", type=float, default=60.)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--base", type=int, default=73_000_000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "ONTS_SOLVER_THREADS"):
        os.environ[name] = "1"
    plan = make_plan(args.output, replicates=args.replicates, seeds=args.seeds, budget=args.budget,
                     workers=args.workers, base=args.base,
                     design=[(24, 2, 2, 1)] if args.smoke else None)
    plan = frozen_plan(args.output, plan)
    (args.output / "protocol.md").write_bytes(PROTOCOL.read_bytes())
    if args.plan_only:
        return
    previous = read_jsonl(args.output / "runs.jsonl")
    done = {r["run_id"] for r in previous}
    if len(done) != len(previous):
        raise ValueError("Duplicate saved run IDs")
    tasks = [(i, method, seed, plan, str(args.output.resolve())) for i, method, seed in expected_runs(plan)
             if f"{i}|{method}|{seed}" not in done]
    order = np.random.default_rng(plan["order_seed"]).permutation(len(tasks))
    print(f"Frozen {len(tasks)} runs, {args.workers} workers", flush=True)
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run, tasks[int(k)]): tasks[int(k)][:3] for k in order}
        for number, future in enumerate(as_completed(futures), 1):
            row = future.result()
            append_record(args.output / "runs.jsonl", row)
            print(f"Completed {number}/{len(tasks)} {futures[future]} error={row['error']}", flush=True)
    if source_hashes() != plan["source_hashes"] or input_hashes() != plan["bundled_input_hashes"]:
        raise RuntimeError("Sources or inputs changed during execution")
    result = analyze(read_jsonl(args.output / "runs.jsonl"), plan)
    save(args.output / "analysis.json", result)
    content = report(result, plan)
    (args.output / "report.md").write_text(content)
    print(content, flush=True)


if __name__ == "__main__":
    main()
