"""Fixed-size confirmation of portfolio v2 on bundled-like instances.

Protocol: research/PORTFOLIO_V2_CONFIRMATORY.md. No tuning or optional stopping.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import onts_instance_to_json_data
from ..hybrid import solve_hybrid
from ..milp import ONTSConfig, ONTSMilp
from ..profile_portfolio import solve_profile_portfolio
from ..ranked_search import solve_ranked
from .confirmatory import infer, save
from .datasets import instance_hash, source_hashes
from .oracle import oracle_check
from .protocol import append_record, objectives_at_budget, read_jsonl
from .realistic import realistic_instance
from .statistics import holm_adjust

METHODS = ("v2", "v1", "heuristic", "uniform", "exact")
PRIMARY = ("heuristic", "uniform", "exact")
SEEDS = (101, 211)
BUDGET, REPAIR, BASE, COUNT = 30.0, 1.0, 61_000_000, 40
V1 = dict(controller="stagnation", patience=10, job_selection="rules", fraction=0.2)
V2 = dict(V1, polish_after_feasible=0.5, surplus_weight=0.05)


def run(task):
    index, seed, method = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    instance = realistic_instance(BASE + index)
    row = dict(
        run_id=f"{index}|{seed}|{method}",
        instance=index,
        method=method,
        seed=seed,
        J=instance.J,
        T=instance.T,
    )
    try:
        if method == "exact":
            result = ONTSMilp(instance).solve(time_limit=BUDGET)
            schedule = result.schedule
            payload = dict(
                schedule=None if schedule is None else schedule.int().tolist(),
                objective=result.objective,
                runtime=result.runtime,
                time_to_first_feasible=result.runtime if schedule is not None else None,
            )
            credited = result.objective if result.runtime <= BUDGET + 0.5 else None
        else:
            if method in ("v1", "v2"):
                result = solve_profile_portfolio(
                    instance,
                    time_limit=BUDGET,
                    repair_time=REPAIR,
                    seed=seed,
                    **(V2 if method == "v2" else V1),
                )
            elif method == "heuristic":
                result = solve_hybrid(
                    instance,
                    selector="heuristic",
                    time_limit=BUDGET,
                    repair_time=REPAIR,
                    neighborhood_fraction=0.5,
                    seed=seed,
                )
            else:
                result = solve_ranked(
                    instance,
                    mode="uniform",
                    time_limit=BUDGET,
                    repair_time=REPAIR,
                    fraction=0.2,
                    seed=seed,
                )
            payload = result.to_dict()
            credited = objectives_at_budget(payload, BUDGET)
        if payload["schedule"] is not None:
            valid, objective, _ = oracle_check(
                onts_instance_to_json_data(instance), payload["schedule"], ONTSConfig()
            )
            if not valid or not np.isclose(objective, payload["objective"], atol=1e-4):
                raise RuntimeError("Independent output validation failed")
        row.update(
            objective=credited,
            feasible=credited is not None,
            runtime=payload["runtime"],
            time_to_first_feasible=payload["time_to_first_feasible"],
            validation_passed=True,
        )
    except Exception as exc:
        row.update(
            error=f"{type(exc).__name__}: {exc}",
            objective=None,
            feasible=False,
            validation_passed=False,
        )
    return row


def analyze(rows):
    indexed = {(r["instance"], r["seed"], r["method"]): r for r in rows}
    expected = {(i, s, m) for i in range(COUNT) for s in SEEDS for m in METHODS}
    if set(indexed) != expected or len(indexed) != len(rows):
        raise ValueError("Complete unique paired records required before inference")
    best = {
        i: max((r["objective"] or 0.0) for k, r in indexed.items() if k[0] == i)
        for i in range(COUNT)
    }
    kept = [i for i in range(COUNT) if best[i] > 0]

    def score(i, m):
        return float(
            np.mean([(indexed[i, s, m]["objective"] or 0.0) / best[i] for s in SEEDS])
        )

    comparisons = []
    for baseline in METHODS[1:]:
        deltas = [100 * (score(i, "v2") - score(i, baseline)) for i in kept]
        comparisons.append(
            dict(
                baseline=baseline,
                primary=baseline in PRIMARY,
                **infer(deltas, family=len(PRIMARY)),
            )
        )
    primary = [c for c in comparisons if c["primary"]]
    for c, p in zip(primary, holm_adjust([c["two_sided_p"] for c in primary])):
        c["holm_p"] = p
        c["superiority"] = p < 0.05 and c["simultaneous_ci95"][0] > 0
    methods = {}
    for m in METHODS:
        runs = [r for r in rows if r["method"] == m]
        firsts = [r["time_to_first_feasible"] for r in runs if r["feasible"]]
        methods[m] = dict(
            feasible_runs=sum(r["feasible"] for r in runs),
            runs=len(runs),
            mean_score_percent=100 * float(np.mean([score(i, m) for i in kept])),
            median_first_feasible=float(np.median(firsts)) if firsts else None,
            errors=sum(bool(r.get("error")) for r in runs),
        )
    return dict(
        instances=COUNT,
        instances_with_feasible=len(kept),
        excluded_no_feasible=COUNT - len(kept),
        methods=methods,
        comparisons=comparisons,
    )


def report(a, out):
    lines = [
        "# Portfolio v2 confirmatory results",
        "",
        f"{a['instances_with_feasible']}/{a['instances']} instances had a feasible "
        f"schedule from some run ({a['excluded_no_feasible']} excluded).",
        "",
        "| Method | Feasible runs | Mean normalized score | Median first feasible | Errors |",
        "|---|---:|---:|---:|---:|",
    ]
    for m, v in a["methods"].items():
        first = (
            f"{v['median_first_feasible']:.2f}s" if v["median_first_feasible"] else "—"
        )
        lines.append(
            f"| {m} | {v['feasible_runs']}/{v['runs']} | {v['mean_score_percent']:.2f}% "
            f"| {first} | {v['errors']} |"
        )
    lines += [
        "",
        "| v2 vs | Primary | Mean diff (pp) | Simultaneous 95% CI | Holm p | W/L/T | Superiority |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    for c in a["comparisons"]:
        lo, hi = c["simultaneous_ci95"]
        holm = f"{c['holm_p']:.2e}" if "holm_p" in c else "—"
        lines.append(
            f"| {c['baseline']} | {c['primary']} | {c['mean_percent']:+.3f} | "
            f"[{lo:+.3f}, {hi:+.3f}] | {holm} | {c['wins']}/{c['losses']}/{c['ties']} "
            f"| {c.get('superiority', '—')} |"
        )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    if (out / "plan.json").exists():
        plan = json.loads((out / "plan.json").read_text())
        if source_hashes() != plan["source_hashes"]:
            raise RuntimeError("Frozen source changed; do not resume this study")
    else:
        hashes = source_hashes()
        with tarfile.open(out / "frozen-source.tar.gz", "w:gz") as tar:
            for rel in hashes:
                data = (root / rel).read_bytes()
                info = tarfile.TarInfo(rel)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        plan = dict(
            format="onts-portfolio-v2-confirmatory-v1",
            source_hashes=hashes,
            v2=V2,
            v1=V1,
            methods=METHODS,
            primary=PRIMARY,
            seeds=SEEDS,
            budget=BUDGET,
            repair_time=REPAIR,
            instance_seeds=[BASE, BASE + COUNT - 1],
            instance_hashes=[
                instance_hash(realistic_instance(BASE + i)) for i in range(COUNT)
            ],
            protocol="research/PORTFOLIO_V2_CONFIRMATORY.md",
        )
        save(out / "plan.json", plan)
    done = {r["run_id"] for r in read_jsonl(out / "runs.jsonl")}
    tasks = [
        (i, s, m)
        for i in range(COUNT)
        for s in SEEDS
        for m in METHODS
        if f"{i}|{s}|{m}" not in done
    ]
    order = np.random.default_rng(9127).permutation(len(tasks))
    with ProcessPoolExecutor(args.workers) as pool:
        for row in pool.map(run, [tasks[int(k)] for k in order]):
            append_record(out / "runs.jsonl", row)
    analysis = analyze(read_jsonl(out / "runs.jsonl"))
    save(out / "analysis.json", analysis)
    report(analysis, out)


if __name__ == "__main__":
    main()
