"""Controlled component study with paired seeds and observed incumbent times.

Run: python -m projects.scheduling.research.portfolio_ablation --output DIR
The output freezes the protocol, sources and instance hashes before execution.
Historical ablation records are incompatible with this format.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..profile_portfolio import solve_profile_portfolio
from .confirmatory import infer
from .datasets import instance_hash, source_hashes
from .oracle import oracle_check
from .protocol import append_record, environment_metadata, objectives_at_budget, read_jsonl
from .realistic import realistic_instance
from .statistics import holm_adjust, normalized_auc

BASE = 61_000_000
COMMON = dict(controller="stagnation", patience=10, job_selection="rules",
              fraction=.2, init="local_search", polish_after_feasible=.5,
              surplus_weight=.05, observe_solver_incumbents=True)
VARIANTS = {
    "improvement/full": {},
    "improvement/no_repairs": {"enable_repairs": False},
    "improvement/no_surplus": {"surplus_weight": 0.},
    "cold/full": {},
    "cold/no_early_stop": {"polish_after_feasible": None},
    "cold/exact": None,
}
PRIMARY = [("improvement", "no_repairs"), ("improvement", "no_surplus"),
           ("cold", "no_early_stop")]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def make_plan(count=40, seeds=(101, 211), budget=30., workers=8):
    if count < 2 or len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError("At least two instances and unique paired seeds required")
    if not math.isfinite(budget) or budget <= 0 or workers < 1:
        raise ValueError("Positive finite budget and positive worker count required")
    return dict(format="onts-controlled-components-v2", count=count, seeds=list(seeds),
                budget=budget, initialization_budget=.3 * budget, workers=workers,
                repair_time=min(1., .1 * budget), base=BASE, variants=VARIANTS,
                common=COMMON, primary=[list(x) for x in PRIMARY],
                order_seed=9127, source_hashes=source_hashes(),
                protocol_sha256=hashlib.sha256(Path(__file__).with_name("CONTROLLED_ABLATION.md").read_bytes()).hexdigest(),
                instance_hashes=[instance_hash(realistic_instance(BASE+i)) for i in range(count)],
                inference_unit="instance mean over paired seeds",
                score="best observed per instance and track; failures zero; all instances retained",
                deadline="validated observations at time <= budget, no tolerance",
                improvement_track="common saved initializer; independent improvement budget; initialization excluded",
                cold_track="end-to-end budget including model construction and validation",
                timing="post-validation wall-clock observations; solver callbacks or explicit return fallback",
                scope="exploratory component study on previously inspected bundled-like instances")


def frozen_plan(output, plan):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "plan.json"
    if path.exists():
        if json.loads(path.read_text()) != plan:
            raise ValueError("Resume requires identical protocol, sources, instances and worker count")
    else:
        if (output / "runs.jsonl").exists():
            raise ValueError("Cannot import historical runs without a matching frozen plan")
        save(path, plan)
        save(output / "environment.json", environment_metadata())
        snapshot = output / "source"
        root = Path(__file__).resolve().parents[1]
        for relative, digest in plan["source_hashes"].items():
            data = (root / relative).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Source changed while freezing")
            target = snapshot / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return plan


def error_row(index, seed, key, budget, error):
    track, method = key.split("/")
    return dict(run_id=f"{index}|{seed}|{key}", instance=index, seed=seed,
                track=track, method=method, budget=budget, objective=None,
                feasible=False, history=[], time_to_first_feasible=None,
                error=error, validation_passed=False)


def verify(instance, schedule):
    if schedule is None:
        return None
    valid, value, violations = oracle_check(onts_instance_to_json_data(instance),
                                           schedule.int().tolist(), ONTSConfig())
    if not valid:
        raise RuntimeError(f"Independent oracle rejected schedule: {violations}")
    return value


def run_pair(task):
    index, seed, plan, output_string = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    output = Path(output_string)
    if source_hashes() != plan["source_hashes"]:
        raise RuntimeError("Source changed after protocol freeze")
    instance = realistic_instance(plan["base"] + index)
    if instance_hash(instance) != plan["instance_hashes"][index]:
        raise RuntimeError("Generated instance differs from frozen hash")
    cache = output / "initializers" / f"{index}-{seed}.json"
    if cache.exists():
        prefix = json.loads(cache.read_text())
    else:
        initialized = solve_profile_portfolio(
            instance, time_limit=plan["initialization_budget"], max_iterations=0,
            seed=seed, repair_time=plan["repair_time"], **dict(COMMON, init_fraction=1.0))
        value = verify(instance, initialized.schedule)
        # A shared starting schedule must be returned and validated within its allowance.
        usable = initialized.schedule is not None and initialized.runtime <= plan["initialization_budget"]
        prefix = dict(schedule=initialized.schedule.int().tolist() if usable else None,
                      objective=value if usable else None, runtime=initialized.runtime,
                      time_to_first_feasible=initialized.time_to_first_feasible,
                      initialization_complete_at=initialized.initialization_complete_at,
                      first_feasible_source=initialized.first_feasible_source,
                      instance_hash=plan["instance_hashes"][index], history=initialized.history)
        cache.parent.mkdir(exist_ok=True)
        save(cache, prefix)
    start = torch.tensor(prefix["schedule"], dtype=torch.bool) if prefix["schedule"] is not None else None
    if start is not None:
        verify(instance, start)
    start_hash = hashlib.sha256(json.dumps(prefix["schedule"]).encode()).hexdigest()
    existing = read_jsonl(output / "runs.jsonl")
    done = {r["run_id"] for r in existing}
    keys = list(VARIANTS)
    rng = np.random.default_rng(plan["order_seed"] + index * 1000 + seed)
    rng.shuffle(keys)
    rows = []
    for key in keys:
        if f"{index}|{seed}|{key}" in done:
            continue
        track, method = key.split("/")
        row = error_row(index, seed, key, plan["budget"], None)
        row.update(instance_hash=plan["instance_hashes"][index], J=instance.J, T=instance.T)
        try:
            if track == "improvement" and start is None:
                row.update(error=None, validation_passed=True,
                           status="shared_initialization_failed", shared_start_hash=start_hash)
                rows.append(row)
                continue
            if key == "cold/exact":
                began = perf_counter()
                model = ONTSMilp(instance)
                offset = perf_counter() - began
                remaining = plan["budget"] - offset
                if remaining <= 0:
                    row.update(error=None, validation_passed=True, status="budget_exhausted_in_construction")
                    rows.append(row)
                    continue
                result = model.solve(time_limit=remaining, solver_threads=1, observe_incumbents=True)
                value = verify(instance, result.schedule)
                history = [dict(e, time=offset+e["time"], event="solver_incumbent")
                           for e in result.incumbent_history]
                payload = dict(history=history, runtime=perf_counter()-began,
                               initialization_complete_at=None,
                               first_feasible_source=result.first_feasible_source,
                               schedule=result.schedule.int().tolist() if result.schedule is not None else None)
            else:
                result = solve_profile_portfolio(
                    instance, time_limit=plan["budget"], repair_time=plan["repair_time"],
                    seed=seed, initial_schedule=start if track == "improvement" else None,
                    **dict(COMMON, **VARIANTS[key]))
                value = verify(instance, result.schedule)
                payload = result.to_dict()
            credited = objectives_at_budget(payload, plan["budget"])
            observations = [e for e in payload["history"]
                            if e.get("objective") is not None and e["time"] <= plan["budget"]]
            first = min((e["time"] for e in observations), default=None)
            row.update(objective=credited, returned_objective=value, feasible=credited is not None,
                       runtime=payload["runtime"], history=payload["history"],
                       time_to_first_feasible=first,
                       first_feasible_source=payload["first_feasible_source"],
                       initialization_complete_at=payload["initialization_complete_at"],
                       validation_passed=True, error=None,
                       shared_start_hash=start_hash if track == "improvement" else None,
                       returned_schedule=payload["schedule"])
        except Exception as exc:
            row.update(error=f"{type(exc).__name__}: {exc}")
        rows.append(row)
    return rows


def analyze(rows, plan):
    expected = {(i, s, *k.split('/')) for i in range(plan['count'])
                for s in plan['seeds'] for k in VARIANTS}
    indexed = {(r['instance'], r['seed'], r['track'], r['method']): r for r in rows}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Complete unique paired records required")
    count, seeds, budget = plan['count'], plan['seeds'], plan['budget']
    references = {(i, track): max((r['objective'] or 0.) for r in rows
                                 if r['instance'] == i and r['track'] == track)
                  for i in range(count) for track in ('improvement', 'cold')}
    scores, auc, capped_first = {}, {}, {}
    for key, row in indexed.items():
        reference = references[key[0], key[2]]
        scores[key] = (row['objective'] or 0.) / reference if reference > 0 else 0.
        auc[key] = normalized_auc(row['history'], budget=budget, reference=reference) if reference > 0 else 0.
        first = row['time_to_first_feasible']
        capped_first[key] = first if first is not None else budget
        if first is not None:
            assert 0 <= first <= budget
        if row['feasible']:
            assert row['validation_passed'] and row['objective'] == objectives_at_budget(row,budget)
    summaries = {}
    for key in VARIANTS:
        track, method = key.split('/')
        selected = [indexed[i,s,track,method] for i in range(count) for s in seeds]
        mean = lambda values: float(np.mean([values[i,s,track,method] for i in range(count) for s in seeds]))
        firsts = [r['time_to_first_feasible'] for r in selected if r['feasible']]
        init = [r.get('initialization_complete_at') for r in selected if r.get('initialization_complete_at') is not None]
        kept = [i for i in range(count) if references[i,track] > 0]
        summaries[key] = dict(runs=len(selected), feasible=sum(r['feasible'] for r in selected),
            errors=sum(bool(r['error']) for r in selected), mean_score_percent=100*mean(scores),
            conditional_score_percent=100*float(np.mean([scores[i,s,track,method] for i in kept for s in seeds])) if kept else None,
            mean_auc_percent=100*mean(auc), mean_capped_first=mean(capped_first),
            median_first_feasible=float(np.median(firsts)) if firsts else None,
            median_initialization_complete=float(np.median(init)) if init else None,
            timing_sources=sorted({r.get('first_feasible_source') for r in selected if r.get('first_feasible_source')}))
    comparisons = []
    for track, baseline in PRIMARY:
        result = dict(track=track, baseline=baseline)
        for metric, values, scale in [('score_pp',scores,100),('auc_pp',auc,100),('capped_first_seconds',capped_first,1)]:
            differences = [scale*float(np.mean([values[i,s,track,'full']-values[i,s,track,baseline] for s in seeds])) for i in range(count)]
            result[metric] = infer(differences, family=len(PRIMARY))
        comparisons.append(result)
    for metric in ('score_pp','auc_pp','capped_first_seconds'):
        adjusted = holm_adjust([c[metric]['two_sided_p'] for c in comparisons])
        for c, adjusted_p in zip(comparisons,adjusted):
            c[metric]['holm_p'] = adjusted_p
    paired = []
    for track, baseline in PRIMARY:
        if track != 'improvement': continue
        assert all(indexed[i,s,track,'full']['shared_start_hash'] == indexed[i,s,track,baseline]['shared_start_hash'] for i in range(count) for s in seeds)
        paired.append(f'{track}/full and {baseline}: shared starts match')
    return dict(instances=count, seeds=seeds, runs=len(rows), methods=summaries,
                comparisons=comparisons, pairing_checks=paired,
                all_unsolved={track:sum(references[i,track]<=0 for i in range(count)) for track in ('improvement','cold')},
                errors=sum(bool(r['error']) for r in rows),
                timing_note='improvement time starts at supplied schedule (zero); cold times are validated discoveries',
                inference_note='score comparisons are primary; AUC and capped-first comparisons are secondary; exploratory study')


def report(analysis):
    lines=['# Controlled component study','',
           'All-instance scores retain failed runs as zero. Normalization uses best observed objectives, not optima.', '',
           '| Track/method | Feasible/runs | Score (%) | AUC (%) | First (s), conditional | Initialization done (s) |',
           '|---|---:|---:|---:|---:|---:|']
    for name,v in analysis['methods'].items():
        number=lambda x: '—' if x is None else f'{x:.3f}'
        lines.append(f"| {name} | {v['feasible']}/{v['runs']} | {v['mean_score_percent']:.3f} | {v['mean_auc_percent']:.3f} | {number(v['median_first_feasible'])} | {number(v['median_initialization_complete'])} |")
    lines += ['',f"Errors: {analysis['errors']}. All-unsolved instances: {analysis['all_unsolved']}.",
              '', 'Primary paired differences: full minus ablated, instance means over seeds.',
              '| Comparison | Score difference (pp) | Simultaneous 95% CI | Holm p |',
              '|---|---:|---|---:|']
    for c in analysis['comparisons']:
        v=c['score_pp'];lo,hi=v['simultaneous_ci95']
        lines.append(f"| {c['track']}/full − {c['baseline']} | {v['mean_percent']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {v['holm_p']:.3g} |")
    lines += ['',analysis['timing_note'],analysis['inference_note']]
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--count',type=int,default=40)
    parser.add_argument('--seeds',type=int,nargs='+',default=[101,211])
    parser.add_argument('--budget',type=float,default=30.)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args()
    plan=frozen_plan(args.output,make_plan(args.count,args.seeds,args.budget,args.workers))
    if args.plan_only: return
    previous=read_jsonl(args.output/'runs.jsonl')
    done={r['run_id'] for r in previous}
    if len(done)!=len(previous): raise ValueError('Duplicate saved run IDs')
    tasks=[(i,s,plan,str(args.output.resolve())) for i in range(args.count) for s in args.seeds
           if any(f'{i}|{s}|{key}' not in done for key in VARIANTS)]
    order=np.random.default_rng(plan['order_seed']).permutation(len(tasks))
    print(f'Frozen {len(tasks)} pairs, {len(VARIANTS)} variants per pair; {len(done)} saved runs',flush=True)
    with ProcessPoolExecutor(args.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(run_pair,tasks[int(k)]):tasks[int(k)] for k in order}
        for future in as_completed(futures):
            task=futures[future]
            # A worker crash is a resumable interruption, not an invented zero run.
            rows=future.result()
            for row in rows: append_record(args.output/'runs.jsonl',row)
            print(f'Completed instance {task[0]}, seed {task[1]}: {len(rows)} records',flush=True)
    if source_hashes() != plan['source_hashes']:
        raise RuntimeError('Source changed during study; analysis refused')
    analysis=analyze(read_jsonl(args.output/'runs.jsonl'),plan)
    save(args.output/'analysis.json',analysis)
    (args.output/'report.md').write_text(report(analysis))
    print(report(analysis),flush=True)


if __name__=='__main__': main()
