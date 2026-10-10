"""Paired development and fresh-instance evaluation of repair/surplus changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from ..profile_portfolio import solve_profile_portfolio
from .confirmatory import infer
from .datasets import instance_hash, source_hashes
from .portfolio_ablation import COMMON, frozen_plan, save, verify
from .protocol import append_record, objectives_at_budget, read_jsonl
from .realistic import realistic_instance
from .statistics import holm_adjust, normalized_auc

VARIANTS = {
    'legacy': {},
    'no_repairs': {'enable_repairs':False},
    'reserve_only': {'surplus_weight':0.,'accept_equal':True},
    'time_window': {'repair_strategy':'time_window'},
    'surplus_reserve': {'surplus_mode':'curtailment','accept_equal':True},
    'combined': {'repair_strategy':'time_window','surplus_mode':'curtailment','accept_equal':True},
}
COMPARISONS = [('time_window','legacy'),('surplus_reserve','legacy'),('combined','legacy'),
               ('time_window','no_repairs'),('surplus_reserve','reserve_only')]


def plan_for(phase, workers):
    count, seeds, budget, base = ((12,[307],10.,64_000_000) if phase=='pilot'
                                else (40,[401,503],30.,63_000_000))
    return dict(format='onts-component-refinements-v3',phase=phase,count=count,seeds=seeds,
        budget=budget,base=base,initialization_budget=9.,repair_time=1.,workers=workers,
        common=COMMON,variants=VARIANTS,comparisons=[list(pair) for pair in COMPARISONS],order_seed=711019,
        source_hashes=source_hashes(),instance_hashes=[instance_hash(realistic_instance(base+i)) for i in range(count)],
        protocol_sha256=hashlib.sha256(Path(__file__).with_name('COMPONENT_REFINEMENTS.md').read_bytes()).hexdigest())


def run_pair(task):
    i,seed,plan,outstring=task
    os.environ['ONTS_SOLVER_THREADS']='1'
    torch.set_num_threads(1)
    if source_hashes()!=plan['source_hashes']:raise RuntimeError('Source changed after freeze')
    out=Path(outstring);instance=realistic_instance(plan['base']+i)
    assert instance_hash(instance)==plan['instance_hashes'][i]
    path=out/'initializers'/f'{i}-{seed}.json'
    if path.exists():prefix=json.loads(path.read_text())
    else:
        result=solve_profile_portfolio(instance,time_limit=9.,max_iterations=0,seed=seed,
            repair_time=1.,**dict(COMMON,init_fraction=1.))
        value=verify(instance,result.schedule)
        usable=result.schedule is not None and result.runtime<=9.
        prefix=dict(schedule=result.schedule.int().tolist() if usable else None,
            objective=value if usable else None,runtime=result.runtime,instance_hash=instance_hash(instance))
        path.parent.mkdir(exist_ok=True);save(path,prefix)
    start=torch.tensor(prefix['schedule'],dtype=torch.bool) if prefix['schedule'] is not None else None
    if start is not None:verify(instance,start)
    fingerprint=hashlib.sha256(json.dumps(prefix['schedule']).encode()).hexdigest()
    methods=list(VARIANTS);np.random.default_rng(plan['order_seed']+i*1000+seed).shuffle(methods)
    done={r['run_id'] for r in read_jsonl(out/'runs.jsonl')};rows=[]
    for method in methods:
        run_id=f'{i}|{seed}|{method}'
        if run_id in done:continue
        row=dict(run_id=run_id,instance=i,seed=seed,method=method,shared_start_hash=fingerprint,
            instance_hash=instance_hash(instance),objective=None,returned_objective=None,
            feasible=False,history=[],returned_schedule=None,error=None,validation_passed=True,
            initial_objective=prefix['objective'],status='shared_initialization_failed')
        if start is not None:
            try:
                result=solve_profile_portfolio(instance,initial_schedule=start,seed=seed,
                    time_limit=plan['budget'],repair_time=1.,**dict(COMMON,**VARIANTS[method]))
                value=verify(instance,result.schedule);payload=result.to_dict()
                credited=objectives_at_budget(payload,plan['budget'])
                row.update(objective=credited,returned_objective=value,feasible=credited is not None,
                    history=result.history,returned_schedule=payload['schedule'],runtime=result.runtime,status=result.status)
            except Exception as exc:
                row.update(error=f'{type(exc).__name__}: {exc}',validation_passed=False)
        rows.append(row)
    return rows


def analyze(rows,plan):
    expected={(i,s,m) for i in range(plan['count']) for s in plan['seeds'] for m in VARIANTS}
    indexed={(r['instance'],r['seed'],r['method']):r for r in rows}
    if len(indexed)!=len(rows) or set(indexed)!=expected:raise ValueError('Complete unique records required')
    refs={i:max(r['objective'] or 0. for r in rows if r['instance']==i) for i in range(plan['count'])}
    score={k:100*(r['objective'] or 0.)/refs[k[0]] if refs[k[0]] else 0. for k,r in indexed.items()}
    auc={k:100*normalized_auc(r['history'],budget=plan['budget'],reference=refs[k[0]]) if refs[k[0]] else 0.
         for k,r in indexed.items()}
    methods={}
    for m in VARIANTS:
        selected=[r for r in rows if r['method']==m]
        diagnostics={op:dict(calls=0,improvements=0,gain=0.,seconds=0.,reserve_moves=0) for op in ['profile','repair']}
        for r in selected:
            old=r['initial_objective']
            for e in r['history']:
                op=e.get('operator');v=e.get('objective')
                if op in diagnostics:
                    d=diagnostics[op];d['calls']+=1;d['seconds']+=e.get('solve_runtime',0.)
                    gain=max(0.,v-old) if v is not None and old is not None else 0.
                    d['improvements']+=gain>1e-9;d['gain']+=gain
                    if op=='profile' and gain<=1e-9 and e.get('changed_cells',0)>0:
                        assert e.get('reserve_gain',0.)>0
                        d['reserve_moves']+=1
                if v is not None:old=v
        keys=[(r['instance'],r['seed'],m) for r in selected]
        methods[m]=dict(feasible=sum(r['feasible'] for r in selected),runs=len(selected),
            score_pp=float(np.mean([score[k] for k in keys])),auc_pp=float(np.mean([auc[k] for k in keys])),
            errors=sum(bool(r['error']) for r in selected),operators=diagnostics)
    comparisons=[]
    for a,b in COMPARISONS:
        c=dict(method=a,baseline=b)
        for name,metric in [('score_pp',score),('auc_pp',auc)]:
            deltas=[float(np.mean([metric[i,s,a]-metric[i,s,b] for s in plan['seeds']])) for i in range(plan['count'])]
            c[name]=infer(deltas,family=len(COMPARISONS))
            c[name].update(wins=sum(d>1e-8 for d in deltas),losses=sum(d < -1e-8 for d in deltas),
                           ties=sum(abs(d)<=1e-8 for d in deltas))
        comparisons.append(c)
    for metric in ['score_pp','auc_pp']:
        for c,p in zip(comparisons,holm_adjust([c[metric]['two_sided_p'] for c in comparisons])):c[metric]['holm_p']=p
    for i,s,_ in expected:
        assert len({indexed[i,s,m]['shared_start_hash'] for m in VARIANTS})==1
    return dict(instances=plan['count'],runs=len(rows),all_unsolved=sum(v<=0 for v in refs.values()),
        methods=methods,comparisons=comparisons,errors=sum(bool(r['error']) for r in rows))


def report(result,plan):
    lines=[f"# Component refinements: {plan['phase']}", '',
        f"{plan['count']} instances, paired seeds {plan['seeds']}, {plan['budget']:g} s improvement budget.",
        'All-instance scores; shared initialization excluded. Intervals simultaneous over five comparisons.', '',
        '| Method | Feasible | Score (%) | AUC (%) | Repair improvements/calls | Repair seconds | Reserve moves |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for m,v in result['methods'].items():
        d=v['operators']['repair'];lines.append(f"| {m} | {v['feasible']}/{v['runs']} | {v['score_pp']:.3f} | {v['auc_pp']:.3f} | {d['improvements']}/{d['calls']} | {d['seconds']:.1f} | {v['operators']['profile']['reserve_moves']} |")
    lines+=['','| Comparison | Score delta (pp) | Simultaneous 95% CI | Holm p | Wins/ties/losses |',
        '|---|---:|---|---:|---|']
    for c in result['comparisons']:
        v=c['score_pp'];ci=v['simultaneous_ci95'];lines.append(f"| {c['method']} − {c['baseline']} | {v['mean_percent']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | {v['holm_p']:.4g} | {v['wins']}/{v['ties']}/{v['losses']} |")
    lines+=['',f"Recorded execution errors: {result['errors']}. All-unsolved instances retained: {result['all_unsolved']}.",
        'Operator diagnostics are descriptive and path-dependent. This study does not evaluate cold-start feasibility.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','fresh'],required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args()
    if args.workers<1:p.error('workers must be positive')
    plan=frozen_plan(args.output,plan_for(args.phase,args.workers))
    (args.output/'protocol.md').write_bytes(Path(__file__).with_name('COMPONENT_REFINEMENTS.md').read_bytes())
    previous=read_jsonl(args.output/'runs.jsonl');done={r['run_id'] for r in previous}
    if len(done)!=len(previous):raise ValueError('Duplicate saved IDs')
    tasks=[(i,s,plan,str(args.output.resolve())) for i in range(plan['count']) for s in plan['seeds']
           if any(f'{i}|{s}|{m}' not in done for m in VARIANTS)]
    order=np.random.default_rng(plan['order_seed']).permutation(len(tasks))
    print(f"Frozen {len(tasks)} pairs, six variants each",flush=True)
    with ProcessPoolExecutor(args.workers,max_tasks_per_child=1) as pool:
        futures={pool.submit(run_pair,tasks[int(k)]):tasks[int(k)] for k in order}
        for f in as_completed(futures):
            for row in f.result():append_record(args.output/'runs.jsonl',row)
            task=futures[f];print(f'Completed instance {task[0]}, seed {task[1]}',flush=True)
    if source_hashes()!=plan['source_hashes']:raise RuntimeError('Source changed during experiment')
    result=analyze(read_jsonl(args.output/'runs.jsonl'),plan);save(args.output/'analysis.json',result)
    content=report(result,plan);(args.output/'report.md').write_text(content);print(content,flush=True)


if __name__=='__main__':main()
