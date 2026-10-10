"""Independent artifact/feasibility/accounting audit of the literature study."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import tarfile
from pathlib import Path
import numpy as np
import torch
from ..environments.onts_env import ONTSInstance
from ..milp import ONTSConfig
from .datasets import instance_hash


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text())
def close(a,b): assert math.isclose(float(a),float(b),rel_tol=1e-8,abs_tol=1e-7),(a,b)


def audit(output):
    config=ONTSConfig();plan=read(output/'plan.json');manifest=read(output/'manifest.json')
    stored=torch.load(output/'instances.pt',map_location='cpu',weights_only=True)
    data={};hashes=[]
    for split,entries in stored.items():
        data[split]=[(ONTSInstance(**params),initial) for params,initial in entries]
        assert len(entries)==len(manifest[split])
        for i,(inst,x) in enumerate(data[split]):
            m=manifest[split][i];assert m['index']==i and m['J']==inst.J and m['T']==inst.T
            h=instance_hash(inst);assert m['hash']==h;hashes.append(h)
            assert config.validate(inst,x).feasible
    assert len(set(hashes))==len(hashes)
    for other in output.parent.glob('rl-*/manifest.json'):
        if other.parent.resolve()!=output.resolve():
            assert all(h not in other.read_text() for h in hashes),other
    root=Path(__file__).resolve().parents[1]
    assert sha(output/'source.tar.gz')==plan['source_archive_sha256']
    with tarfile.open(output/'source.tar.gz') as archive:
        assert set(archive.getnames())==set(plan['source_hashes'])
        for path,h in plan['source_hashes'].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest()==h,path
            assert sha(root/path)==h,path
    checkpoints=read(output/'checkpoint-hashes.json');assert len(checkpoints)==16
    for name,h in checkpoints.items():
        assert sha(output/name)==h
        saved=torch.load(output/name,map_location='cpu',weights_only=True)
        assert isinstance(saved['state_dict'],dict)
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values())
    totals=dict(proposals=0,late_feasible=0,transitions=0,feasible_decreases=0,teacher_outcomes=0)

    def schedule(inst,raw):
        x=torch.tensor(raw)
        assert x.shape==(inst.J,inst.T) and ((x==0)|(x==1)).all()
        report=config.validate(inst,x);assert report.feasible,report.violations
        return x.bool(),report.objective

    def trajectory(inst,initial,records,budget,reported_objective,reported_schedule):
        base=initial.bool();initial_value=config.validate(inst,initial).objective
        current_value=best_value=initial_value;best=base.clone()
        scale=max(float(inst.priority.clamp(min=0).sum())*inst.T,1.)
        deadline=budget-min(.05,.05*budget);rewards=0.
        for event in records:
            totals['transitions']+=1
            close(event['before_current'],current_value);close(event['before_best'],best_value)
            eligible=[]
            for prop in event['proposals']:
                totals['proposals']+=1
                if not prop['executed']:
                    assert not prop['eligible'] and prop['schedule'] is None
                    continue
                if prop.get('schedule') is not None:
                    candidate,value=schedule(inst,prop['schedule']);close(value,prop['objective'])
                    free=torch.zeros(inst.J,dtype=torch.bool);free[prop['jobs']]=True
                    assert torch.equal(candidate[~free],base[~free])
                    ontime=prop['checked_elapsed']<=deadline+1e-12
                    assert bool(prop['eligible'])==ontime
                    if ontime:eligible.append((value,candidate))
                    else:totals['late_feasible']+=1
                else:assert not prop['eligible']
            current,value=schedule(inst,event['current_schedule']);close(value,event['current_objective'])
            newbest,newbest_value=schedule(inst,event['best_schedule']);close(newbest_value,event['best_objective'])
            assert newbest_value>=best_value-1e-7
            if event['accepted']:
                assert event['accepted_elapsed']<=deadline+1e-12
                assert eligible
                maximum=max(v for v,_ in eligible);close(value,maximum)
                assert any(torch.equal(current,x) for v,x in eligible if abs(v-maximum)<1e-7)
                if value<current_value-1e-7:totals['feasible_decreases']+=1
            else:assert torch.equal(current,base)
            if newbest_value>best_value+1e-7:
                assert any(torch.equal(newbest,x) for v,x in eligible if abs(v-newbest_value)<1e-7)
                close(newbest_value,max(v for v,_ in eligible))
            else:assert torch.equal(newbest,best)
            close(event['reward'],(newbest_value-best_value)/scale)
            assert event['reward']>=-1e-9
            rewards+=event['reward'];base=current;current_value=value;best=newbest;best_value=newbest_value
        final,value=schedule(inst,reported_schedule);close(value,reported_objective)
        assert value>=initial_value-1e-7
        if records:
            assert torch.equal(final,best);close(rewards,(value-initial_value)/scale)
        return value,initial_value,scale

    for seed in plan['seeds']:
        teacher=read(output/f'teacher-{seed}.json');assert len(teacher)==len(data['train'])
        for row in teacher:
            inst,base=data['train'][row['instance']]
            initial_value=config.validate(inst,base).objective
            scale=max(float(inst.priority.clamp(min=0).sum())*inst.T,1.)
            for outcome in row['outcomes']:
                x,value=schedule(inst,outcome['schedule']);close(value,outcome['objective'])
                free=torch.zeros(inst.J,dtype=torch.bool);free[outcome['jobs']]=True
                assert torch.equal(x[~free],base.bool()[~free]);assert value>=initial_value-1e-7
                close(outcome['gain'],(value-initial_value)/scale);totals['teacher_outcomes']+=1
    training_count=0
    for name in ('resched','mistar','rllns','dralns'):
        for seed in plan['seeds']:
            records=read(output/f'training-{name}-{seed}.json')['records']
            episodes=[r for r in records if 'episode' in r]
            assert len(episodes)==plan['episodes']
            for i,row in enumerate(episodes):
                assert row['episode']==i
                inst,x=data['train'][row['instance']]
                trajectory(inst,x,row['transitions'],plan['train_budget'],row['objective'],row['schedule'])
                training_count+=1
    rows=[json.loads(s) for s in (output/'evaluations.jsonl').read_text().splitlines()]
    expected={(split,i,seed,method) for split in ('development','ood') for i in range(len(data[split]))
              for seed in plan['seeds'] for method in plan['methods']}
    observed={(r['split'],r['instance'],r['seed'],r['method']) for r in rows}
    assert observed==expected and len(rows)==len(expected)
    for row in rows:
        inst,x=data[row['split']][row['instance']]
        if row['method']=='profile_heuristic':
            _,actual=schedule(inst,row['schedule'])
            initial=config.validate(inst,x).objective
            scale=max(float(inst.priority.clamp(min=0).sum())*inst.T,1.)
            value=row['objective'];assert initial-1e-7<=value<=actual+1e-7
            close(actual,row['returned_objective'])
            history=row['baseline_history'];last=initial
            for event in history:
                if event.get('objective') is not None:
                    assert event['objective']>=last-1e-7;last=event['objective']
            upper_cutoff=plan['budget']-min(.25,plan['budget']*.05)
            credited_candidates=[initial]+[e['objective'] for e in history if e.get('objective') is not None and e['time']<=upper_cutoff]
            assert any(abs(value-v)<1e-7 for v in credited_candidates)
            assert value<=max(credited_candidates)+1e-7
        else:
            value,initial,scale=trajectory(inst,x,row['transitions'],plan['budget'],row['objective'],row['schedule'])
        close(row['initial'],initial);close(row['improvement'],(value-initial)/scale)
        close(row['overrun'],max(0.,row['runtime']-plan['budget']))
        assert row['feasible']
        assert row['feasible_decreases']==sum(e['accepted'] and e['current_objective']<e['before_current']-1e-7 for e in row['transitions'])
    summary=read(output/'summary.json')
    for entry in summary['means']:
        selected=[r for r in rows if r['split']==entry['split'] and r['method']==entry['method']]
        close(entry['improvement'],np.mean([r['improvement'] for r in selected]))
        close(entry['runtime'],np.mean([r['runtime'] for r in selected]))
        assert entry['overruns']==sum(r['overrun']>0 for r in selected)
        assert entry['feasible_decreases']==sum(r['feasible_decreases'] for r in selected)
    for entry in summary['paired']:
        split,a,b=entry['split'],entry['method'],entry['baseline'];deltas=[]
        for i in sorted({r['instance'] for r in rows if r['split']==split}):
            vals=lambda m:[r['improvement'] for r in rows if r['split']==split and r['instance']==i and r['method']==m]
            deltas.append(np.mean(vals(a))-np.mean(vals(b)))
        rng=np.random.default_rng(821);values=np.asarray(deltas)
        means=values[rng.integers(0,len(values),(4000,len(values)))].mean(1)
        close(entry['delta']['mean'],values.mean())
        for actual,wanted in zip(entry['delta']['ci95'],np.quantile(means,[.025,.975])):close(actual,wanted)
    return dict(status='passed',audit_script_sha256=sha(Path(__file__)),evaluations=len(rows),
        training_outputs=training_count,checkpoints=len(checkpoints),**totals,
        overruns=sum(r['overrun']>0 for r in rows),max_overrun=max(r['overrun'] for r in rows),
        limitation='Full-MIP has no proposal timestamp log; only its returned schedule and quality are independently audited. Best-update exact timestamps are not stored; eligible proposal guard and acceptance timestamps are checked. Companion baseline profile_budget excludes setup and is not stored; credit is bounded by returned quality and matched to a history event within the conservative maximum cutoff.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=audit(args.output)
    (args.output/'independent-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
