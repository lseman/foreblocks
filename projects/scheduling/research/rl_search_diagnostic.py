"""Frozen training-state headroom probes and inference-only MIStar ablations."""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import platform
import tarfile
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..environments.onts_env import ONTSInstance
from ..milp import ONTSConfig, ONTSMilp
from ..job_profile import reprofile
from ..rl_alns import LiteratureSearchEnv, DRALNSActorCritic
from ..rl_mistar import MistarActorCritic
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_literature_study import bootstrap, write
from .rl_literature_continuation import evaluate

METHODS = ['profile_heuristic','mistar_il','mistar_full','mistar_nomemory',
           'mistar_single','mistar_nomemory_single','mistar_noembedding','mistar_nobias','dralns_ppo']


class MemoryIntervention(MistarActorCritic):
    zero_embedding = False

    def _retrieve(self, observation, tasks, times):
        memory, repetition, similarity = super()._retrieve(observation, tasks, times)
        return (torch.zeros_like(memory) if self.zero_embedding else memory), repetition, similarity


def variant(baseline, method, seed):
    if method == 'profile_heuristic': return None
    if method == 'dralns_ppo': return DRALNSActorCritic.load(baseline/f'dralns-ppo-{seed}.pt')
    phase = 'il' if method == 'mistar_il' else 'ppo'
    model = MemoryIntervention.load(baseline/f'mistar-{phase}-{seed}.pt')
    if method in ('mistar_nomemory','mistar_nomemory_single'): model.memory = False
    if method in ('mistar_single','mistar_nomemory_single'): model.proposal_count = 1
    if method == 'mistar_nobias': model.repetition_penalty = 0.
    if method == 'mistar_noembedding': model.zero_embedding = True
    return model


def summarize(rows):
    means, paired = [], []
    for split in ('development','ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            means.append(dict(split=split, method=method, improvement=float(np.mean([r['improvement'] for r in selected])),
                runtime=float(np.mean([r['runtime'] for r in selected])),overruns=sum(r['overrun']>0 for r in selected)))
        contrasts = [('mistar_full', b) for b in ('mistar_il','mistar_nomemory','mistar_single','mistar_noembedding','mistar_nobias','profile_heuristic')]
        contrasts += [('mistar_single','mistar_nomemory_single'),('mistar_nomemory','mistar_nomemory_single')]
        instances = sorted({r['instance'] for r in rows if r['split']==split})
        for a,b in contrasts:
            deltas = []
            for i in instances:
                quality = lambda m: np.mean([r['improvement'] for r in rows if r['split']==split and r['instance']==i and r['method']==m])
                deltas.append(quality(a)-quality(b))
            paired.append(dict(split=split,method=a,baseline=b,delta=bootstrap(deltas)))
    return dict(means=means,paired=paired)


def training_diagnostics(baseline, data):
    output, selected = [], {}
    config = ONTSConfig()
    for name in ('mistar','dralns'):
        for arm in ('ppo','replay'):
            for seed in (11,29):
                bins = {b:Counter() for b in ('1-4','5-8','9-16')}; operators=defaultdict(Counter)
                with gzip.open(baseline/f'training-{name}-{arm}-{seed}.jsonl.gz','rt') as stream:
                    for line in stream:
                        episode = json.loads(line)
                        if 'episode' not in episode: continue
                        inst,base = data[episode['instance']];base=base.bool()
                        seen = set();scale=max(float(inst.priority.clamp(min=0).sum())*inst.T,1.)
                        for event in episode['transitions']:
                            step=event['step']; bucket='1-4' if step<=4 else '5-8' if step<=8 else '9-16'
                            stats=bins[bucket];stats['decisions']+=1;stats['positive_reward']+=event['reward']>1e-9
                            state=base.numpy().tobytes();batch_seen=set()
                            for prop in event['proposals']:
                                key=(state,tuple(prop['jobs']),prop['operator'])
                                stats['proposals']+=1;stats['prior_transition_repeats']+=key in seen
                                stats['within_batch_repeats']+=key in batch_seen;batch_seen.add(key)
                                if prop['executed']:
                                    o=operators[prop['operator']];o['executed']+=1;o['eligible']+=prop['eligible']
                                    if prop['eligible']:
                                        o['improve_best']+=prop['objective']>event['before_best']+scale*1e-9
                                        o['improve_current']+=prop['objective']>event['before_current']+scale*1e-9
                            seen.update(batch_seen)
                            if name=='mistar' and arm=='ppo' and seed==11 and step>=8 and event['reward']<=1e-9 and episode['instance'] not in selected:
                                x=torch.tensor(event['current_schedule'])
                                if abs(event['current_objective']-event['best_objective'])<1e-7 and config.validate(inst,x).feasible:
                                    selected[episode['instance']]=dict(instance=episode['instance'],episode=episode['episode'],step=step,schedule=x)
                            base=torch.tensor(event['current_schedule']).bool()
                output.append(dict(model=name,arm=arm,seed=seed,bins={k:dict(v) for k,v in bins.items()},operators={k:dict(v) for k,v in operators.items()}))
    chosen = [selected[i] for i in sorted(selected) if i<24][:6] + [selected[i] for i in sorted(selected) if i>=24][:6]
    assert len(chosen)==12
    return output, chosen


def headroom(data, states):
    records=[];config=ONTSConfig()
    for number,state in enumerate(states):
        inst,_=data[state['instance']];base=state['schedule'];problem=ONTSMilp(inst)
        initial=config.validate(inst,base).objective
        env=LiteratureSearchEnv(inst,base,budget=100.,seed=8100+state['instance'])
        groups=sorted({a.jobs for a in env.actions})
        outcomes=[]
        for jobs in groups:
            rng=np.random.default_rng(8100+state['instance']);tick=perf_counter()
            candidate,_,info=reprofile(problem,base,jobs,rng,deadline=perf_counter()+2.)
            report=config.validate(inst,candidate);assert report.feasible
            outcomes.append(dict(operator='profile',jobs=list(jobs),allowance=2.,runtime=perf_counter()-tick,
                objective=report.objective,schedule=candidate.int().tolist(),status='profile',info=info))
            free=torch.zeros_like(base,dtype=torch.bool);free[list(jobs)]=True
            for allowance in (.05,.25):
                tick=perf_counter();result=problem.solve(time_limit=allowance,incumbent=base,free_mask=free,solver_threads=1)
                report=config.validate(inst,result.schedule);assert report.feasible
                outcomes.append(dict(operator='milp',jobs=list(jobs),allowance=allowance,runtime=perf_counter()-tick,
                    objective=report.objective,schedule=result.schedule.int().tolist(),status=result.status))
        tick=perf_counter();result=problem.solve(time_limit=2.,incumbent=base,solver_threads=1)
        report=config.validate(inst,result.schedule);assert report.feasible
        outcomes.append(dict(operator='milp',jobs=list(range(inst.J)),allowance=2.,runtime=perf_counter()-tick,
            objective=report.objective,schedule=result.schedule.int().tolist(),status=result.status))
        records.append(dict(instance=state['instance'],episode=state['episode'],step=state['step'],
            initial=initial,schedule=base.int().tolist(),outcomes=outcomes))
        print(f'headroom state {number+1}/{len(states)}',flush=True)
    return records


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,default=Path('projects/scheduling/research/results/rl-literature-continuation-20261006'))
    args=parser.parse_args();output=args.output;output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
    seeds=[11,29];data={};manifest={};data_seed=2026109900
    for split,count,large,offset in [('development',6,False,0),('ood',6,True,100)]:
        data[split]=[witness_instance(seed=data_seed+offset+i,regime=('balanced','energy','periodic')[i%3],large=large) for i in range(count)]
        manifest[split]=[dict(index=i,hash=instance_hash(inst),J=inst.J,T=inst.T) for i,(inst,_) in enumerate(data[split])]
    hashes=[r['hash'] for entries in manifest.values() for r in entries];assert len(hashes)==len(set(hashes))
    for old in output.parent.glob('rl-*/manifest.json'):assert not any(h in old.read_text() for h in hashes),old
    write(output/'manifest.json',manifest)
    torch.save({split:[(inst.__dict__,x) for inst,x in entries] for split,entries in data.items()},output/'instances.pt')
    root=Path(__file__).resolve().parents[1];sources=source_hashes()
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in sources:archive.add(root/path,arcname=path)
    checkpoints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.baseline.glob('*-*.pt') if p.name.startswith(('mistar-','dralns-'))}
    write(output/'plan.json',dict(seeds=seeds,methods=METHODS,budget=1.2,repair_budget=.05,data_seed=data_seed,
        baseline=str(args.baseline.resolve()),input_checkpoint_hashes=checkpoints,
        training_artifact_hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in list(args.baseline.glob('training-*.jsonl.gz'))+[args.baseline/'instances.pt']},
        source_hashes=sources,source_archive_sha256=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest(),
        intervention='Inference only, identical weights; memory-off disables both embedding and bias; noembedding preserves bias; nobias preserves embedding; single reduces proposals to one',
        headroom='12 deterministic stalled training states; 6 small, 6 larger; profile 2 s, same candidate groups MILP 0.05/0.25 s, all-task MILP 2 s; offline probe, not a paid-budget competitor',
        status='Exploratory mechanism diagnosis; no retraining or checkpoint selection; larger sizes occur in preceding training'))
    write(output/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=1))
    stored=torch.load(args.baseline/'instances.pt',weights_only=True,map_location='cpu')['train']
    training=[(ONTSInstance(**params),initial) for params,initial in stored]
    stats,states=training_diagnostics(args.baseline,training);write(output/'training-diagnostics.json',stats)
    write(output/'headroom.json',headroom(training,states))
    rows=[]
    with gzip.open(output/'evaluations.jsonl.gz','wt') as stream:
        for split in ('development','ood'):
            for i,(inst,initial) in enumerate(data[split]):
                for seed in seeds:
                    for method in np.random.default_rng(1922+i+seed+(10000 if split=='ood' else 0)).permutation(METHODS):
                        row=evaluate(str(method),variant(args.baseline,str(method),seed),inst,initial,seed,1.2,.05)
                        row.update(split=split,instance=i,seed=seed,method=str(method));rows.append(row)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(f'evaluated {split} {i+1}/{len(data[split])}',flush=True)
    summary=summarize(rows);write(output/'summary.json',summary)
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Inference-only MIStar mechanism interventions with fixed weights, on six fresh small and six larger instances at 1.2 s; two seeds.}',r'\label{tab:search-diagnostic}',r'\begin{tabular}{lrr}',r'\toprule',r'Method & Small held-out & Larger held-out \\',r'\midrule']
    for method in METHODS:
        values=[next(r['improvement'] for r in summary['means'] if r['split']==s and r['method']==method) for s in ('development','ood')]
        tex.append(f"{method.replace('_',' ')} & {values[0]:.6f} & {values[1]:.6f}"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}'];(output/'summary.tex').write_text('\n'.join(tex)+'\n')
    unchanged=all(hashlib.sha256((root/path).read_bytes()).hexdigest()==h for path,h in sources.items())
    write(output/'source-integrity.json',dict(unchanged=unchanged));assert unchanged
    print(json.dumps(summary['means'],indent=2),flush=True)


if __name__=='__main__':main()
