"""Frozen ONTS development comparison of four literature-inspired alternatives.

This verifies executable adaptations; it is not a reproduction or SOTA study.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import tarfile
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..environments.onts_env import ONTSInstance
from ..job_profile import reprofile
from ..milp import ONTSConfig, ONTSMilp
from ..rl_alns import AdaptiveLNS, DRALNSActorCritic, LiteratureSearchEnv, literature_rollout
from ..rl_learning import ppo_batch_update
from ..rl_lns import RLLNSActorCritic
from ..rl_mistar import MistarActorCritic
from ..rl_resched import ReschedActorCritic
from ..rl_subset import NonemptySubset
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_portfolio_study import evaluate as evaluate_companion


MODELS = dict(resched=ReschedActorCritic,mistar=MistarActorCritic,rllns=RLLNSActorCritic,dralns=DRALNSActorCritic)
METHODS = ['profile_heuristic','full_mip','alns','uniform_anneal','random_mistar4','random_rllns',
           *[f'{name}_{phase}' for name in MODELS for phase in ('il','ppo')]]


def write(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def bootstrap(values,seed=821):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    means=values[rng.integers(0,len(values),(4000,len(values)))].mean(1)
    return dict(mean=float(values.mean()),ci95=np.quantile(means,[.025,.975]).tolist(),instances=len(values))


def summarize(rows):
    means=[];pairs=[]
    for split in ('development','ood'):
        for method in METHODS:
            records=[r for r in rows if r['split']==split and r['method']==method]
            means.append(dict(split=split,method=method,improvement=float(np.mean([r['improvement'] for r in records])),
                runtime=float(np.mean([r['runtime'] for r in records])),
                feasible_decreases=sum(r['feasible_decreases'] for r in records),
                overruns=sum(r['overrun']>0 for r in records)))
        contrasts=[(f'{name}_ppo',f'{name}_il') for name in MODELS]
        contrasts += [('alns','uniform_anneal'),('mistar_ppo','random_mistar4'),('rllns_ppo','random_rllns')]
        contrasts += [(f'{name}_ppo','profile_heuristic') for name in MODELS]
        instances=sorted({r['instance'] for r in rows if r['split']==split})
        for a,b in contrasts:
            deltas=[np.mean([r['improvement'] for r in rows if r['split']==split and r['instance']==i and r['method']==a])-
                    np.mean([r['improvement'] for r in rows if r['split']==split and r['instance']==i and r['method']==b]) for i in instances]
            pairs.append(dict(split=split,method=a,baseline=b,delta=bootstrap(deltas)))
    return dict(means=means,paired=pairs)


def imitate(model,optimizer,examples,*,subset=False,epochs=8,seed=0):
    losses=[];rng=np.random.default_rng(seed)
    for _ in range(epochs):
        for i in rng.permutation(len(examples)):
            obs,mask,label,weight=examples[i]
            dist,_=model(obs,mask)
            if isinstance(model,MistarActorCritic):
                # Teach a single proposal; deployment draws K independent samples.
                likelihood=dist.base.log_prob(torch.tensor(label))
            else: likelihood=dist.log_prob(torch.tensor(label))
            loss=-weight*likelihood/(len(label) if subset else 1)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite imitation')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.5,error_if_nonfinite=True);optimizer.step()
            losses.append(float(loss.detach()))
    return dict(updates=len(losses),loss=float(np.mean(losses)) if losses else None)


def teacher(data,seed,repair_budget):
    examples={name:[] for name in MODELS};records=[]
    rllns=RLLNSActorCritic(hidden=8)
    for i,(inst,initial) in enumerate(data):
        env=LiteratureSearchEnv(inst,initial,budget=100.,seed=seed+i,repair_budget=repair_budget)
        obs,mask=env.observe();graph,subset_mask=rllns.observation(env)
        # Label candidates under a fresh common incumbent; no evaluation instances.
        unique={}
        for index,a in enumerate(env.actions):
            if a.temperature==0 and a.operator!='perturb':
                unique.setdefault(a.operator,[]).append(index)
        selected=[]
        rng=np.random.default_rng(seed+i)
        for operator,indices in unique.items():
            selected.extend(rng.choice(indices,min(6,len(indices)),replace=False).tolist())
        outcomes=[]
        for index in selected:
            a=env.actions[index];base=env.current.clone();tick=perf_counter()
            candidate,status=env._repair(a,base)
            checked=env.problem.config.validate(inst,candidate)
            assert checked.feasible and checked.objective>=env.initial_objective
            gain=(checked.objective-env.initial_objective)/env.scale
            outcomes.append(dict(action=index,jobs=list(a.jobs),operator=a.operator,status=status,
                gain=gain,objective=checked.objective,runtime=perf_counter()-tick,schedule=candidate.int().tolist()))
        best=max(outcomes,key=lambda r:r['gain'])
        mixed_positive=sorted([r for r in outcomes if r['gain']>1e-9],key=lambda r:r['gain'],reverse=True)[:3]
        for outcome in mixed_positive:
            for name in ('resched','mistar','dralns'):
                examples[name].append((obs,mask,outcome['action'],outcome['gain']/best['gain']))
        mip=[r for r in outcomes if r['operator']=='milp' and r['gain']>1e-9]
        if mip:
            norm=max(r['gain'] for r in mip)
            for outcome in sorted(mip,key=lambda r:r['gain'],reverse=True)[:3]:
                bits=torch.zeros(inst.J,dtype=torch.long);bits[outcome['jobs']]=1
                examples['rllns'].append((graph,subset_mask,bits.tolist(),outcome['gain']/norm))
        records.append(dict(instance=i,actions=[a.__dict__ for a in env.actions],outcomes=outcomes))
    return examples,records


def evaluate(method,model,instance,initial,seed,budget,repair_budget):
    if method=='profile_heuristic':
        row=evaluate_companion(instance,initial,method,None,budget=budget,seed=seed)
        row['baseline_history']=row.pop('transitions')
        row.update(transitions=[],feasible_decreases=0,status='companion_profile_portfolio')
        return row
    tick=perf_counter();torch.manual_seed(seed)
    env=LiteratureSearchEnv(instance,initial,budget=budget,seed=seed,repair_budget=repair_budget)
    env.deadline=min(env.deadline,tick+budget-env.reserve)
    status=None
    if method=='full_mip':
        if env.remaining()>0:
            result=env.problem.solve(time_limit=env.remaining(),incumbent=initial,solver_threads=1)
            checked=env.problem.config.validate(instance,result.schedule)
            status=result.status
            if perf_counter()<=env.deadline and checked.feasible:
                env.best=result.schedule;env.best_objective=checked.objective
    elif method in ('alns','uniform_anneal'):
        controller=AdaptiveLNS()
        while not env.done:
            mask=env.action_mask()
            if not mask.any():break
            a,key=controller.select(env,mask,adaptive=method=='alns')
            if env.done:break
            env.step_actions(a)
            if method=='alns':controller.update(key,env.log[-1])
    elif method=='random_mistar4':
        while not env.done:
            indices=torch.where(env.action_mask())[0].numpy()
            if not len(indices):break
            proposals=env.selection_rng.choice(indices,4,replace=True)
            if env.done:break
            env.step_actions(proposals)
    elif method=='random_rllns':
        while not env.subset_action_mask()[0]:
            bits=NonemptySubset(torch.zeros(instance.J)).sample()
            if env.done:break
            env.step_subset(bits)
    else:literature_rollout(env,model,subset=method.startswith('rllns'),collect=False)
    report=env.problem.config.validate(instance,env.best)
    runtime=perf_counter()-tick
    assert report.feasible and report.objective>=env.initial_objective
    return dict(initial=env.initial_objective,objective=report.objective,improvement=(report.objective-env.initial_objective)/env.scale,
        runtime=runtime,overrun=max(0.,runtime-budget),feasible=True,status=status,
        feasible_decreases=sum(r['accepted'] and r['current_objective']<r['before_current']-1e-7 for r in env.log),
        schedule=env.best.int().tolist(),transitions=env.log)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=32);parser.add_argument('--budget',type=float,default=1.2)
    args=parser.parse_args()
    if args.episodes<8 or args.episodes%4:raise ValueError('episodes must be a multiple of four, >=8')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    data={};manifest={};seeds=[11,29];regimes=('balanced','energy','periodic');data_seed=2026109200
    for split,count,large,offset in [('train',9,False,0),('development',3,False,100),('ood',6,True,200)]:
        data[split]=[witness_instance(seed=data_seed+offset+i,regime=regimes[i%3],large=large) for i in range(count)]
        manifest[split]=[dict(index=i,hash=instance_hash(inst),J=inst.J,T=inst.T) for i,(inst,_) in enumerate(data[split])]
    allhash=[r['hash'] for entries in manifest.values() for r in entries]
    assert len(allhash)==len(set(allhash))
    for old in output.parent.glob('rl-*/manifest.json'):
        assert not any(h in old.read_text() for h in allhash)
    write(output/'manifest.json',manifest)
    torch.save({split:[(inst.__dict__,x) for inst,x in entries] for split,entries in data.items()},output/'instances.pt')
    hashes=source_hashes();root=Path(__file__).resolve().parents[1]
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in hashes:archive.add(root/path,arcname=path)
    plan=dict(seeds=seeds,episodes=args.episodes,methods=METHODS,budget=args.budget,train_budget=.35,
        max_train_steps=4,repair_budget=.05,hidden=24,teacher_candidates_per_operator=6,teacher_epochs=8,
        continuation='PPO only, no extra teacher replay',ppo_epochs=3,ppo_minibatch=16,
        gamma=1.,trace_lambda=1.,entropy_coef=.001,data_seed=data_seed,source_hashes=hashes,
        source_archive_sha256=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest(),
        status='exploratory implementation validation; final checkpoints frozen before evaluation')
    write(output/'plan.json',plan)
    write(output/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=1,processor=platform.processor()))
    models={}
    for seed in seeds:
        tick=perf_counter();examples,teacher_records=teacher(data['train'],seed,.05)
        write(output/f'teacher-{seed}.json',teacher_records)
        torch.save(examples,output/f'teacher-examples-{seed}.pt')
        teacher_runtime=perf_counter()-tick
        for name,cls in MODELS.items():
            torch.manual_seed(seed);model=cls(hidden=24)
            torch.nn.init.zeros_(model.critic[-1].weight);torch.nn.init.zeros_(model.critic[-1].bias)
            il=imitate(model,torch.optim.Adam(model.parameters(),lr=.001),examples[name],subset=name=='rllns',seed=seed)
            model.save(output/f'{name}-il-{seed}.pt',metadata=dict(seed=seed,episodes=0,labels=len(examples[name])))
            models[(name+'_il',seed)]=copy.deepcopy(model).eval()
            opt=torch.optim.Adam(model.parameters(),lr=.0003);records=[];batch=[];rng=np.random.default_rng(seed+819)
            tick=perf_counter()
            for episode in range(args.episodes):
                i=int(rng.integers(len(data['train'])));inst,initial=data['train'][i]
                torch.manual_seed(seed*10000+episode)
                env=LiteratureSearchEnv(inst,initial,budget=.35,seed=seed*10000+episode,repair_budget=.05)
                trajectory=literature_rollout(env,model,subset=name=='rllns',max_steps=4)
                batch.append(trajectory)
                records.append(dict(episode=episode,instance=i,objective=env.best_objective,schedule=env.best.int().tolist(),transitions=env.log))
                if len(batch)==4:
                    metrics=ppo_batch_update(model,opt,batch,epochs=3,minibatch_size=16,trace_lambda=1.,entropy_coef=.001,seed=seed+episode)
                    records.append(dict(update=episode,metrics=metrics));batch=[]
            online_runtime=perf_counter()-tick
            model.save(output/f'{name}-ppo-{seed}.pt',metadata=dict(seed=seed,episodes=args.episodes,labels=len(examples[name])))
            models[(name+'_ppo',seed)]=model.eval()
            write(output/f'training-{name}-{seed}.json',dict(teacher_runtime=teacher_runtime,imitation=il,
                online_runtime=online_runtime,parameters=sum(p.numel() for p in model.parameters()),records=records))
            print(f'trained {name} seed {seed}: labels={len(examples[name])}, online={online_runtime:.2f}s',flush=True)
    write(output/'checkpoint-hashes.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.pt') if p.name not in ('instances.pt',) and not p.name.startswith('teacher-examples')})
    rows=[]
    with (output/'evaluations.jsonl').open('w') as stream:
        for split in ('development','ood'):
            for i,(inst,initial) in enumerate(data[split]):
                for seed in seeds:
                    for method in np.random.default_rng(822+seed+i+(10000 if split=='ood' else 0)).permutation(METHODS):
                        row=evaluate(str(method),models.get((str(method),seed)),inst,initial,seed,args.budget,.05)
                        row.update(split=split,instance=i,seed=seed,method=str(method))
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(f'evaluated {split} {i+1}/{len(data[split])}',flush=True)
    summary=summarize(rows);write(output/'summary.json',summary)
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Executable literature-inspired ONTS adaptations at '+str(args.budget)+r' s. Three development and six larger instances, two policy seeds. These are preliminary development comparisons, not original-paper reproductions.}',r'\label{tab:literature}',r'\begin{tabular}{lrr}',r'\toprule',r'Method & Development & Larger instances \\',r'\midrule']
    for method in METHODS:
        values=[next(r['improvement'] for r in summary['means'] if r['split']==s and r['method']==method) for s in ('development','ood')]
        tex.append(f"{method.replace('_',' ')} & {values[0]:.6f} & {values[1]:.6f}"+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'summary.tex').write_text('\n'.join(tex)+'\n')
    current=source_hashes();unchanged=all(current.get(p)==h for p,h in hashes.items())
    write(output/'source-integrity.json',dict(unchanged=unchanged));assert unchanged
    print(json.dumps(summary['means'],indent=2),flush=True)


if __name__=='__main__':main()
