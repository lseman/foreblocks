"""Frozen experiment: correlated ordered task groups and alternative RL baselines."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import tarfile
from time import perf_counter

import numpy as np
import torch

from ..job_profile import reprofile, select_jobs
from ..rl_bipartite import bipartite_observation
from ..rl_sequence import (SequenceActorCritic, SequenceEnv, rloo_update,
    sequence_imitation, sequence_ppo_update, sequence_rollout)
from ..rl_subset import SubsetActorCritic, SubsetEnv, subset_imitation, subset_rollout
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_portfolio_study import evaluate as evaluate_template, write_json
from .statistics import bootstrap_mean

METHODS=('profile_heuristic','previous_factor_il','factor_il','ar_profile_il','random_profile',
         'random_mixed','ar_mixed_il','ar_mixed_replay','ar_mixed_ppo','ar_independent_ppo',
         'ar_mixed_rloo','ar_ppo_profile_only')


def data_splits():
    regimes=('balanced','energy','periodic')
    return {split:[witness_instance(seed=2026109100+offset+i,regime=regimes[i%3],large=large)
        for i in range(count)] for split,offset,count,large in
        [('train',0,12,False),('development',20000,8,False),('ood',30000,6,True)]}


def collect(data,*,seed):
    rng=np.random.default_rng(seed)
    mixed,profile,records,pools=[],[],[],[]
    tick=perf_counter()
    for index,(instance,witness) in enumerate(data):
        pool=SequenceEnv(instance,witness,budget=3,seed=seed+index,mixed=False)
        stages=[witness.bool().clone()]
        for step in range(6):
            count=int(rng.integers(1,instance.J+1))
            jobs=rng.choice(instance.J,count,replace=False).tolist()
            pool.step_sequence([0,count,*jobs,*([-1]*(instance.J-count))])
            if step in (1,5):stages.append(pool.incumbent.clone())
        pools.append(stages)
        for stage,initial in enumerate(stages):
            env=SequenceEnv(instance,initial,budget=.6,seed=seed+100*index+stage)
            observation,mask=bipartite_observation(env)
            groups={tuple(sorted(select_jobs(env.problem,initial,rng,k,rule='pressure')))
                    for k in (1,2,3,instance.J)}
            for _ in range(4):
                count=int(rng.integers(1,instance.J+1))
                groups.add(tuple(sorted(rng.choice(instance.J,count,replace=False).tolist())))
            outcomes=[]
            for jobs in sorted(groups):
                free=torch.zeros_like(initial,dtype=torch.bool);free[list(jobs)]=True
                for operator in range(3):
                    started=perf_counter()
                    if operator==0:
                        candidate,_,_=reprofile(env.problem,initial,jobs,rng,deadline=perf_counter()+.04)
                    else:
                        candidate=env.problem.solve(time_limit=(.025,.1)[operator-1],incumbent=initial,
                                                    free_mask=free,solver_threads=1).schedule
                    checked=env.problem.config.validate(instance,candidate)
                    if not checked.feasible:raise RuntimeError('Invalid teacher repair')
                    gain=max(0.,checked.objective-env.objective)/env.scale
                    outcomes.append(dict(operator=operator,jobs=list(jobs),gain=gain,duration=perf_counter()-started,
                                         schedule=candidate.int().tolist()))
            norm=max([r['gain'] for r in outcomes] or [1.])
            positive=sorted([r for r in outcomes if r['gain']>1e-9],key=lambda r:r['gain'],reverse=True)[:4]
            prof=sorted([r for r in outcomes if r['operator']==0 and r['gain']>1e-9],key=lambda r:r['gain'],reverse=True)[:4]
            for candidates,target,enabled in [(positive,mixed,mask),(prof,profile,torch.tensor([False,True,False,False]))]:
                target.extend((observation,enabled,r['operator'],r['jobs'],r['gain']/max(norm,1e-9)) for r in candidates)
            records.append(dict(instance=index,stage=stage,initial=initial.int().tolist(),
                                initial_objective=env.objective,outcomes=outcomes))
        print(f'teacher seed={seed} {index+1}/{len(data)}',flush=True)
    return mixed,profile,pools,dict(runtime=perf_counter()-tick,records=records,
                                    mixed_labels=len(mixed),profile_labels=len(profile))


def continuation(policy,examples,data,pools,*,seed,episodes,output,name,algorithm):
    optimizer=torch.optim.Adam(policy.parameters(),lr=.0003)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    pending,groups,records=[],[],[]
    tick=perf_counter()
    for block in range(episodes//4):
        index=int(rng.integers(len(data)))
        stage=int(rng.choice(3,p=[.5,.3,.2]))
        trajectories=[]
        for replica in range(4):
            episode=block*4+replica
            if algorithm!='replay':
                env=SequenceEnv(data[index][0],pools[index][stage],budget=.6,seed=seed*10000+episode)
                trajectories.append(sequence_rollout(env,policy))
                records.append(dict(episode=episode,instance=index,stage=stage,transitions=env.log,
                                    schedule=env.incumbent.int().tolist(),objective=env.objective))
        pending.extend(trajectories);groups.append(trajectories)
        if (block+1)%2==0:
            metrics=None
            if algorithm=='ppo':
                metrics=sequence_ppo_update(policy,optimizer,pending,epochs=3,minibatch_size=16,
                    trace_lambda=1,target_kl=.03,seed=seed+block)
            elif algorithm=='rloo':metrics=rloo_update(policy,optimizer,groups)
            selected=rng.choice(len(examples),min(8,len(examples)),replace=False)
            replay=sequence_imitation(policy,optimizer,[examples[i] for i in selected],seed=seed+block)
            records.append(dict(block=block,metrics=metrics,replay=replay,replay_indices=selected.tolist()))
            pending,groups=[],[]
            print(f'{name} seed={seed} {4*(block+1)}/{episodes}',flush=True)
    write_json(output/f'training-{name}-{seed}.json',records)
    return dict(runtime=perf_counter()-tick,online_episodes=episodes if algorithm!='replay' else 0,
                replay_updates=episodes//8)


def evaluate(instance,initial,method,policy,*,seed,budget):
    if method=='profile_heuristic':return evaluate_template(instance,initial,method,None,seed=seed,budget=budget)
    started=perf_counter();torch.manual_seed(seed)
    if method in ('previous_factor_il','factor_il'):
        env=SubsetEnv(instance,initial,budget=budget,seed=seed)
        env.deadline=min(env.deadline,started+budget-env.reserve)
        subset_rollout(env,policy)
    else:
        mixed=method not in ('ar_profile_il','random_profile','ar_ppo_profile_only')
        env=SequenceEnv(instance,initial,budget=budget,seed=seed,mixed=mixed)
        env.deadline=min(env.deadline,started+budget-env.reserve)
        sequence_rollout(env,policy)
    checked=env.problem.config.validate(instance,env.incumbent)
    runtime=perf_counter()-started
    improvement=(checked.objective-env.initial_objective)/env.scale
    if not checked.feasible or not np.isclose(improvement,sum(t['reward'] for t in env.log)):
        raise RuntimeError('Invalid evaluation accounting')
    return dict(method=method,feasible=True,initial=env.initial_objective,objective=checked.objective,
        returned_objective=checked.objective,improvement=improvement,runtime=runtime,overrun=max(0.,runtime-budget),
        steps=sum(t['executed'] for t in env.log),milp_steps=sum(t['executed'] and t['allowance']>0 for t in env.log),
        selection_runtime=sum(t['selection_runtime'] for t in env.log),
        schedule=env.incumbent.int().tolist(),transitions=env.log)


def summarize(output,rows):
    means,paired=[],[]
    for split in ('development','ood'):
        for method in METHODS:
            selected=[r for r in rows if r['split']==split and r['method']==method]
            means.append(dict(split=split,method=method,improvement=float(np.mean([r['improvement'] for r in selected])),
                selection_runtime=float(np.mean([r.get('selection_runtime',0) for r in selected])),
                milp_steps=float(np.mean([r.get('milp_steps',0) for r in selected]))))
        for a,b in [('ar_mixed_ppo','ar_mixed_il'),('ar_mixed_ppo','ar_mixed_replay'),
                    ('ar_mixed_rloo','ar_mixed_replay'),('ar_mixed_rloo','ar_mixed_ppo'),
                    ('ar_mixed_ppo','ar_independent_ppo'),('ar_profile_il','factor_il'),
                    ('ar_mixed_il','ar_profile_il'),('ar_mixed_ppo','ar_ppo_profile_only'),
                    ('ar_mixed_ppo','random_mixed'),('ar_mixed_ppo','previous_factor_il'),
                    ('ar_mixed_ppo','profile_heuristic'),('ar_mixed_rloo','random_mixed')]:
            delta=[]
            for index in sorted({r['instance'] for r in rows if r['split']==split}):
                values=[[r['improvement'] for r in rows if r['split']==split and r['instance']==index and r['method']==m] for m in (a,b)]
                delta.append(float(np.mean(values[0])-np.mean(values[1])))
            paired.append(dict(split=split,method=a,baseline=b,delta=bootstrap_mean(delta,seed=445,resamples=4000)))
    write_json(output/'summary.json',dict(means=means,paired=paired))
    lines=['# Ordered group and repair experiment','',
        'Normalized improvement over common witnesses at 2 seconds; two seeds averaged within instance.',
        'Eight development and six larger instances. Fixed final checkpoints; sampled deployment only.',
        'Marginal exploratory intervals, no multiplicity adjustment.','',
        '| Split | Method | Improvement | Selection seconds | MILP steps |','| --- | --- | --- | --- | --- |']
    for r in means:lines.append(f"| {r['split']} | {r['method']} | {r['improvement']:.6f} | {r['selection_runtime']:.4f} | {r['milp_steps']:.1f} |")
    lines+=['','| Split | Comparison | Difference | Marginal 95% interval |','| --- | --- | --- | --- |']
    for r in paired:
        lo,hi=r['delta']['ci95'];lines.append(f"| {r['split']} | {r['method']} − {r['baseline']} | {r['delta']['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Ordered-group study: normalized improvement over common witnesses at 2 s. Eight development and six larger instances; two seeds per instance; final checkpoints and stochastic deployment fixed before evaluation.}',
         r'\label{tab:sequence}',r'\begin{tabular}{lrr}',r'\toprule',r'Method & Development & Larger instances \\',r'\midrule']
    for method in METHODS:
        values=[next(r['improvement'] for r in means if r['split']==s and r['method']==method) for s in ('development','ood')]
        tex.append(f"{method.replace('_',' ')} & {values[0]:.6f} & {values[1]:.6f}"+r' \\')
    tex+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'summary.tex').write_text('\n'.join(tex)+'\n');print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=64);args=parser.parse_args()
    if args.episodes<8 or args.episodes%8:raise ValueError('Episodes must be a positive multiple of eight')
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);data=data_splits();seeds=(11,29)
    manifest={s:[dict(index=i,hash=instance_hash(inst),J=inst.J,T=inst.T) for i,(inst,_) in enumerate(entries)] for s,entries in data.items()}
    hashes=[r['hash'] for entries in manifest.values() for r in entries]
    assert len(set(hashes))==len(hashes)
    for path in output.parent.glob('rl-*/manifest.json'):
        if path.parent!=output:assert not any(h in path.read_text() for h in hashes)
    write_json(output/'manifest.json',manifest)
    torch.save({s:[(vars(inst),x) for inst,x in entries] for s,entries in data.items()},output/'instances.pt')
    root=Path(__file__).resolve().parents[1];sources=source_hashes()
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in sources:archive.add(root/path,arcname=path)
    previous=root/'research/results/rl-subset-20261006'
    write_json(output/'plan.json',dict(methods=METHODS,seeds=seeds,episodes=args.episodes,train_budget=.6,eval_budget=2.,
        data_seed=2026109100,hidden=24,teacher_states=36,teacher_groups_max=8,teacher_profile_allowance=.04,
        milp_allowances=[.025,.1],teacher_objective='normalized gain, not improvement rate',
        imitation_epochs=12,training_groups='two groups of four same-instance/same-incumbent rollouts per update',
        entropy_regularization=0,ppo_epochs=3,ppo_minibatch=16,ppo_target_kl=.03,gamma=1,lambda_gae=1,
        continuation_lr=.0003,il_lr=.001,inference='sampled only; final checkpoints; no tuning or best-of-N',
        comparisons='predeclared in archived summarize function; exploratory marginal intervals',
        source_hashes=sources,source_archive_sha256=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest(),
        previous_checkpoints={str(s):hashlib.sha256((previous/f'hindsight_il-{s}.pt').read_bytes()).hexdigest() for s in seeds}))
    policies,costs={},{}
    for seed in seeds:
        mixed,profile,pools,teacher=collect(data['train'],seed=seed)
        write_json(output/f'teachers-{seed}.json',teacher)
        torch.save(dict(mixed=mixed,profile=profile,pools=pools),output/f'teacher-examples-{seed}.pt')
        torch.manual_seed(seed);initial=SequenceActorCritic()
        il=copy.deepcopy(initial);prof=copy.deepcopy(initial)
        sequence_imitation(il,torch.optim.Adam(il.parameters(),lr=.001),mixed,epochs=12,seed=seed)
        sequence_imitation(prof,torch.optim.Adam(prof.parameters(),lr=.001),profile,epochs=12,seed=seed)
        factor_examples=[]
        for obs,mask,op,jobs,weight in profile:
            bits=[int(j in jobs) for j in range(len(obs.tasks))];factor_examples.append((obs,mask,bits,weight))
        torch.manual_seed(seed);factor=SubsetActorCritic()
        subset_imitation(factor,torch.optim.Adam(factor.parameters(),lr=.001),factor_examples,epochs=12,seed=seed)
        replay=copy.deepcopy(il);ppo=copy.deepcopy(il);rloo=copy.deepcopy(il)
        torch.manual_seed(seed);independent=SequenceActorCritic(correlated=False)
        sequence_imitation(independent,torch.optim.Adam(independent.parameters(),lr=.001),mixed,epochs=12,seed=seed)
        costs[str(seed)]=dict(teacher_runtime=teacher['runtime'])
        for name,model,algorithm in [('replay',replay,'replay'),('ppo',ppo,'ppo'),('rloo',rloo,'rloo'),('independent_ppo',independent,'ppo')]:
            costs[str(seed)][name]=continuation(model,mixed,data['train'],pools,seed=seed,episodes=args.episodes,
                output=output,name=name,algorithm=algorithm)
        for name,model in [('mixed_il',il),('profile_il',prof),('factor_il',factor),('replay',replay),('ppo',ppo),('rloo',rloo),('independent_ppo',independent)]:
            model.save(output/f'{name}-{seed}.pt',metadata=dict(seed=seed,method=name,source_hashes=sources,
                online_episodes=args.episodes if name in ('ppo','rloo','independent_ppo') else 0))
        policies[seed]=dict(previous_factor_il=SubsetActorCritic.load(previous/f'hindsight_il-{seed}.pt'),factor_il=factor,
            ar_profile_il=prof,ar_mixed_il=il,ar_mixed_replay=replay,ar_mixed_ppo=ppo,
            ar_independent_ppo=independent,ar_mixed_rloo=rloo,ar_ppo_profile_only=ppo)
    write_json(output/'costs.json',costs)
    rows=[]
    with (output/'evaluations.jsonl').open('w') as stream:
        for split in ('development','ood'):
            for index,(instance,initial) in enumerate(data[split]):
                for seed in seeds:
                    for method in np.random.default_rng(1933+seed+index+(10000 if split=='ood' else 0)).permutation(METHODS):
                        row=evaluate(instance,initial,str(method),policies[seed].get(method),seed=seed,budget=2.)
                        row.update(split=split,instance=index,seed=seed);rows.append(row)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(f'evaluation {split} {index+1}/{len(data[split])}',flush=True)
    summarize(output,rows)
    assert source_hashes()==sources
    write_json(output/'source-integrity.json',dict(unchanged=True))


if __name__=='__main__':main()
