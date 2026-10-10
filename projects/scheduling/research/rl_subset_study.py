"""Frozen development experiment: compact subset policies and verified hindsight."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import tarfile
from time import perf_counter

import numpy as np
import torch

from ..job_profile import reprofile, select_jobs
from ..rl_bipartite import BipartiteActorCritic, bipartite_observation
from ..rl_learning import ppo_batch_update
from ..rl_subset import SubsetActorCritic, SubsetEnv, subset_imitation, subset_rollout
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_bipartite_study import evaluate as evaluate_graph
from .rl_portfolio_study import evaluate as evaluate_template, write_json
from .statistics import bootstrap_mean

METHODS = ('profile_heuristic', 'previous_graph', 'subset_random', 'one_step_il_greedy',
           'hindsight_il_sample', 'hindsight_il_greedy', 'replay_sample', 'replay_greedy',
           'ppo_sample', 'ppo_greedy', 'graph_off_ppo_greedy')


def make_data(seed):
    regimes = ('balanced', 'energy', 'periodic')
    return {split: [witness_instance(seed=seed+offset+i, regime=regimes[i % 3], large=large)
                   for i in range(count)] for split, offset, count, large in
            [('train', 0, 12, False), ('development', 20000, 8, False), ('ood', 30000, 6, True)]}


def collect(data, *, seed):
    started, examples, one_step, records, pools = perf_counter(), [], [], [], []
    rng = np.random.default_rng(seed)
    for index, (instance, witness) in enumerate(data):
        pool_env = SubsetEnv(instance, witness, budget=3, seed=seed+index)
        stages = [witness.bool().clone()]
        for step in range(6):
            bits = torch.zeros(instance.J, dtype=torch.long)
            bits[rng.choice(instance.J, int(rng.integers(1, instance.J+1)), replace=False)] = 1
            pool_env.step_subset(bits)
            if step in (1, 5):
                stages.append(pool_env.incumbent.clone())
        pools.append(stages)
        for stage, initial in enumerate(stages):
            env = SubsetEnv(instance, initial, budget=.4, seed=seed+index*100+stage)
            observation, mask = bipartite_observation(env)
            groups = {tuple(select_jobs(env.problem, initial, rng, count, rule='pressure'))
                      for count in (1, 2, 3, instance.J)}
            for _ in range(16):
                groups.add(tuple(sorted(rng.choice(instance.J, int(rng.integers(1, instance.J+1)), replace=False).tolist())))
            outcomes = []

            def repair_label(jobs, origin):
                tick = perf_counter()
                candidate, _, _ = reprofile(env.problem, initial, jobs, rng, deadline=perf_counter()+.04)
                report = env.problem.config.validate(instance, candidate)
                if not report.feasible:
                    raise RuntimeError('Invalid teacher repair')
                duration = perf_counter()-tick
                gain = max(0., report.objective-env.objective)/env.scale
                bits = [int(j in jobs) for j in range(instance.J)]
                item = dict(origin=origin, jobs=list(jobs), bits=bits, gain=gain,
                            duration=duration, utility=gain/max(duration,.002),
                            schedule=candidate.int().tolist())
                outcomes.append(item)
                return item

            for jobs in sorted(groups):
                repair_label(jobs, 'one_step')
            # Future changed tasks are hypotheses until a fresh repair succeeds.
            current = initial.clone()
            trajectory = []
            for _ in range(4):
                count = int(rng.integers(1, instance.J+1))
                jobs = tuple(sorted(rng.choice(instance.J, count, replace=False).tolist()))
                tick = perf_counter()
                current, _, _ = reprofile(env.problem, current, jobs, rng, deadline=perf_counter()+.04)
                checked = env.problem.config.validate(instance, current)
                if not checked.feasible:
                    raise RuntimeError('Invalid hindsight trajectory')
                trajectory.append(dict(jobs=list(jobs), objective=checked.objective,
                    duration=perf_counter()-tick, schedule=current.int().tolist()))
            changed = torch.where((current != initial).any(1))[0].tolist()
            hindsight = repair_label(tuple(changed), 'hindsight') if changed else None
            positive = sorted([r for r in outcomes if r['origin']=='one_step' and r['gain']>1e-9],
                              key=lambda r: r['utility'], reverse=True)[:4]
            scale = max([r['utility'] for r in outcomes] or [1.])
            shared = [(observation, mask, r['bits'], r['utility']/max(scale,1e-9)) for r in positive]
            one_step.extend(shared)
            examples.extend(shared)
            if hindsight and hindsight['gain']>1e-9:
                examples.append((observation, mask, hindsight['bits'], hindsight['utility']/max(scale,1e-9)))
            records.append(dict(instance=index, stage=stage, initial=initial.int().tolist(),
                                initial_objective=env.objective, outcomes=outcomes,
                                hindsight_trajectory=trajectory))
        print(f'teacher seed={seed} instance={index+1}/{len(data)}', flush=True)
    return examples, one_step, pools, dict(runtime=perf_counter()-started, states=len(records),
                one_step_examples=len(one_step), total_examples=len(examples), records=records)


def continue_training(policy, examples, data, pools, *, seed, episodes, output, name, rl):
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    rng, pending, records, started = np.random.default_rng(seed), [], [], perf_counter()
    torch.manual_seed(seed)
    for episode in range(episodes):
        index = int(rng.integers(len(data)))
        stage = int(rng.choice(3, p=[.5,.3,.2]))
        if rl:
            env = SubsetEnv(data[index][0], pools[index][stage], budget=.4, seed=seed*10000+episode)
            pending.append(subset_rollout(env, policy))
        metrics, replay = None, None
        if (episode+1)%8 == 0 or episode == episodes-1:
            if rl:
                metrics = ppo_batch_update(policy, optimizer, pending, epochs=3, trace_lambda=1,
                                           entropy_coef=.001, minibatch_size=16, seed=seed+episode)
                pending = []
            selected = rng.choice(len(examples), min(8,len(examples)), replace=False)
            replay = subset_imitation(policy, optimizer, [examples[i] for i in selected], seed=seed+episode)
            print(f'{name} seed={seed} episodes={episode+1}/{episodes}', flush=True)
        records.append(dict(episode=episode, instance=index, stage=stage, ppo=metrics, replay=replay,
                            transitions=env.log if rl else []))
    write_json(output/f'training-{name}-{seed}.json', records)
    return dict(runtime=perf_counter()-started, episodes=episodes if rl else 0, replay_batches=(episodes+7)//8)


def evaluate(instance, witness, method, policy, *, seed, budget):
    if method=='profile_heuristic':
        return evaluate_template(instance,witness,method,None,budget=budget,seed=seed)
    if method=='previous_graph':
        result=evaluate_graph(instance,witness,'graph_greedy',policy,budget=budget,seed=seed)
        result['method']=method
        return result
    started=perf_counter()
    torch.manual_seed(seed)
    env=SubsetEnv(instance,witness,budget=budget,seed=seed)
    env.deadline=min(env.deadline,started+budget-env.reserve)
    subset_rollout(env,policy,greedy=method.endswith('greedy'))
    report=env.problem.config.validate(instance,env.incumbent)
    runtime=perf_counter()-started
    gain=(report.objective-env.initial_objective)/env.scale
    if not report.feasible or not np.isclose(gain,sum(t['reward'] for t in env.log)):
        raise RuntimeError('Invalid final outcome')
    return dict(method=method,feasible=True,initial=env.initial_objective,objective=report.objective,
                improvement=gain,runtime=runtime,overrun=max(0.,runtime-budget),steps=len(env.log),
                selection_runtime=sum(t['selection_runtime'] for t in env.log),
                mean_group_size=float(np.mean([len(t['jobs']) for t in env.log])) if env.log else 0.,
                schedule=env.incumbent.int().tolist(),transitions=env.log)


def summarize(output, rows):
    means, paired=[],[]
    for split in ('development','ood'):
        for method in METHODS:
            subset=[r for r in rows if r['split']==split and r['method']==method]
            means.append(dict(split=split,method=method,improvement=float(np.mean([r['improvement'] for r in subset])),
                selection_runtime=float(np.mean([r.get('selection_runtime',0) for r in subset])),
                steps=float(np.mean([r['steps'] for r in subset])),
                group_size=float(np.mean([r.get('mean_group_size',0) for r in subset]))))
        for a,b in [('ppo_sample','replay_sample'),('ppo_greedy','replay_greedy'),
                    ('ppo_sample','hindsight_il_sample'),('ppo_greedy','hindsight_il_greedy'),
                    ('hindsight_il_greedy','one_step_il_greedy'),('ppo_greedy','subset_random'),
                    ('ppo_greedy','previous_graph'),('ppo_greedy','profile_heuristic'),
                    ('ppo_greedy','graph_off_ppo_greedy')]:
            delta=[]
            for index in sorted({r['instance'] for r in rows if r['split']==split}):
                values=[[r['improvement'] for r in rows if r['split']==split and r['instance']==index and r['method']==m] for m in (a,b)]
                delta.append(float(np.mean(values[0])-np.mean(values[1])))
            paired.append(dict(split=split,method=a,baseline=b,delta=bootstrap_mean(delta,seed=901,resamples=4000)))
    write_json(output/'summary.json',dict(means=means,paired=paired))
    lines=['# Compact subset policy development experiment','',
        'All scores are normalized witness improvement. Two seeds per instance; eight development and six larger cases.',
        'Final checkpoints and both inference modes fixed before evaluation. Marginal exploratory intervals, no multiplicity adjustment.','',
        '| Split | Method | Improvement | Selection seconds | Steps | Group size |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in means:
        lines.append(f"| {r['split']} | {r['method']} | {r['improvement']:.6f} | {r['selection_runtime']:.4f} | {r['steps']:.1f} | {r['group_size']:.2f} |")
    lines+=['','| Split | Comparison | Difference | Marginal 95% CI |','| --- | --- | --- | --- |']
    for r in paired:
        lo,hi=r['delta']['ci95']
        lines.append(f"| {r['split']} | {r['method']} − {r['baseline']} | {r['delta']['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    tex=[r'\begin{table}[t]',r'\centering',r'\caption{Compact subset study: mean normalized improvement over common witnesses at 1.2 s. Eight development and six larger instances, two seeds per instance. All inference modes are predeclared; comparisons are exploratory.}',
        r'\label{tab:subset}',r'\begin{tabular}{lrr}',r'\toprule',r'Method & Development & Larger instances \\',r'\midrule']
    for method in METHODS:
        values=[next(r['improvement'] for r in means if r['method']==method and r['split']==s) for s in ('development','ood')]
        tex.append(f"{method.replace('_',' ')} & {values[0]:.6f} & {values[1]:.6f}"+r' \\')
    tex+=[r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'summary.tex').write_text('\n'.join(tex)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--episodes',type=int,default=64)
    args=parser.parse_args()
    output=args.output
    output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    seeds=(11,29)
    data=make_data(2026106100)
    manifest={s:[dict(index=i,hash=instance_hash(inst),J=inst.J,T=inst.T) for i,(inst,_) in enumerate(entries)] for s,entries in data.items()}
    hashes=[r['hash'] for entries in manifest.values() for r in entries]
    prior=[]
    for path in output.parent.glob('rl-*/manifest.json'):
        if path.parent!=output:
            prior.extend(h for h in hashes if h in path.read_text())
    if len(set(hashes))!=len(hashes) or prior:
        raise RuntimeError('Instance content overlap')
    write_json(output/'manifest.json',manifest)
    torch.save({s:[(vars(inst),w) for inst,w in entries] for s,entries in data.items()},output/'instances.pt')
    sources=source_hashes()
    root=Path(__file__).resolve().parents[1]
    with tarfile.open(output/'source.tar.gz','w:gz') as archive:
        for path in sources:
            archive.add(root/path,arcname=path)
    previous=root/'research/results/rl-bipartite-20261006/continuation-diagnostic'
    plan=dict(methods=METHODS,seeds=seeds,data_seed=2026106100,episodes=args.episodes,
        train_budget=.4,eval_budget=1.2,teacher_repair_budget=.04,teacher_candidates=20,
        hindsight_steps=4,imitation_epochs=12,hidden=24,source_hashes=sources,
        source_archive_sha256=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest(),
        old_checkpoints={str(s):hashlib.sha256((previous/f'graph_replay-{s}.pt').read_bytes()).hexdigest() for s in seeds},
        python=platform.python_version(),torch=torch.__version__,cpu=platform.processor(),
        inference='sample and conditional MAP predeclared; no validation tuning or test selection',
        comparison='subset policies share profile repair; previous portfolio and heuristic have their own controllers',
        model_selection='final checkpoint, no early stopping or score-based selection')
    write_json(output/'plan.json',plan)
    policies,costs={},{}
    for seed in seeds:
        examples,one_step,pools,teacher=collect(data['train'],seed=seed)
        write_json(output/f'teachers-{seed}.json',teacher)
        torch.save(dict(examples=examples,one_step=one_step,pools=pools),output/f'teacher-examples-{seed}.pt')
        torch.manual_seed(seed)
        base=SubsetActorCritic()
        one=copy.deepcopy(base)
        subset_imitation(one,torch.optim.Adam(one.parameters(),lr=.001),one_step,epochs=12,seed=seed)
        start=perf_counter()
        subset_imitation(base,torch.optim.Adam(base.parameters(),lr=.001),examples,epochs=12,seed=seed)
        costs[str(seed)]=dict(teacher_runtime=teacher['runtime'],imitation_runtime=perf_counter()-start,
                             hindsight_labels=teacher['total_examples']-teacher['one_step_examples'])
        replay=copy.deepcopy(base)
        costs[str(seed)]['replay']=continue_training(replay,examples,data['train'],pools,seed=seed,
            episodes=args.episodes,output=output,name='replay',rl=False)
        ppo=copy.deepcopy(base)
        costs[str(seed)]['ppo']=continue_training(ppo,examples,data['train'],pools,seed=seed,
            episodes=args.episodes,output=output,name='ppo',rl=True)
        torch.manual_seed(seed)
        off=SubsetActorCritic(graph=False)
        subset_imitation(off,torch.optim.Adam(off.parameters(),lr=.001),examples,epochs=12,seed=seed)
        costs[str(seed)]['graph_off_ppo']=continue_training(off,examples,data['train'],pools,seed=seed,
            episodes=args.episodes,output=output,name='graph_off_ppo',rl=True)
        for name,model in [('one_step_il',one),('hindsight_il',base),('replay',replay),('ppo',ppo),('graph_off_ppo',off)]:
            model.save(output/f'{name}-{seed}.pt',metadata=dict(seed=seed,episodes=args.episodes,final=True))
        policies[seed]={'one_step_il_greedy':one,'hindsight_il_sample':base,'hindsight_il_greedy':base,
            'replay_sample':replay,'replay_greedy':replay,'ppo_sample':ppo,'ppo_greedy':ppo,
            'graph_off_ppo_greedy':off,'previous_graph':BipartiteActorCritic.load(previous/f'graph_replay-{seed}.pt')}
    write_json(output/'costs.json',costs)
    rows=[]
    with (output/'evaluations.jsonl').open('w') as stream:
        for split in ('development','ood'):
            for index,(instance,witness) in enumerate(data[split]):
                for seed in seeds:
                    for method in np.random.default_rng(718+index+seed+(10000 if split=='ood' else 0)).permutation(METHODS):
                        row=evaluate(instance,witness,str(method),policies[seed].get(method),seed=seed,budget=1.2)
                        row.update(split=split,instance=index,seed=seed)
                        rows.append(row)
                        stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                print(f'evaluation {split} {index+1}/{len(data[split])}',flush=True)
    summarize(output,rows)
    if source_hashes()!=sources:
        raise RuntimeError('Experiment sources changed during run')
    write_json(output/'source-integrity.json',dict(unchanged=True))


if __name__=='__main__':
    main()
