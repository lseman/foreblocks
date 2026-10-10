"""Matched development study of concrete task groups and task--time attention."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import tarfile
from time import perf_counter

import numpy as np
import torch

from ..job_profile import reprofile
from ..rl_bipartite import (BipartiteActorCritic, TaskNeighborhoodEnv,
                            bipartite_observation, listwise_update)
from ..rl_learning import ppo_batch_update
from ..rl_portfolio import PortfolioActorCritic, PortfolioEnv, portfolio_rollout
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_portfolio_study import evaluate as evaluate_template, write_json
from .statistics import bootstrap_mean

METHODS = ('template_bandit', 'template_checkpoint', 'direct_uniform', 'direct_bandit',
           'flat_hybrid', 'graph_off_hybrid', 'graph_il', 'graph_hybrid',
           'graph_greedy', 'profile_heuristic')


def splits(seed, *, train_count=24, development_count=12, ood_count=6):
    regimes = ('balanced', 'energy', 'periodic')
    data = {split: [witness_instance(seed=seed+offset+i, regime=regimes[i % 3], large=large)
                   for i in range(count)]
            for split, offset, count, large in [('train', 0, train_count, False),
                ('validation', 10000, 3, False), ('development', 20000, development_count, False),
                ('ood', 30000, ood_count, True)]}
    hashes = [instance_hash(i) for entries in data.values() for i, _ in entries]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('Generated instance overlap')
    return data


def training_pools(data, seed):
    started, pools = perf_counter(), []
    for index, (instance, witness) in enumerate(data):
        env = TaskNeighborhoodEnv(instance, witness, budget=5, seed=seed+index)
        stages = [witness.bool().clone()]
        for step in range(12):
            available = torch.where(env.action_mask())[0].tolist()
            profiles = [a for a in available if a and env.actions[a-1].family >= 4]
            if profiles:
                env.step(int(env.selection_rng.choice(profiles)))
            if step in (1, 5, 11):
                stages.append(env.incumbent.clone())
        pools.append(stages)
    return pools, perf_counter()-started


def collect_teachers(data, pools, *, seed, budget):
    examples, records = [], []
    started = perf_counter()
    for index, (instance, _) in enumerate(data):
        for stage, incumbent in enumerate(pools[index]):
            env = TaskNeighborhoodEnv(instance, incumbent, budget=budget, seed=seed+100*index+stage)
            observation, mask = bipartite_observation(env)
            utility, outcomes = [0.], []
            for action_index, action in enumerate(env.actions, 1):
                tick = perf_counter()
                if action.family < 4:
                    result = env.problem.solve(time_limit=action.allowance, incumbent=env.incumbent,
                                              free_mask=action.mask, solver_threads=1)
                    candidate, status = result.schedule, result.status
                else:
                    candidate, _, _ = reprofile(env.problem, env.incumbent, action.jobs,
                        np.random.default_rng(seed+10000*index+1000*stage+action_index),
                        deadline=perf_counter()+.1)
                    status = 'profile'
                checked = env.problem.config.validate(instance, candidate)
                if not checked.feasible:
                    raise RuntimeError('Infeasible teacher candidate')
                duration = perf_counter()-tick
                gain = max(0., checked.objective-env.objective)/env.scale
                utility.append(gain/max(duration, .002))
                outcomes.append(dict(jobs=list(action.jobs), family=action.family,
                                     gain=gain, duration=duration, status=status,
                                     allowance=action.allowance))
            examples.append((observation, mask, utility))
            records.append(dict(instance=index, stage=stage, initial_objective=env.objective,
                                utilities=utility, outcomes=outcomes))
        print(f'teachers seed={seed} instance={index+1}/{len(data)}', flush=True)
    return examples, dict(runtime=perf_counter()-started, states=len(examples), records=records)


def train_hybrid(policy, examples, data, pools, *, seed, episodes, budget, output, name):
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
    rng = np.random.default_rng(seed)
    pending, started = [], perf_counter()
    torch.manual_seed(seed)
    with (output/f'training-{name}-{seed}.jsonl').open('w') as stream:
        for episode in range(episodes):
            index = int(rng.integers(len(data)))
            stage = int(rng.choice(4, p=[.4, .25, .2, .15]))
            env = TaskNeighborhoodEnv(data[index][0], pools[index][stage], budget=budget,
                                       seed=seed*10000+episode)
            pending.append(portfolio_rollout(env, policy))
            metrics, replay_metrics = None, None
            if len(pending) == 8 or episode == episodes-1:
                metrics = ppo_batch_update(policy, optimizer, pending, epochs=3, trace_lambda=1,
                                          entropy_coef=.001, minibatch_size=16, seed=seed+episode)
                selected = rng.choice(len(examples), min(8, len(examples)), replace=False)
                replay_metrics = listwise_update(policy, optimizer, [examples[i] for i in selected],
                                                  epochs=1, seed=seed+episode)
                pending = []
                print(f'seed={seed} method={name} episodes={episode+1}/{episodes}', flush=True)
            stream.write(json.dumps(dict(episode=episode, instance=index, stage=stage,
                                transitions=env.log, ppo=metrics, teacher_replay=replay_metrics),
                                allow_nan=False)+'\n')
            stream.flush()
    return dict(seed=seed, method=name, runtime=perf_counter()-started)


def evaluate(instance, initial, method, policy, *, budget, seed):
    if method in ('profile_heuristic', 'template_bandit', 'template_checkpoint'):
        baseline = {'template_bandit': 'bandit', 'template_checkpoint': 'preference_ppo'}.get(method, method)
        row = evaluate_template(instance, initial, baseline, policy, budget=budget, seed=seed)
        row['method'] = method
        return row
    tick = perf_counter()
    torch.manual_seed(seed)
    env = TaskNeighborhoodEnv(instance, initial, budget=budget, seed=seed)
    env.deadline = min(env.deadline, tick+budget-env.reserve)
    mode = {'direct_uniform': 'uniform', 'direct_bandit': 'bandit'}.get(method, 'ppo')
    portfolio_rollout(env, policy, mode=mode, greedy=method == 'graph_greedy')
    checked = env.problem.config.validate(instance, env.incumbent)
    runtime = perf_counter()-tick
    if not checked.feasible or not np.isclose(sum(t['reward'] for t in env.log),
                                    (checked.objective-env.initial_objective)/env.scale):
        raise RuntimeError('Failed independent outcome accounting')
    return dict(method=method, feasible=True, initial=env.initial_objective,
                objective=env.objective, returned_objective=checked.objective,
                improvement=(env.objective-env.initial_objective)/env.scale,
                runtime=runtime, overrun=max(0., runtime-budget),
                steps=sum(t['executed'] for t in env.log),
                profile_steps=sum(t['executed'] and t['family'].startswith('profile') for t in env.log),
                milp_steps=sum(t['executed'] and t['family'].startswith('milp') for t in env.log),
                selection_runtime=sum(t['selection_runtime'] for t in env.log),
                schedule=env.incumbent.int().tolist(), transitions=env.log)


def summarize(output, rows, *, budget=1.2):
    means, paired = [], []
    lines = ['# Task--time bipartite development experiment', '',
             'Mean normalized improvement over common feasible witnesses; not optimality gaps.',
             'Policy/search seeds averaged within instance. Intervals are exploratory and marginal.', '',
             '| Split | Method | Mean improvement | Profile steps | MILP steps | Selection seconds |',
             '| --- | --- | --- | --- | --- | --- |']
    tex = [r'\begin{table}[t]', r'\centering',
           rf'\caption{{Task--time study: mean normalized incumbent improvement at {budget:g} s.',
           r'Expanded controllers share every singleton/pair, three triple proposals and eight MILP actions.',
           r'Template controls retain the previous 17 proposals. Means average seeds within instance.',
           r'These are development results, not comparisons to published ONTS implementations.}',
           r'\label{tab:bipartite}', r'\begin{tabular}{lrr}', r'\toprule',
           r'Method & Development & Larger instances \\', r'\midrule']
    for split in ('validation', 'development', 'ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue
            ids = sorted({r['instance'] for r in selected})
            values = [np.mean([r['improvement'] for r in selected if r['instance'] == i]) for i in ids]
            mean = float(np.mean(values))
            means.append(dict(split=split, method=method, improvement=mean, instances=len(ids)))
            lines.append(f"| {split} | {method} | {mean:.6f} | "
                         f"{np.mean([r['profile_steps'] for r in selected]):.1f} | "
                         f"{np.mean([r['milp_steps'] for r in selected]):.1f} | "
                         f"{np.mean([r['selection_runtime'] for r in selected]):.4f} |")
    for method in METHODS:
        values = [next(r['improvement'] for r in means if r['split'] == s and r['method'] == method)
                  for s in ('development', 'ood')]
        tex.append(f"{method.replace('_', ' ')} & {values[0]:.6f} & {values[1]:.6f}" + r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    pairs = [('graph_hybrid', 'graph_il'), ('graph_hybrid', 'graph_off_hybrid'),
             ('graph_hybrid', 'flat_hybrid'), ('graph_hybrid', 'direct_bandit'),
             ('graph_hybrid', 'profile_heuristic'), ('graph_greedy', 'graph_hybrid'),
             ('graph_greedy', 'template_checkpoint'), ('graph_greedy', 'profile_heuristic'),
             ('direct_bandit', 'template_bandit')]
    lines += ['', '| Split | Paired comparison | Difference | Marginal 95% interval |',
              '| --- | --- | --- | --- |']
    for split in ('development', 'ood'):
        for method, baseline in pairs:
            deltas = []
            for i in sorted({r['instance'] for r in rows if r['split'] == split}):
                a, b = [[r['improvement'] for r in rows if r['split'] == split
                         and r['instance'] == i and r['method'] == m] for m in (method, baseline)]
                deltas.append(float(np.mean(a)-np.mean(b)))
            result = bootstrap_mean(deltas, resamples=4000)
            paired.append(dict(split=split, method=method, baseline=baseline, delta=result))
            lo, hi = result['ci95']
            lines.append(f"| {split} | {method} − {baseline} | {result['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    write_json(output/'summary.json', dict(means=means, comparisons=paired))
    (output/'summary.tex').write_text('\n'.join(tex)+'\n')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:30]), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--episodes', type=int, default=128)
    parser.add_argument('--train-budget', type=float, default=.6)
    parser.add_argument('--eval-budget', type=float, default=1.2)
    parser.add_argument('--data-seed', type=int, default=2026103100)
    parser.add_argument('--policy-seeds', type=int, nargs='+', default=[11, 29])
    parser.add_argument('--train-count', type=int, default=24)
    parser.add_argument('--development-count', type=int, default=12)
    parser.add_argument('--ood-count', type=int, default=6)
    parser.add_argument('--teacher-epochs', type=int, default=12)
    parser.add_argument('--template-results', type=Path,
                        default=Path(__file__).parent/'results/rl-portfolio-20261006')
    args = parser.parse_args()
    if min(args.episodes, args.train_count, args.development_count, args.ood_count, args.teacher_epochs) < 1:
        parser.error('Counts must be positive')
    if not args.policy_seeds or len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('Policy seeds must be nonempty and unique')
    if not all(np.isfinite(v) and v > .15 for v in (args.train_budget, args.eval_budget)):
        parser.error('Budgets must be finite and greater than .15 seconds')
    # Check all baseline checkpoints before creating an immutable result directory.
    baseline_paths = [args.template_results/f'preference_ppo-{s}.pt' for s in args.policy_seeds]
    if not all(p.exists() for p in baseline_paths):
        parser.error('Missing template baseline checkpoint')
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['ONTS_SOLVER_THREADS'] = '1'
    torch.set_num_threads(1)
    data = splits(args.data_seed, train_count=args.train_count,
                  development_count=args.development_count, ood_count=args.ood_count)
    hashes = {s: [instance_hash(i) for i, _ in entries] for s, entries in data.items()}
    current = set(sum(hashes.values(), []))
    for previous in args.output.parent.glob('rl-*/manifest.json'):
        contents = json.loads(previous.read_text())
        prior = {entry['hash'] if isinstance(entry, dict) else entry
                 for entries in contents.values() for entry in entries}
        if current & prior:
            raise RuntimeError(f'Overlap with examined instances: {previous}')
    write_json(args.output/'manifest.json', hashes)
    torch.save({s: [(vars(i), x) for i, x in entries] for s, entries in data.items()}, args.output/'instances.pt')
    sources, root = source_hashes(), Path(__file__).resolve().parents[1]
    with tarfile.open(args.output/'frozen-source.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(root/path, arcname=path)
    plan = dict(episodes=args.episodes, train_budget=args.train_budget, eval_budget=args.eval_budget,
                data_seed=args.data_seed, policy_seeds=args.policy_seeds, methods=METHODS,
                source_hashes=sources, splits={s: len(v) for s, v in data.items()},
                architecture=dict(hidden=32, heads=4, layers=2, edge_features=6),
                actions='every singleton/pair, three rule-based triples, eight original MILP actions',
                failure_memory='mask failed profile subset until incumbent schedule changes',
                teacher=dict(stages=[0, 2, 6, 12], epochs=args.teacher_epochs, temperature=.2,
                             utility='normalized gain / max(measured repair seconds, .002)',
                             profiles_allowance=.1),
                ppo=dict(batch_episodes=8, epochs=3, trace_lambda=1, entropy_coef=.001,
                         minibatch_size=16, clip=.2, target_kl=.03, learning_rate=.0003,
                         teacher_replay_states=8),
                stage_probabilities=[.4, .25, .2, .15], checkpoint='final fixed in advance',
                baseline_checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in baseline_paths},
                hardware=dict(platform=platform.platform(), torch=str(torch.__version__), threads=1),
                archive_sha256=hashlib.sha256((args.output/'frozen-source.tar.gz').read_bytes()).hexdigest())
    write_json(args.output/'plan.json', plan)
    pools, runtime = training_pools(data['train'], args.data_seed)
    torch.save(pools, args.output/'training-incumbents.pt')
    costs, rows = [dict(method='training_pools', runtime=runtime)], []
    with (args.output/'evaluation.jsonl').open('w') as stream:
        for seed in args.policy_seeds:
            examples, record = collect_teachers(data['train'], pools, seed=seed, budget=args.train_budget)
            write_json(args.output/f'teachers-{seed}.json', record)
            costs.append(dict(seed=seed, method='teachers', runtime=record['runtime']))
            torch.save(examples, args.output/f'teacher-observations-{seed}.pt')
            policies = {}
            for name, constructor in [('flat_hybrid', PortfolioActorCritic),
                    ('graph_off_hybrid', lambda: BipartiteActorCritic(graph=False)),
                    ('graph_hybrid', BipartiteActorCritic)]:
                torch.manual_seed(seed)
                policy = constructor()
                started = perf_counter()
                metrics = listwise_update(policy, torch.optim.Adam(policy.parameters(), lr=.001),
                                           examples, epochs=args.teacher_epochs, seed=seed)
                costs.append(dict(seed=seed, method=name+'_distillation', runtime=perf_counter()-started, **metrics))
                if name == 'graph_hybrid':
                    policies['graph_il'] = copy.deepcopy(policy)
                costs.append(train_hybrid(policy, examples, data['train'], pools, seed=seed,
                    episodes=args.episodes, budget=args.train_budget, output=args.output, name=name))
                policies[name] = policy
            policies['graph_greedy'] = policies['graph_hybrid']
            for name, policy in policies.items():
                policy.save(args.output/f'{name}-{seed}.pt', metadata=dict(method=name, plan=plan))
            template = PortfolioActorCritic.load(args.template_results/f'preference_ppo-{seed}.pt')
            policies['template_checkpoint'] = template
            for split in ('validation', 'development', 'ood'):
                for index, (instance, witness) in enumerate(data[split]):
                    search_seed = seed*100000+{'validation':10000, 'development':20000, 'ood':30000}[split]+index
                    for method in np.random.default_rng(search_seed).permutation(METHODS):
                        row = evaluate(instance, witness, str(method), policies.get(method),
                                       budget=args.eval_budget, seed=search_seed)
                        row.update(seed=seed, split=split, instance=index, instance_hash=instance_hash(instance))
                        rows.append(row)
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                        stream.flush()
                    print(f'seed={seed} evaluated={split}:{index}', flush=True)
            write_json(args.output/'offline-costs.json', costs)
    if source_hashes() != sources:
        raise RuntimeError('Source changed during frozen experiment')
    summarize(args.output, rows, budget=args.eval_budget)
    write_json(args.output/'integrity.json', dict(runs=len(rows), all_feasible=all(r['feasible'] for r in rows),
        sources_match=True, evaluation_sha256=hashlib.sha256((args.output/'evaluation.jsonl').read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
