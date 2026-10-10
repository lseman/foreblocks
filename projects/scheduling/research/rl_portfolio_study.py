"""Development study of RL-controlled profile/MILP portfolios for paper2."""
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
from ..milp import ONTSConfig
from ..rl_learning import ppo_batch_update, preference_update
from ..rl_portfolio import FAMILIES, PortfolioActorCritic, PortfolioEnv, portfolio_rollout
from .datasets import instance_hash, source_hashes
from .rl_upgrade import evaluate as evaluate_profile, generate_splits
from .statistics import bootstrap_mean

METHODS = ('uniform', 'bandit', 'profile_only', 'untrained', 'preference',
           'ppo', 'preference_ppo', 'profile_heuristic')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def initial_pool(data, seed):
    """Saved training incumbents: witness, two profile moves, six profile moves."""
    started, pools = perf_counter(), []
    for index, (instance, witness) in enumerate(data):
        env = PortfolioEnv(instance, witness, budget=5, seed=seed+index)
        pool = [witness.bool().clone()]
        for step in range(6):
            env.step(9+(step % 9))
            if step in (1, 5):
                pool.append(env.incumbent.clone())
        pools.append(pool)
    return pools, perf_counter()-started


def teachers(data, pools, policy, seed, budget):
    examples, logs = [], []
    started = perf_counter()
    for index, (instance, _) in enumerate(data):
        stage = index % 2
        env = PortfolioEnv(instance, pools[index][stage], budget=budget, seed=seed+index)
        observation, mask = policy.observation(env)
        utilities, outcomes = [0.], []
        for j, action in enumerate(env.actions):
            tick = perf_counter()
            if action.family < 4:
                result = env.problem.solve(time_limit=action.allowance,
                            incumbent=env.incumbent, free_mask=action.mask, solver_threads=1)
                candidate, status = result.schedule, result.status
            else:
                candidate, _, _ = reprofile(env.problem, env.incumbent, action.jobs,
                            np.random.default_rng(seed+1000*index+j), deadline=perf_counter()+.1)
                status = 'profile'
            report = env.problem.config.validate(instance, candidate)
            if not report.feasible:
                raise RuntimeError('Invalid teacher candidate')
            duration = perf_counter()-tick
            gain = max(0., report.objective-env.objective)/env.scale
            utilities.append(gain/max(duration, .002))
            outcomes.append(dict(family=FAMILIES[action.family], jobs=action.jobs,
                                 allowance=action.allowance, gain=gain, duration=duration,
                                 status=status))
        examples.append((observation, mask, utilities))
        logs.append(dict(instance=index, initial_stage=stage, utilities=utilities, outcomes=outcomes))
    return examples, dict(runtime=perf_counter()-started, states=len(examples), records=logs)


def evaluate(instance, witness, method, policy, *, budget, seed):
    if method == 'profile_heuristic':
        row = evaluate_profile(instance, witness, method='profile_heuristic', policy=None,
                               budget=budget, repair_time=.1, search_seed=seed)
        row['method'] = method
        row['profile_steps'] = sum(t.get('operator') == 'profile' for t in row['transitions'])
        row['milp_steps'] = sum(t.get('operator') == 'repair' for t in row['transitions'])
        return row
    started = perf_counter()
    torch.manual_seed(seed)
    env = PortfolioEnv(instance, witness, budget=budget, seed=seed)
    env.deadline = min(env.deadline, started+budget-env.reserve)
    mode = method if method in ('uniform', 'bandit', 'profile_only') else 'ppo'
    portfolio_rollout(env, policy, mode=mode)
    report = env.problem.config.validate(instance, env.incumbent)
    runtime = perf_counter()-started
    if not report.feasible:
        raise RuntimeError('Invalid final incumbent')
    improvement = (env.objective-env.initial_objective)/env.scale
    if not np.isclose(improvement, sum(t['reward'] for t in env.log)):
        raise RuntimeError('Reward accounting mismatch')
    return dict(method=method, feasible=True, initial=env.initial_objective,
                objective=env.objective, returned_objective=report.objective,
                improvement=improvement, runtime=runtime, overrun=max(0., runtime-budget),
                steps=sum(t['executed'] for t in env.log),
                profile_steps=sum(t['executed'] and t['family'].startswith('profile') for t in env.log),
                milp_steps=sum(t['executed'] and t['family'].startswith('milp') for t in env.log),
                selection_runtime=sum(t['selection_runtime'] for t in env.log),
                schedule=env.incumbent.int().tolist(), transitions=env.log)


def report(output, rows, costs):
    lines = ['# Learned profile/MILP portfolio development study', '',
             'Quality is normalized improvement over the supplied witness, not an optimum gap.',
             'Two policy seeds are averaged within instance. All intervals are exploratory,',
             'marginal instance-level bootstrap intervals without multiplicity correction.', '',
             '| Split | Selector | Feasible | Improvement | Profile actions | MILP actions |',
             '| --- | --- | --- | --- | --- | --- |']
    for split in ('validation', 'development', 'ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split']==split and r['method']==method]
            if not selected:
                continue
            lines.append(f"| {split} | {method} | {sum(r['feasible'] for r in selected)}/{len(selected)} | "
                         f"{np.mean([r['improvement'] for r in selected]):.6f} | "
                         f"{np.mean([r.get('profile_steps', 0) for r in selected]):.1f} | "
                         f"{np.mean([r.get('milp_steps', 0) for r in selected]):.1f} |")
    comparisons = []
    for split in ('development', 'ood'):
        for method, baseline in [('ppo', 'untrained'), ('preference_ppo', 'preference'),
                                 ('preference_ppo', 'ppo'), ('preference_ppo', 'uniform'),
                                 ('preference_ppo', 'bandit'), ('preference_ppo', 'profile_only'),
                                 ('preference_ppo', 'profile_heuristic')]:
            deltas = []
            for index in sorted({r['instance'] for r in rows if r['split']==split}):
                a = [r['improvement'] for r in rows if r['split']==split and r['instance']==index and r['method']==method]
                b = [r['improvement'] for r in rows if r['split']==split and r['instance']==index and r['method']==baseline]
                deltas.append(float(np.mean(a)-np.mean(b)))
            comparisons.append(dict(split=split, method=method, baseline=baseline,
                                    delta=bootstrap_mean(deltas, resamples=2000)))
    lines += ['', '| Split | Comparison | Difference | Marginal 95% interval |',
              '| --- | --- | --- | --- |']
    for c in comparisons:
        lo, hi = c['delta']['ci95']
        lines.append(f"| {c['split']} | {c['method']} − {c['baseline']} | "
                     f"{c['delta']['mean']:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'The learned policy encodes task features, the full time-indexed solar/battery',
              'trajectory and candidate actions with temporal and candidate self-attention.',
              'Uniform and contextual-bandit controls share all 17 candidate actions. Profile-only',
              'selection uses the nine profile templates; the full heuristic has its own controller.',
              'All learned-policy evaluations sample actions, rather than choosing checkpoints or',
              'inference modes from development performance.', '',
              'Warm-start labels use one-action normalized gain per measured second. They are',
              'offline rate preferences, not PPO rewards. PPO uses undiscounted objective increments',
              'and Monte Carlo returns (gamma=1, lambda=1) for action-dependent-duration episodes.',
              'Training samples witness, partially improved and further improved incumbents.',
              'Validation is diagnostic only; the final checkpoint is fixed before evaluation.', '',
              'The study uses new content-disjoint planted instances in the same generator family.',
              'Three larger instances define size OOD; cold starts, constellations, external',
              'benchmarks and independently retrained architecture ablations are not evaluated.',
              'Policy and evaluation seeds are coupled, so their variance components are not',
              'separately estimated. This study cannot establish ONTS state-of-the-art performance.', '',
              '## Offline costs', '', '```json', json.dumps(costs, indent=2), '```']
    write_json(output/'analysis.json', comparisons)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:34]), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--train-budget', type=float, default=.6)
    parser.add_argument('--eval-budget', type=float, default=1.2)
    parser.add_argument('--data-seed', type=int, default=2026102000)
    parser.add_argument('--policy-seeds', type=int, nargs='+', default=[11, 29])
    args = parser.parse_args()
    if args.episodes < 1 or len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('Positive episodes and unique policy seeds required')
    if not all(np.isfinite(v) and v > .15 for v in (args.train_budget, args.eval_budget)):
        parser.error('Budgets must be finite and exceed .15 seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ['ONTS_SOLVER_THREADS'] = '1'
    data = generate_splits(args.data_seed)
    hashes = {split: [instance_hash(i) for i, _ in entries] for split, entries in data.items()}
    # Previous examined splits must not supply new development evidence.
    for previous in (args.output.parent/'rl-upgrade-20261006'/'manifest.json',):
        if previous.exists():
            old = json.loads(previous.read_text())
            if set(sum(hashes.values(), [])) & {entry['hash'] for rows in old.values() for entry in rows}:
                raise RuntimeError('Overlap with previously examined instances')
    torch.save({split: [(vars(i), x) for i, x in entries] for split, entries in data.items()}, args.output/'instances.pt')
    write_json(args.output/'manifest.json', hashes)
    sources = source_hashes()
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(args.output/'frozen-source.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(root/path, arcname=path)
    plan = dict(episodes=args.episodes, train_budget=args.train_budget, eval_budget=args.eval_budget,
                data_seed=args.data_seed, policy_seeds=args.policy_seeds, methods=METHODS,
                source_hashes=sources, families=FAMILIES, repair_budgets=[.025, .1],
                profile_counts=[1, 2, 3], profile_floor=.002, transformer=dict(hidden=32, heads=4),
                ppo=dict(batch_episodes=8, epochs=4, minibatch_size=8, trace_lambda=1,
                         entropy_coef=.002, learning_rate=.0003, clip=.2, target_kl=.03),
                preference=dict(epochs=8, learning_rate=.001, objective='gain per measured second',
                                profile_teacher_allowance=.1),
                initial_distribution=[.5, .35, .15], checkpoint='final, fixed in advance',
                hardware=dict(platform=platform.platform(), torch=str(torch.__version__), threads=1),
                archive_sha256=hashlib.sha256((args.output/'frozen-source.tar.gz').read_bytes()).hexdigest())
    write_json(args.output/'plan.json', plan)
    pools, init_cost = initial_pool(data['train'], args.data_seed)
    torch.save(pools, args.output/'training-incumbents.pt')
    costs, rows = [dict(method='initial_incumbent_pool', seconds=init_cost)], []
    with (args.output/'training.jsonl').open('w') as training, (args.output/'evaluation.jsonl').open('w') as evaluation:
        for seed in args.policy_seeds:
            torch.manual_seed(seed)
            fresh = PortfolioActorCritic()
            warm = copy.deepcopy(fresh)
            teacher, record = teachers(data['train'], pools, warm, seed, args.train_budget)
            write_json(args.output/f'teachers-{seed}.json', record)
            costs.append(dict(seed=seed, method='teachers', seconds=record['runtime']))
            started = perf_counter()
            metrics = preference_update(warm, torch.optim.Adam(warm.parameters(), lr=.001), teacher, epochs=8)
            costs.append(dict(seed=seed, method='preference', seconds=perf_counter()-started, **metrics))
            policies = dict(untrained=fresh, preference=copy.deepcopy(warm))
            for method, initial in [('ppo', fresh), ('preference_ppo', warm)]:
                policy = copy.deepcopy(initial)
                torch.manual_seed(seed)
                rng = np.random.default_rng(seed)
                optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
                pending, started = [], perf_counter()
                for episode in range(args.episodes):
                    index = (episode*5+seed) % len(data['train'])
                    stage = int(rng.choice(3, p=[.5, .35, .15]))
                    instance, _ = data['train'][index]
                    env = PortfolioEnv(instance, pools[index][stage], budget=args.train_budget,
                                       seed=seed*10000+episode)
                    trajectory = portfolio_rollout(env, policy)
                    pending.append(trajectory)
                    metrics = None
                    if len(pending)==8 or episode==args.episodes-1:
                        metrics = ppo_batch_update(policy, optimizer, pending, trace_lambda=1,
                                                  entropy_coef=.002, minibatch_size=8, seed=seed+episode)
                        pending = []
                    training.write(json.dumps(dict(seed=seed, method=method, episode=episode,
                            instance=index, initial_stage=stage, metrics=metrics, transitions=env.log), allow_nan=False)+'\n')
                    training.flush()
                    if (episode+1) % 8 == 0:
                        print(f'seed={seed} method={method} episodes={episode+1}/{args.episodes}', flush=True)
                policies[method] = policy
                costs.append(dict(seed=seed, method=method, seconds=perf_counter()-started))
            for method, policy in policies.items():
                policy.save(args.output/f'{method}-{seed}.pt', metadata=dict(method=method, plan=plan))
            for split in ('validation', 'development', 'ood'):
                for index, (instance, witness) in enumerate(data[split]):
                    search_seed = seed*100000+{'validation':10000, 'development':20000, 'ood':30000}[split]+index
                    for method in np.random.default_rng(search_seed).permutation(METHODS):
                        row = evaluate(instance, witness, str(method), policies.get(method),
                                       budget=args.eval_budget, seed=search_seed)
                        row.update(split=split, instance=index, seed=seed, instance_hash=instance_hash(instance))
                        rows.append(row)
                        evaluation.write(json.dumps(row, allow_nan=False)+'\n')
                        evaluation.flush()
                    print(f'seed={seed} evaluated={split}:{index}', flush=True)
            write_json(args.output/'offline-costs.json', costs)
    if sources != source_hashes():
        raise RuntimeError('Sources changed; the snapshot must be used for replay')
    report(args.output, rows, costs)
    write_json(args.output/'integrity.json', dict(runs=len(rows), all_feasible=all(r['feasible'] for r in rows),
                sources_match=True, evaluation_sha256=hashlib.sha256((args.output/'evaluation.jsonl').read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
