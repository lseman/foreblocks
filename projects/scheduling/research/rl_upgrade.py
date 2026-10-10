"""Paired development experiment for stable PPO and semantic graph search.

Fresh planted splits, offline preference teachers, two independent policy seeds,
matched wall budgets, durable transition logs and source snapshots. Not a
confirmatory study or a claim of ONTS state of the art.
"""
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

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig
from ..profile_portfolio import solve_profile_portfolio
from ..rl_learning import (GraphActorCritic, initialize_value_head,
                           ppo_batch_update, preference_update)
from ..rl_search import ActorCritic, SearchEnv, ppo_update, rollout
from .datasets import instance_hash, source_hashes, witness_instance
from .statistics import bootstrap_mean

METHODS = ('uniform', 'utility', 'bandit', 'legacy_ppo', 'stable_ppo',
           'graph_ppo', 'graph_preference', 'graph_preference_ppo', 'profile_heuristic')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def generate_splits(seed):
    regimes = ('balanced', 'energy', 'periodic')
    splits = {}
    for split, offset, count, large in [('train', 0, 12, False),
                                       ('validation', 10000, 3, False),
                                       ('development', 20000, 6, False),
                                       ('ood', 30000, 3, True)]:
        splits[split] = [witness_instance(seed=seed+offset+i, regime=regimes[i % 3], large=large)
                         for i in range(count)]
    hashes = [instance_hash(instance) for data in splits.values() for instance, _ in data]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('Content overlap between generated instances')
    return splits


def collect_preferences(data, policy, *, budget, repair_time, seed):
    examples, records = [], []
    started = perf_counter()
    for index, (instance, initial) in enumerate(data):
        for stage in range(2):
            env = SearchEnv(instance, initial, budget=budget,
                            repair_time=repair_time, seed=seed+index*2+stage,
                            stop_mode='exhaustion')
            if stage:
                # Include states encountered after search, not only sparse witnesses.
                env.step(1+(index % 4))
                env.step(1+((index+1) % 4))
            observation, mask = policy.observation(env)
            gains = [0.0]
            outcomes = []
            for candidate in env.candidates:
                result = env.problem.solve(time_limit=repair_time, incumbent=env.incumbent,
                                           free_mask=candidate.mask, solver_threads=1)
                report = env.problem.config.validate(instance, result.schedule)
                if not report.feasible:
                    raise RuntimeError('Invalid teacher repair')
                gains.append(max(0.0, report.objective-env.objective)/env.scale)
                outcomes.append(dict(operator=candidate.operator, objective=report.objective,
                                     status=result.status, runtime=result.runtime))
            examples.append((observation, mask, gains))
            records.append(dict(instance=index, stage=stage, initial_objective=env.objective,
                                gains=gains, outcomes=outcomes))
    return examples, dict(runtime=perf_counter()-started, states=len(examples), records=records)


def evaluate(instance, initial, *, method, policy, budget, repair_time, search_seed):
    started = perf_counter()
    torch.manual_seed(search_seed)
    config = ONTSConfig()
    initial_objective = config.validate(instance, initial).objective
    scale = max(float(instance.priority.clamp(min=0).sum())*instance.T, 1.0)
    if method == 'profile_heuristic':
        profile_budget = budget-(perf_counter()-started)
        result = solve_profile_portfolio(instance, initial_schedule=initial,
                                         time_limit=profile_budget, repair_time=repair_time,
                                         seed=search_seed)
        report = config.validate(instance, result.schedule)
        # The backend logs validated incumbent events. Never credit late events.
        objective = max([initial_objective]+[
            event['objective'] for event in result.history
            if event.get('objective') is not None and event['time'] <= profile_budget-min(.25, profile_budget*.05)])
        schedule = result.schedule
        log, steps = result.history, result.iterations
        stops, selection = 0, 0.0
    else:
        stop_mode = 'optional' if method == 'legacy_ppo' else 'exhaustion'
        env = SearchEnv(instance, initial, budget=budget, repair_time=repair_time,
                        seed=search_seed, stop_mode=stop_mode)
        env.deadline = min(env.deadline, started+budget-env.reserve)
        mode = method if method in ('uniform', 'utility', 'bandit') else 'ppo'
        rollout(env, policy, mode=mode, bandit=[np.eye(18), np.zeros(18)])
        report = config.validate(instance, env.incumbent)
        objective, schedule = env.objective, env.incumbent
        log, steps = env.log, sum(t['action'] != 0 for t in env.log)
        stops = sum(t['action'] == 0 for t in env.log)
        selection = sum(t.get('selection_runtime', 0) for t in env.log)
        if not np.isclose(sum(t['reward'] for t in log), (objective-initial_objective)/scale):
            raise RuntimeError('Reward accounting mismatch')
    runtime = perf_counter()-started
    return dict(method=method, feasible=report.feasible, objective=objective,
                returned_objective=report.objective, initial=initial_objective,
                improvement=(objective-initial_objective)/scale,
                runtime=runtime, overrun=max(0.0, runtime-budget),
                steps=steps, stops=stops, selection_runtime=selection,
                schedule=schedule.int().tolist(), transitions=log)


def report_results(output, rows, training_costs):
    lines = ['# RL upgrade: paired development experiment', '',
             'All quality values are improvements divided by the unconstrained weighted-execution',
             'scale, not percentage gains relative to a baseline or optimality gaps.',
             'Seed means are averaged within instance. Intervals below are exploratory and marginal.', '',
             '| Split | Method | Feasible | Mean improvement | Mean repairs | Mean selection seconds |',
             '| --- | --- | --- | --- | --- | --- |']
    for split in ('validation', 'development', 'ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue
            lines.append(f"| {split} | {method} | {sum(r['feasible'] for r in selected)}/{len(selected)} | "
                         f"{np.mean([r['improvement'] for r in selected]):.6f} | "
                         f"{np.mean([r['steps'] for r in selected]):.1f} | "
                         f"{np.mean([r['selection_runtime'] for r in selected]):.4f} |")
    comparisons = []
    for split in ('development', 'ood'):
        for method, baseline in [('stable_ppo', 'legacy_ppo'), ('graph_ppo', 'stable_ppo'),
                                 ('graph_preference_ppo', 'graph_ppo'),
                                 ('graph_preference_ppo', 'graph_preference'),
                                 ('graph_preference_ppo', 'bandit'),
                                 ('graph_preference_ppo', 'profile_heuristic')]:
            instance_deltas = []
            ids = sorted({r['instance'] for r in rows if r['split'] == split})
            for index in ids:
                a = [r['improvement'] for r in rows if r['split'] == split and r['instance'] == index and r['method'] == method]
                b = [r['improvement'] for r in rows if r['split'] == split and r['instance'] == index and r['method'] == baseline]
                if a and b:
                    instance_deltas.append(float(np.mean(a)-np.mean(b)))
            if instance_deltas:
                comparison = dict(split=split, method=method, baseline=baseline,
                                  delta=bootstrap_mean(instance_deltas, resamples=2000))
                comparisons.append(comparison)
    lines += ['', '| Split | Comparison | Mean difference | Marginal bootstrap 95% interval |',
              '| --- | --- | --- | --- |']
    for c in comparisons:
        mean = c['delta']['mean']
        lo, hi = c['delta']['ci95']
        lines.append(f"| {c['split']} | {c['method']} − {c['baseline']} | {mean:.6f} | [{lo:.6f}, {hi:.6f}] |")
    lines += ['', 'Legacy PPO retains optional STOP and single-episode updates. Stable PPO changes',
              'STOP masking, critic initialization and batched PPO together; this is a package',
              'comparison and cannot attribute gains to one change. Graph PPO uses the same',
              'stabilized training, adding typed sparse MILP attention and explicit operator history.',
              'Graph preference is the warm-start checkpoint before RL; graph preference PPO',
              'continues from it. Offline teacher costs are recorded separately.', '',
              'All learned policies use stochastic categorical evaluation with matched search seeds.',
              'There is one evaluation trajectory per policy-seed/instance/method. Policy training',
              'seeds and evaluation seeds are paired, not independent variance estimates.',
              'Final checkpoints are fixed in advance; validation is diagnostic only.', '',
              'The profile heuristic has its own profile/MILP action portfolio. It is an end-to-end',
              'quality comparator, while the other methods isolate selection over four MILP masks.',
              'Profile deadline quality comes from validated event history; its separately saved',
              'final schedule may have a higher objective if the last event arrived late.',
              'All methods include setup, inference, repair and validation. Native solver limits',
              'are cooperative; runtime overruns are saved and late outcomes receive no credit.', '',
              'These are small planted synthetic splits, with larger instances as OOD, not a',
              'new generator family or external benchmark. No cold-start, constellation, certified',
              'optimality, independent retraining of feature ablations, or SOTA claim is supported.',
              'Bootstrap intervals do not include multiple-comparison correction.', '',
              '## Offline costs', '']
    lines.append('```json\n'+json.dumps(training_costs, indent=2)+'\n```')
    write_json(output/'analysis.json', comparisons)
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:35]), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--train-budget', type=float, default=0.9)
    parser.add_argument('--eval-budget', type=float, default=1.2)
    parser.add_argument('--repair-time', type=float, default=0.1)
    parser.add_argument('--data-seed', type=int, default=2026100610)
    parser.add_argument('--policy-seeds', type=int, nargs='+', default=[11, 29])
    args = parser.parse_args()
    if args.episodes < 1 or not args.policy_seeds or len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('Positive episode count and unique policy seeds required')
    if not all(np.isfinite(v) and v > 0 for v in (args.train_budget, args.eval_budget, args.repair_time)):
        parser.error('Positive finite budgets required')
    if args.repair_time >= min(args.train_budget, args.eval_budget)*.8:
        parser.error('Repair time must leave room for setup and several decisions')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    os.environ['ONTS_SOLVER_THREADS'] = '1'
    data = generate_splits(args.data_seed)
    manifest = {split: [dict(hash=instance_hash(i), instance=onts_instance_to_json_data(i),
                            witness=x.int().tolist()) for i, x in entries]
                for split, entries in data.items()}
    write_json(args.output/'manifest.json', manifest)
    # Preserve float64 tensors for exact replay; JSON loaders can change dtype.
    torch.save({split: [(vars(instance), x) for instance, x in entries]
                for split, entries in data.items()}, args.output/'instances.pt')
    sources = source_hashes()
    root = Path(__file__).resolve().parents[1]
    with tarfile.open(args.output/'frozen-source.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(root/path, arcname=path)
    plan = dict(episodes=args.episodes, train_budget=args.train_budget,
                eval_budget=args.eval_budget, repair_time=args.repair_time,
                data_seed=args.data_seed, policy_seeds=args.policy_seeds,
                sources=sources, checkpoint='final episode, fixed before evaluation',
                methods=METHODS, teacher_stages=2, preference_epochs=8,
                ppo=dict(batch_episodes=8, epochs=4, minibatch_size=4, trace_lambda=.95,
                         clip=.2, target_kl=.03, entropy_coef=.01, learning_rate=.0003),
                hardware=dict(platform=platform.platform(), torch=str(torch.__version__), threads=1),
                archive_sha256=hashlib.sha256((args.output/'frozen-source.tar.gz').read_bytes()).hexdigest())
    write_json(args.output/'plan.json', plan)
    rows, costs = [], []
    with (args.output/'training.jsonl').open('w') as training, (args.output/'evaluation.jsonl').open('w') as evaluation:
        for seed in args.policy_seeds:
            torch.manual_seed(seed)
            warm = GraphActorCritic()
            teacher, teacher_record = collect_preferences(data['train'], warm,
                        budget=args.train_budget, repair_time=args.repair_time, seed=seed+70000)
            write_json(args.output/f'teachers-{seed}.json', teacher_record)
            start = perf_counter()
            pref_optimizer = torch.optim.Adam(warm.parameters(), lr=1e-3)
            pref_metrics = preference_update(warm, pref_optimizer, teacher, epochs=8)
            costs.append(dict(seed=seed, method='preference_teacher', seconds=teacher_record['runtime']))
            costs.append(dict(seed=seed, method='preference_pretraining', seconds=perf_counter()-start, **pref_metrics))
            policies = {'graph_preference': copy.deepcopy(warm)}
            for method in ('legacy_ppo', 'stable_ppo', 'graph_ppo', 'graph_preference_ppo'):
                torch.manual_seed(seed)
                policy = (copy.deepcopy(warm) if method == 'graph_preference_ppo' else
                          GraphActorCritic() if method == 'graph_ppo' else ActorCritic())
                if method == 'stable_ppo':
                    initialize_value_head(policy)
                optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
                pending = []
                started = perf_counter()
                for episode in range(args.episodes):
                    index = (episode*5+seed) % len(data['train'])
                    instance, witness = data['train'][index]
                    env = SearchEnv(instance, witness, budget=args.train_budget,
                                    repair_time=args.repair_time, seed=seed*10000+episode,
                                    stop_mode='optional' if method == 'legacy_ppo' else 'exhaustion')
                    trajectory = rollout(env, policy)
                    if method == 'legacy_ppo':
                        metrics = ppo_update(policy, optimizer, trajectory)
                    else:
                        pending.append(trajectory)
                        metrics = None
                        if len(pending) == 8 or episode == args.episodes-1:
                            metrics = ppo_batch_update(policy, optimizer, pending,
                                                       minibatch_size=4, seed=seed+episode)
                            pending = []
                    training.write(json.dumps(dict(seed=seed, method=method, episode=episode,
                        instance=index, metrics=metrics, transitions=env.log), allow_nan=False)+'\n')
                    training.flush()
                    if (episode+1) % 8 == 0:
                        print(f'seed={seed} method={method} episodes={episode+1}/{args.episodes}', flush=True)
                policies[method] = policy
                costs.append(dict(seed=seed, method=method, seconds=perf_counter()-started))
            for method, policy in policies.items():
                is_graph = isinstance(policy, GraphActorCritic)
                torch.save(dict(format='onts-rl-upgrade-v1', method=method,
                                architecture='graph' if is_graph else 'compact',
                                model_config=dict(hidden=policy.hidden, layers=policy.layers,
                                                  semantics=policy.semantics, memory=policy.memory)
                                if is_graph else dict(width=32),
                                state_dict=policy.state_dict(), plan=plan), args.output/f'{method}-{seed}.pt')
            # Randomize method order per instance to reduce execution-order bias.
            for split in ('validation', 'development', 'ood'):
                for index, (instance, witness) in enumerate(data[split]):
                    search_seed = seed*100000+ {'validation':10000, 'development':20000, 'ood':30000}[split]+index
                    for method in np.random.default_rng(search_seed).permutation(METHODS):
                        row = evaluate(instance, witness, method=str(method), policy=policies.get(method),
                                       budget=args.eval_budget, repair_time=args.repair_time,
                                       search_seed=search_seed)
                        row.update(split=split, instance=index, seed=seed, instance_hash=instance_hash(instance))
                        if not row['feasible']:
                            raise RuntimeError('Evaluation returned an infeasible incumbent')
                        rows.append(row)
                        evaluation.write(json.dumps(row, allow_nan=False)+'\n')
                        evaluation.flush()
                    print(f'seed={seed} evaluated={split}:{index}', flush=True)
            write_json(args.output/'offline-costs.json', costs)
        if source_hashes() != sources:
            raise RuntimeError('Source changed during execution; use the preserved snapshot to audit')
        report_results(args.output, rows, costs)
    write_json(args.output/'integrity.json', dict(source_hashes_match=True, runs=len(rows),
               feasible=sum(r['feasible'] for r in rows),
               manifest_sha256=hashlib.sha256((args.output/'manifest.json').read_bytes()).hexdigest(),
               evaluation_sha256=hashlib.sha256((args.output/'evaluation.jsonl').read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
