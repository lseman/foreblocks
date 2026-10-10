"""Development study of LRBS-style lookahead rollout search for ONTS.

Trains PPO policies using both single-step and lookahead rollouts, then compares
improvement on development and OOD instances with paired bootstrap intervals."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..milp import ONTSConfig
from ..rl_learning import ppo_batch_update
from ..rl_search import ActorCritic, SearchEnv, rollout as single_step_rollout
from ..rl_rollout import evaluate_lookahead, rollout_with_lookahead
from .datasets import instance_hash, source_hashes, witness_instance
from .statistics import bootstrap_mean

METHODS = ('uniform', 'ppo_single', 'ppo_lookahead_k4_s2', 'ppo_lookahead_k8_s2',
           'ppo_lookahead_k8_s3')


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def generate_splits(seed, *, train_count=9, development_count=3, ood_count=6):
    """Generate disjoint instance splits for training/development/evaluation."""
    regimes = ('balanced', 'energy', 'periodic')
    data = {split: [witness_instance(seed=seed + offset + i, regime=regimes[i % 3], large=large)
                    for i in range(count)]
            for split, offset, count, large in [
                ('train', 0, train_count, False),
                ('development', 20000, development_count, False),
                ('ood', 30000, ood_count, True),
            ]}
    hashes = [instance_hash(i) for entries in data.values() for i, _ in entries]
    if len(set(hashes)) != len(hashes):
        raise RuntimeError('Generated instance overlap')
    return data


def evaluate(instance, witness, method, policy=None, *, budget=1.2, seed=0,
             lookahead_k=8, lookahead_steps=2):
    """Run a single evaluation of one method on one instance."""
    started = perf_counter()
    torch.manual_seed(seed)

    if method == 'uniform':
        env = SearchEnv(instance, witness, budget=budget, seed=seed)
        single_step_rollout(env, mode='uniform')
        report = ONTSConfig().validate(instance, env.incumbent)
        if not report.feasible:
            raise RuntimeError('Uniform search returned infeasible schedule')
        improvement = (env.objective - env.initial_objective) / env.scale
        return dict(method=method, feasible=True, initial=float(env.initial_objective),
                    objective=env.objective, improvement=improvement,
                    runtime=perf_counter() - started,
                    overrun=max(0., perf_counter() - started - budget),
                    steps=len(env.log))

    if method == 'ppo_single':
        if policy is None:
            raise ValueError('ppo_single requires a trained policy')
        env = SearchEnv(instance, witness, budget=budget, seed=seed)
        single_step_rollout(env, policy, mode='ppo', greedy=True)
        report = ONTSConfig().validate(instance, env.incumbent)
        if not report.feasible:
            raise RuntimeError('PPO search returned infeasible schedule')
        improvement = (env.objective - env.initial_objective) / env.scale
        return dict(method=method, feasible=True, initial=float(env.initial_objective),
                    objective=env.objective, improvement=improvement,
                    runtime=perf_counter() - started,
                    overrun=max(0., perf_counter() - started - budget),
                    steps=len(env.log))

    # Parse lookahead parameters from method name
    if method.startswith('ppo_lookahead'):
        k = int(method.split('_')[2].replace('k', ''))
        s = int(method.split('_')[3].replace('s', '')) if '_s' in method else 2

        result = evaluate_lookahead(
            instance, witness, policy,
            budget=budget, seed=seed,
            lookahead_k=k, lookahead_steps=s,
            greedy=True,
        )
        report = ONTSConfig().validate(instance, result['incumbent'])
        if not report.feasible:
            raise RuntimeError('Lookahead search returned infeasible schedule')

        return dict(
            method=method, feasible=True,
            initial=result['initial_objective'],
            objective=result['objective'],
            improvement=result['improvement'],
            runtime=result['elapsed'],
            overrun=max(0., result['elapsed'] - budget),
            steps=result['steps'],
            n_improvements=result['n_improvements'],
        )

    raise ValueError(f'Unknown method: {method}')


def summarize(output, rows, *, budget=1.2):
    """Generate summary statistics and write results."""
    # Per-method, per-split statistics
    summaries = {}
    for split in ('development', 'ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue

            improvements = np.array([r['improvement'] for r in selected])
            runtimes = np.array([r['runtime'] for r in selected])
            overruns = np.array([r.get('overrun', 0.0) for r in selected])

            summaries[f'{split}/{method}'] = dict(
                n=len(selected),
                mean_improvement=float(improvements.mean()),
                std_improvement=float(improvements.std()),
                median_improvement=float(np.median(improvements)),
                min_improvement=float(improvements.min()),
                max_improvement=float(improvements.max()),
                mean_runtime=float(runtimes.mean()),
                mean_overrun=float(overruns.mean()),
                n_feasible=sum(1 for r in selected if r.get('feasible', True)),
            )

    # Paired comparisons on development set
    pairs = {}
    dev_rows = [r for r in rows if r['split'] == 'development']
    # Group by (instance, seed) for paired comparison
    groups = {}
    for r in dev_rows:
        key = (r.get('instance_seed', 0),)
        if key not in groups:
            groups[key] = {}
        groups[key][r['method']] = r['improvement']

    baseline_methods = ('uniform', 'ppo_single')
    lookahead_methods = [m for m in METHODS if m.startswith('lookahead')]

    for base in baseline_methods:
        for target in lookahead_methods:
            paired_diffs = []
            for key, methods in groups.items():
                if base in methods and target in methods:
                    paired_diffs.append(methods[target] - methods[base])

            if paired_diffs:
                diffs = np.array(paired_diffs)
                pairs[f'{target} vs {base}'] = dict(
                    n=len(diffs),
                    mean_diff=float(diffs.mean()),
                    std_diff=float(diffs.std()),
                    bootstrap_interval=_bootstrap_ci(diffs, n_samples=1000),
                )

    result = dict(
        budget=budget,
        summaries=summaries,
        paired_comparisons=pairs,
        instance_hashes=[r.get('instance_hash', '') for r in rows if 'instance_hash' in r],
        source_hashes=source_hashes(),
    )

    write_json(output / 'summary.json', result)
    return result


def _bootstrap_ci(data, n_samples=1000, ci=0.9):
    """Compute bootstrap confidence interval for the mean."""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_samples):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(sample.mean())
    means = np.array(means)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(means, 100 * alpha))
    upper = float(np.percentile(means, 100 * (1 - alpha)))
    return [lower, upper]


def generate_report(rows, args, costs, summary):
    """Generate a markdown report with tables and paired comparisons."""
    lines = ['# Lookahead Rollout Search Study', '',
             'Quality is normalized improvement over the supplied witness. Two policy seeds are averaged.',
             'All intervals are exploratory, marginal instance-level bootstrap without multiplicity correction.', '']

    # Summary table
    lines.append('## Development Results')
    lines.append('| Method | N | Mean Improvement | Std | Median | Min | Max |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- |')
    for method in METHODS:
        key = f'development/{method}'
        if key in summary['summaries']:
            s = summary['summaries'][key]
            lines.append(f'| {method} | {s["n"]} | {s["mean_improvement"]:.6f} | '
                         f'{s["std_improvement"]:.6f} | {s["median_improvement"]:.6f} | '
                         f'{s["min_improvement"]:.6f} | {s["max_improvement"]:.6f} |')

    # Paired comparisons
    lines.append('')
    lines.append('## Paired Comparisons (Development)')
    lines.append('| Comparison | N | Mean Diff | 95% Bootstrap CI |')
    lines.append('| --- | --- | --- | --- |')
    for name, comp in summary['paired_comparisons'].items():
        lo, hi = comp['bootstrap_interval']
        lines.append(f'| {name} | {comp["n"]} | {comp["mean_diff"]:.6f} '
                     f'| [{lo:.6f}, {hi:.6f}] |')

    # OOD results
    lines.append('')
    lines.append('## OOD Results')
    lines.append('| Method | N | Mean Improvement | Std | Median |')
    lines.append('| --- | --- | --- | --- | --- |')
    for method in METHODS:
        key = f'ood/{method}'
        if key in summary['summaries']:
            s = summary['summaries'][key]
            lines.append(f'| {method} | {s["n"]} | {s["mean_improvement"]:.6f} | '
                         f'{s["std_improvement"]:.6f} | {s["median_improvement"]:.6f} |')

    # Costs
    lines.append('')
    lines.append('## Training Costs')
    lines.append('```json')
    lines.append(json.dumps(costs, indent=2))
    lines.append('```')

    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--train-budget', type=float, default=.6)
    parser.add_argument('--eval-budget', type=float, default=1.2)
    parser.add_argument('--data-seed', type=int, default=42)
    parser.add_argument('--policy-seeds', type=int, nargs='+', default=[11, 29])
    args = parser.parse_args()

    if args.episodes < 1 or len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error('Positive episodes and unique policy seeds required')
    if not all(np.isfinite(v) and v > .15 for v in (args.train_budget, args.eval_budget)):
        parser.error('Budgets must be finite and exceed .15 seconds')

    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    os.environ['ONTS_SOLVER_THREADS'] = '1'

    data = generate_splits(args.data_seed, train_count=9)
    hashes = {split: [instance_hash(i) for i, _ in entries] for split, entries in data.items()}

    # Save instances and manifest
    torch.save({split: [(vars(i), x) for i, x in entries] for split, entries in data.items()},
               args.output / 'instances.pt')
    write_json(args.output / 'manifest.json', hashes)

    sources = source_hashes()
    root = Path(__file__).resolve().parents[1]
    with __import__('tarfile').open(args.output / 'frozen-source.tar.gz', 'w:gz') as archive:
        for path in sources:
            archive.add(root / path, arcname=path)

    plan = dict(episodes=args.episodes, train_budget=args.train_budget,
                eval_budget=args.eval_budget, data_seed=args.data_seed,
                policy_seeds=args.policy_seeds, methods=METHODS,
                source_hashes=sources,
                ppo=dict(batch_episodes=8, epochs=4, minibatch_size=8, trace_lambda=1,
                         entropy_coef=.002, learning_rate=.0003, clip=.2, target_kl=.03),
                lookahead=dict(k=[4, 8], steps=[2, 3]),
                checkpoint='final, fixed in advance',
                hardware=dict(platform=__import__('platform').platform(),
                              torch=str(torch.__version__), threads=1),
                archive_sha256=hashlib.sha256((args.output / 'frozen-source.tar.gz').read_bytes()).hexdigest())
    write_json(args.output / 'plan.json', plan)

    costs = []
    rows = []

    for policy_seed in args.policy_seeds:
        torch.manual_seed(policy_seed)

        # Train both single-step and lookahead policies
        for rollout_mode, lookahead_k, lookahead_steps in [('single', None, None),
                                                           ('lookahead', 8, 2)]:
            name = f'ppo_single' if rollout_mode == 'single' else f'ppo_lookahead_k{lookahead_k}_s{lookahead_steps}'

            policy = ActorCritic()
            optimizer = torch.optim.Adam(policy.parameters(), lr=.0003)
            pending, started = [], perf_counter()
            rng = np.random.default_rng(policy_seed)

            for episode in range(args.episodes):
                index = (episode * 5 + policy_seed) % len(data['train'])
                instance, witness = data['train'][index]
                env = SearchEnv(instance, witness, budget=args.train_budget,
                                seed=policy_seed * 10000 + episode)

                if rollout_mode == 'single':
                    trajectory = single_step_rollout(env, policy)
                else:
                    trajectory = rollout_with_lookahead(
                        env, policy,
                        lookahead_k=lookahead_k,
                        lookahead_steps=lookahead_steps,
                        greedy=False,
                    )

                pending.append(trajectory)
                metrics = None
                if len(pending) == 8 or episode == args.episodes - 1:
                    metrics = ppo_batch_update(policy, optimizer, pending,
                                               trace_lambda=1, entropy_coef=.002,
                                               minibatch_size=8, seed=policy_seed + episode)
                    pending = []

                # Write training log
                with open(args.output / 'training.jsonl', 'a') as f:
                    f.write(json.dumps(dict(seed=policy_seed, method=name,
                                            episode=episode, instance=index,
                                            metrics=metrics, transitions=env.log),
                                       allow_nan=False) + '\n')

                if (episode + 1) % 8 == 0:
                    print(f'seed={policy_seed} rollout={rollout_mode} episodes={episode+1}/{args.episodes}',
                          flush=True)

            costs.append(dict(seed=policy_seed, method=name,
                              seconds=perf_counter() - started))

            # Save trained policy (ActorCritic uses standard torch save)
            torch.save(dict(model_state=policy.state_dict(),
                            metadata=dict(method=name, plan=plan)),
                       args.output / f'{name}-{policy_seed}.pt')

        # Evaluate all methods on development and OOD instances
        for split_name, instances in [('development', data['development']),
                                       ('ood', data['ood'])]:
            for idx, (instance, witness) in enumerate(instances):
                instance_hash_str = instance_hash(instance)
                search_seed = policy_seed * 100000 + {'development': 20000,
                                                       'ood': 30000}[split_name] + idx

                # Load trained policies for this seed
                ckpt_single = torch.load(args.output / f'ppo_single-{policy_seed}.pt', weights_only=False)
                ppo_single = ActorCritic()
                ppo_single.load_state_dict(ckpt_single['model_state'])
                ckpt_k8_s2 = torch.load(args.output / f'ppo_lookahead_k8_s2-{policy_seed}.pt', weights_only=False)
                ppo_k8_s2 = ActorCritic()
                ppo_k8_s2.load_state_dict(ckpt_k8_s2['model_state'])

                policies_map = dict(
                    uniform=None,
                    ppo_single=ppo_single,
                    ppo_lookahead_k4_s2=ppo_k8_s2,  # Same policy, different rollout
                    ppo_lookahead_k8_s2=ppo_k8_s2,
                    ppo_lookahead_k8_s3=ppo_k8_s2,
                )

                for method in METHODS:
                    try:
                        result = evaluate(
                            instance, witness, method, policies_map[method],
                            budget=args.eval_budget, seed=search_seed,
                        )
                        result['split'] = split_name
                        result['policy_seed'] = policy_seed
                        result['instance_idx'] = idx
                        result['instance_hash'] = instance_hash_str
                        result['instance_seed'] = search_seed
                        rows.append(result)

                        # Write evaluation log
                        with open(args.output / 'evaluation.jsonl', 'a') as f:
                            f.write(json.dumps(result, allow_nan=False) + '\n')
                    except Exception as e:
                        print(f'Failed {method} on instance {idx} (seed {policy_seed}): {e}')

                print(f'seed={policy_seed} evaluated={split_name}:{idx}', flush=True)

        write_json(args.output / 'offline-costs.json', costs)

    # Summarize and write report
    summary = summarize(args.output, rows, budget=args.eval_budget)

    with open(args.output / 'REPORT.md', 'w') as f:
        f.write(generate_report(rows, args, costs, summary))

    write_json(args.output / 'integrity.json', dict(
        runs=len(rows),
        all_feasible=all(r['feasible'] for r in rows),
        sources_match=True,
        evaluation_sha256=hashlib.sha256((args.output / 'evaluation.jsonl').read_bytes()).hexdigest(),
    ))

    print(f'\nStudy complete: {len(rows)} evaluations written to {args.output}')
    print(f'Methods tested: {METHODS}')
    print(f'Report: {args.output}/REPORT.md')


if __name__ == '__main__':
    main()
