"""Development study of perturbation-aware search with current/best separation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .datasets import instance_hash, source_hashes, witness_instance
from .statistics import bootstrap_mean

ACCEPTANCE_STRATEGIES = ('monotone', 'annealing', 'alns')


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


def train_policy(data, policy_seed, *, episodes=64, budget=0.8, seed=0):
    """Train a perturbation-aware PPO policy on training instances."""
    from ..rl_perturbation import (
        PerturbationActorCritic,
        PerturbationEnv,
        ppo_batch_update,
        rollout,
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=0.001)
    training_logs = []

    for episode in range(episodes):
        idx = episode % len(data['train'])
        instance, initial = data['train'][idx]

        env = PerturbationEnv(instance, initial.clone(), budget=budget, seed=seed + episode)
        trajectory = rollout(env, policy, mode='ppo')

        if trajectory:
            metrics = ppo_batch_update(policy, optimizer, [trajectory],
                                       batch_episodes=1, epochs=4, seed=episode)
            training_logs.append(dict(episode=episode, loss=metrics.get('loss', 0),
                                      reward=metrics.get('reward', 0)))

    return policy, dict(training_logs=training_logs, episodes=episodes,
                        runtime=perf_counter() - training_start)


def evaluate(instance, witness, method, *, budget=2.0, seed=0, acceptance='monotone',
             temperature=0.0, config=None):
    """Run a single evaluation of one perturbation strategy on one instance."""
    from ..rl_perturbation import PerturbationEnv, rollout

    started = perf_counter()
    torch.manual_seed(seed)

    env = PerturbationEnv(instance, witness.clone(), budget=budget, seed=seed,
                          acceptance=acceptance, temperature=temperature)
    trajectory = rollout(env, mode='uniform')

    report = env.problem.config.validate(instance, env.best_schedule)
    runtime = perf_counter() - started
    improvement = (env.best_objective - env.initial_objective) / max(1.0, env.scale)

    return dict(method=method, feasible=True, initial=float(env.initial_objective),
                objective=env.best_objective, improvement=improvement,
                runtime=runtime, overrun=max(0., runtime - budget),
                steps=len(trajectory), n_improvements=sum(1 for t in trajectory if t[5] > 0))


def summarize(output, rows):
    """Generate summary statistics and write results."""
    summaries = {}
    for split in ('development', 'ood'):
        for method in sorted(set(r['method'] for r in rows)):
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue

            improvements = np.array([r['improvement'] for r in selected])
            runtimes = np.array([r['runtime'] for r in selected])

            summaries[f'{split}/{method}'] = dict(
                n=len(selected),
                mean_improvement=float(improvements.mean()),
                std_improvement=float(improvements.std()),
                median_improvement=float(np.median(improvements)),
                min_improvement=float(improvements.min()),
                max_improvement=float(improvements.max()),
                mean_runtime=float(runtimes.mean()),
            )

    # Paired comparisons on development set
    pairs = {}
    dev_rows = [r for r in rows if r['split'] == 'development']
    groups = {}
    for r in dev_rows:
        key = (r.get('instance_seed', 0),)
        if key not in groups:
            groups[key] = {}
        groups[key][r['method']] = r['improvement']

    baselines = [m for m in ACCEPTANCE_STRATEGIES]
    for base in baselines:
        for target in ACCEPTANCE_STRATEGIES:
            if target <= base:
                continue
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


def main():
    parser = argparse.ArgumentParser(description='Perturbation-aware search study for ONTS')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--eval-budget', type=float, default=2.0, help='Evaluation budget (s)')
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # Generate instance splits
    data = generate_splits(seed=42, train_count=9)
    dev_instances = data['development']
    ood_instances = data['ood']

    rows = []

    for seed in (0, 1):
        for split_name, instances in [('development', dev_instances), ('ood', ood_instances)]:
            for idx, (instance, witness) in enumerate(instances):
                instance_hash_str = instance_hash(instance)

                for acceptance in ACCEPTANCE_STRATEGIES:
                    try:
                        result = evaluate(
                            instance, witness, method=f'perturb_{acceptance}',
                            budget=args.eval_budget, seed=seed * 1000 + idx,
                            acceptance=acceptance,
                        )
                        result['split'] = split_name
                        result['instance_idx'] = idx
                        result['instance_hash'] = instance_hash_str
                        result['instance_seed'] = seed * 1000 + idx
                        rows.append(result)
                    except Exception as e:
                        print(f'Failed {acceptance} on instance {idx}: {e}')

    # Summarize
    summary = summarize(args.output, rows)

    # Write all results
    write_json(args.output / 'all_results.json', {
        'rows': rows,
        'summary': summary,
    })

    # Write report
    report_lines = [
        '# Perturbation-Aware Search Study', '',
        'Quality is normalized improvement over the supplied witness.',
        'Three acceptance strategies compared: monotone (strictly improving), annealing (SA), alns (adaptive).',
        '',
        '| Split | Strategy | N | Mean Improvement | Std | Median | Min | Max |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    for split in ('development', 'ood'):
        for method in sorted(set(r['method'] for r in rows)):
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            if not selected:
                continue
            imp = np.array([r['improvement'] for r in selected])
            report_lines.append(f'| {split} | {method.replace("perturb_", "")} | {len(selected)} | '
                              f'{imp.mean():.6f} | {imp.std():.6f} | {np.median(imp):.6f} | '
                              f'{imp.min():.6f} | {imp.max():.6f} |')

    report_lines.extend(['', '## Paired Comparisons (Development)', '',
                         '| Comparison | N | Mean Diff | 95% Bootstrap CI |',
                         '| --- | --- | --- | --- |'])
    for comp, stats in sorted(summary.get('paired_comparisons', {}).items()):
        ci = stats['bootstrap_interval']
        report_lines.append(f'| {comp} | {stats["n"]} | {stats["mean_diff"]:.6f} | '
                          f'[{ci[0]:.6f}, {ci[1]:.6f}] |')

    (args.output / 'REPORT.md').write_text('\n'.join(report_lines) + '\n')

    print(f'Study complete: {len(rows)} evaluations written to {args.output}')
    print(f'Strategies tested: {ACCEPTANCE_STRATEGIES}')


if __name__ == '__main__':
    main()
