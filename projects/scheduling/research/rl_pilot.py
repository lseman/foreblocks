"""Bounded development pilot; run with python -m projects.scheduling.research.rl_pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..rl_search import ActorCritic, SearchEnv, ppo_update, rollout
from .datasets import instance_hash, source_hashes, witness_instance


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=24)
    parser.add_argument('--budget', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=202610060)
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('episodes must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy = ActorCritic()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    regimes = ('balanced', 'energy', 'periodic')
    splits = {split: [witness_instance(seed=args.seed + offset + i,
                                     regime=regimes[i % 3]) for i in range(6)]
              for split, offset in [('train', 0), ('validation', 10000), ('development', 20000)]}
    hashes = {split: [instance_hash(instance) for instance, _ in data]
              for split, data in splits.items()}
    if len(set(sum(hashes.values(), []))) != 18:
        raise RuntimeError('Instance overlap')
    plan = dict(seed=args.seed, episodes=args.episodes, budget=args.budget,
                repair_time=0.1, hashes=hashes, sources=source_hashes(),
                scope='single-satellite planted development smoke pilot',
                checkpoint_selection='final episode; validation diagnostic only')
    (args.output / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    with (args.output / 'training.jsonl').open('w') as stream:
        for episode in range(args.episodes):
            instance, initial = splits['train'][episode % 6]
            env = SearchEnv(instance, initial, budget=args.budget, seed=args.seed + episode)
            trajectory = rollout(env, policy)
            metrics = ppo_update(policy, optimizer, trajectory)
            stream.write(json.dumps(dict(episode=episode, **metrics, transitions=env.log)) + '\n')
            stream.flush()
    torch.save(dict(state_dict=policy.state_dict(), plan=plan), args.output / 'policy.pt')
    records = []
    for split in ('validation', 'development'):
        for i, (instance, initial) in enumerate(splits[split]):
            for mode in ('uniform', 'utility', 'bandit', 'ppo'):
                # Match candidate-generation seeds; trajectories may diverge.
                env = SearchEnv(instance, initial, budget=args.budget, seed=args.seed + 30000 + i)
                rollout(env, policy, mode=mode, bandit=[np.eye(18), np.zeros(18)])
                report = env.problem.config.validate(instance, env.incumbent)
                records.append(dict(split=split, instance=i, mode=mode, feasible=report.feasible,
                                    initial=env.initial_objective, objective=env.objective,
                                    improvement=(env.objective-env.initial_objective)/env.scale,
                                    transitions=env.log))
    (args.output / 'evaluation.json').write_text(json.dumps(records, indent=2) + '\n')
    lines = ['# RL development smoke pilot', '',
             'Single training seed; six planted instances per split. Common planted incumbents.',
             'This checks execution and learning plumbing, not statistical superiority.', '',
             '| Split | Selector | Feasible | Mean normalized improvement |',
             '| --- | --- | --- | --- |']
    for split in ('validation', 'development'):
        for mode in ('uniform', 'utility', 'bandit', 'ppo'):
            rows = [r for r in records if r['split'] == split and r['mode'] == mode]
            lines.append(f"| {split} | {mode} | {sum(r['feasible'] for r in rows)}/{len(rows)} | {np.mean([r['improvement'] for r in rows]):.6f} |")
    lines += ['', 'Includes model construction, candidate generation, inference and validation in the',
              'episode deadline. STOP is permitted for PPO; other selectors repair until exhausted.',
              'No full heuristic comparator, OOD evaluation, or confirmatory test is included yet.',
              'Compact state is an approximation. Candidate actions use MILP repair only.',
              'These newly seeded instances use the existing planted generator family.']
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
