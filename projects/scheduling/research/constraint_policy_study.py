"""Study of constraint-aware policies for ONTS RL.

Compares constraint-aware policies against baselines (uniform, MILP-only) on
ONTS task scheduling instances. Tests whether policies that can learn to relax
constraints strategically outperform policies that only select tasks from fixed
candidate sets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .datasets import instance_hash, source_hashes
from ..constraint_aware_training import (
    ConstraintAwareSearchEnv,
    rollout_constraint_aware,
    evaluate_policy,
    train_constraint_aware_policy,
    generate_splits,
)


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def evaluate_baseline(instance, initial_schedule, *, budget=2.0, seed=0):
    """Evaluate MILP-only baseline (no RL policy)."""
    from .milp import ONTSMilp
    from .ranking import SearchState, make_candidates
    
    env = ONTSMilp(instance)
    state = SearchState.empty(instance.J, instance.T)
    incumbent = initial_schedule.detach().cpu().bool().clone()
    report = env.config.validate(instance, incumbent)
    objective = report.objective
    scale = max(float(instance.priority.clamp(min=0).sum()) * instance.T, 1.0)
    
    started = perf_counter()
    steps = 0
    
    while started + budget > perf_counter():
        candidates = make_candidates(env, incumbent, state, np.random.default_rng(seed + steps), count=4)
        if not candidates:
            break
        
        best_gain = 0
        for i, cand in enumerate(candidates):
            result = env.solve(time_limit=0.1, incumbent=incumbent, free_mask=cand.mask, solver_threads=1)
            if result.schedule is not None:
                rep = env.config.validate(instance, result.schedule)
                if rep.feasible and rep.objective > objective:
                    gain = (rep.objective - objective) / scale
                    if gain > best_gain:
                        best_gain = gain
                        incumbent = result.schedule.detach().cpu().bool().clone()
                        objective = rep.objective
        
        if best_gain == 0:
            break
        state.advance(candidates[0].mask, incumbent.clone(), incumbent, best_gain * scale, 0.1)
        steps += 1
    
    improvement = (objective - report.objective) / max(scale, 1.0)
    return dict(method='milp_only', feasible=True,
                initial=float(report.objective), objective=float(objective),
                improvement=improvement, runtime=perf_counter() - started,
                overrun=max(0, perf_counter() - started - budget), steps=steps)


def evaluate_uniform(instance, initial_schedule, *, budget=2.0, seed=0):
    """Evaluate uniform random baseline."""
    from .milp import ONTSMilp
    from .ranking import SearchState, make_candidates
    
    env = ONTSMilp(instance)
    state = SearchState.empty(instance.J, instance.T)
    incumbent = initial_schedule.detach().cpu().bool().clone()
    report = env.config.validate(instance, incumbent)
    objective = report.objective
    scale = max(float(instance.priority.clamp(min=0).sum()) * instance.T, 1.0)
    
    started = perf_counter()
    steps = 0
    
    while started + budget > perf_counter():
        candidates = make_candidates(env, incumbent, state, np.random.default_rng(seed + steps), count=4)
        if not candidates:
            break
        
        # Try each candidate with MILP
        best_gain = 0
        for i, cand in enumerate(candidates):
            result = env.solve(time_limit=0.1, incumbent=incumbent, free_mask=cand.mask, solver_threads=1)
            if result.schedule is not None:
                rep = env.config.validate(instance, result.schedule)
                if rep.feasible and rep.objective > objective:
                    gain = (rep.objective - objective) / scale
                    if gain > best_gain:
                        best_gain = gain
                        incumbent = result.schedule.detach().cpu().bool().clone()
                        objective = rep.objective
        
        if best_gain == 0:
            break
        state.advance(candidates[0].mask, incumbent.clone(), incumbent, best_gain * scale, 0.1)
        steps += 1
    
    improvement = (objective - report.objective) / max(scale, 1.0)
    return dict(method='uniform', feasible=True,
                initial=float(report.objective), objective=float(objective),
                improvement=improvement, runtime=perf_counter() - started,
                overrun=max(0, perf_counter() - started - budget), steps=steps)


def run_study(output_dir, *, eval_budget=2.0, train_episodes=64, policy_seed=0):
    """Run the full constraint-aware policy study."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Generate instance splits
    data = generate_splits(seed=42, train_count=9)
    dev_instances = data['development']
    ood_instances = data['ood']
    
    rows = []
    
    print("=== Training Constraint-Aware Policy ===")
    policy = train_constraint_aware_policy(
        data, policy_seed, episodes=train_episodes, budget=0.8, seed=0
    )
    print(f"Training complete: {train_episodes} episodes")
    
    # Evaluate on development instances
    for seed in (0, 1):
        for split_name, instances in [('development', dev_instances), ('ood', ood_instances)]:
            for idx, (instance, witness) in enumerate(instances):
                instance_hash_str = instance_hash(instance)
                
                # Evaluate constraint-aware policy
                try:
                    result_ca = evaluate_policy(
                        instance, witness, policy, budget=eval_budget, seed=seed * 1000 + idx,
                        use_constraint_aware=True
                    )
                    result_ca['split'] = split_name
                    result_ca['instance_idx'] = idx
                    result_ca['instance_hash'] = instance_hash_str
                    result_ca['instance_seed'] = seed * 1000 + idx
                    rows.append(result_ca)
                except Exception as e:
                    print(f"Failed constraint-aware on {split_name}[{idx}]: {e}")
                
                # Evaluate MILP-only baseline
                try:
                    result_milp = evaluate_baseline(
                        instance, witness, budget=eval_budget, seed=seed * 1000 + idx
                    )
                    result_milp['split'] = split_name
                    result_milp['instance_idx'] = idx
                    result_milp['instance_hash'] = instance_hash_str
                    result_milp['instance_seed'] = seed * 1000 + idx
                    rows.append(result_milp)
                except Exception as e:
                    print(f"Failed MILP-only on {split_name}[{idx}]: {e}")
                
                # Evaluate uniform baseline
                try:
                    result_uni = evaluate_uniform(
                        instance, witness, budget=eval_budget, seed=seed * 1000 + idx
                    )
                    result_uni['split'] = split_name
                    result_uni['instance_idx'] = idx
                    result_uni['instance_hash'] = instance_hash_str
                    result_uni['instance_seed'] = seed * 1000 + idx
                    rows.append(result_uni)
                except Exception as e:
                    print(f"Failed uniform on {split_name}[{idx}]: {e}")
    
    # Summarize results
    summaries = {}
    for method in sorted(set(r['method'] for r in rows)):
        selected = [r for r in rows if r['method'] == method and r.get('feasible')]
        if not selected:
            continue
        
        improvements = np.array([r['improvement'] for r in selected])
        runtimes = np.array([r['runtime'] for r in selected])
        
        summaries[method] = dict(
            n=len(selected),
            mean_improvement=float(improvements.mean()),
            std_improvement=float(improvements.std()),
            median_improvement=float(np.median(improvements)),
            min_improvement=float(improvements.min()),
            max_improvement=float(improvements.max()),
            mean_runtime=float(runtimes.mean()),
        )
    
    result = dict(
        summaries=summaries,
        instance_hashes=[r.get('instance_hash', '') for r in rows if 'instance_hash' in r],
        source_hashes=source_hashes(),
    )
    
    write_json(output / 'summary.json', result)
    
    # Write all results
    write_json(output / 'all_results.json', {
        'rows': rows,
        'summary': result,
    })
    
    # Write report
    report_lines = [
        '# Constraint-Aware Policy Study', '',
        'Compares constraint-aware policies against baselines (uniform, MILP-only) on ONTS task scheduling.',
        '',
        '| Method | N | Mean Improvement | Std | Median | Min | Max | Mean Runtime (s) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ]
    
    for method in sorted(summaries.keys()):
        s = summaries[method]
        report_lines.append(f'| {method} | {s["n"]} | '
                          f'{s["mean_improvement"]:.6f} | {s["std_improvement"]:.6f} | '
                          f'{s["median_improvement"]:.6f} | {s["min_improvement"]:.6f} | '
                          f'{s["max_improvement"]:.6f} | {s["mean_runtime"]:.3f} |')
    
    (output / 'REPORT.md').write_text('\n'.join(report_lines) + '\n')
    
    print(f'\nStudy complete: {len(rows)} evaluations written to {output}')
    print(f'Methods tested: {sorted(summaries.keys())}')
    for method, s in sorted(summaries.items()):
        print(f'  {method}: mean_improvement={s["mean_improvement"]:.4f} ± {s["std_improvement"]:.4f} (n={s["n"]})')
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Constraint-aware policy study for ONTS RL')
    parser.add_argument('--output', type=Path, required=True, help='Output directory')
    parser.add_argument('--eval-budget', type=float, default=2.0, help='Evaluation budget (s)')
    parser.add_argument('--train-episodes', type=int, default=64, help='Number of training episodes')
    args = parser.parse_args()
    
    run_study(args.output, eval_budget=args.eval_budget, train_episodes=args.train_episodes)


if __name__ == '__main__':
    main()
