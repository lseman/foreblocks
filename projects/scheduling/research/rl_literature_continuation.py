"""Longer, predeclared continuation of the MIStar and DR-ALNS ONTS adaptations.

Final-checkpoint comparisons use fresh synthetic data, not checkpoint selection.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import hashlib
import json
import math
import platform
import shutil
import tarfile
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from ..rl_alns import AdaptiveLNS, DRALNSActorCritic, LiteratureSearchEnv, literature_rollout
from ..rl_mistar import MistarActorCritic
from ..rl_learning import ppo_batch_update
from .datasets import instance_hash, source_hashes, witness_instance
from .rl_literature_study import bootstrap, teacher, write
from .rl_portfolio_study import evaluate as evaluate_companion

MODELS = {'mistar': MistarActorCritic, 'dralns': DRALNSActorCritic}
METHODS = ['profile_heuristic', 'alns', 'uniform_anneal', 'random_mistar4',
           *[f'{name}_{phase}' for name in MODELS for phase in ('il', 'short', 'replay', 'ppo')]]


def replay_update(model, optimizer, examples, trajectories, *, epochs=3, minibatch=32, seed=0):
    """Teacher-only actor continuation with rollout-return critic supervision.

    Same nominal optimizer-step schedule as PPO, before its KL early stopping.
    MIStar's teacher is a single proposal, not an artificial repeated K-vector.
    """
    rng = np.random.default_rng(seed)
    returns = []
    for trajectory in trajectories:
        total = 0.
        for observation, mask, _, _, _, reward in reversed(trajectory):
            total += reward
            returns.append((observation, mask, total))
    if not returns:
        return dict(steps=0, updates=0)
    metrics = []
    updates = epochs * math.ceil(len(returns) / minibatch)
    for _ in range(updates):
        actor_losses, critic_losses = [], []
        for i in rng.integers(len(examples), size=min(minibatch, len(returns))):
            observation, mask, label, weight = examples[int(i)]
            dist, _ = model(observation, mask)
            likelihood = dist.base.log_prob(torch.tensor(label)) if isinstance(model, MistarActorCritic) else dist.log_prob(torch.tensor(label))
            actor_losses.append(-weight * likelihood)
        for i in rng.integers(len(returns), size=min(minibatch, len(returns))):
            observation, mask, target = returns[int(i)]
            _, value = model(observation, mask)
            critic_losses.append((value - target).square())
        loss = torch.stack(actor_losses).mean() + .5 * torch.stack(critic_losses).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite replay loss')
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .5, error_if_nonfinite=True)
        optimizer.step(); metrics.append(float(loss.detach()))
    return dict(steps=len(returns), updates=updates, loss=float(np.mean(metrics)))


def evaluate(method, model, instance, initial, seed, budget, repair_budget):
    if method == 'profile_heuristic':
        row = evaluate_companion(instance, initial, method, None, budget=budget, seed=seed)
        row['baseline_history'] = row.pop('transitions')
        row.update(transitions=[], feasible_decreases=0, status='companion_profile_portfolio')
        return row
    tick = perf_counter(); torch.manual_seed(seed)
    env = LiteratureSearchEnv(instance, initial, budget=budget, seed=seed, repair_budget=repair_budget)
    env.deadline = min(env.deadline, tick + budget - env.reserve)
    cutoff = env.deadline - env.started
    if method in ('alns', 'uniform_anneal'):
        controller = AdaptiveLNS()
        while not env.done:
            mask = env.action_mask()
            if not mask.any(): break
            action, key = controller.select(env, mask, adaptive=method == 'alns')
            if env.done: break
            env.step_actions(action)
            if method == 'alns': controller.update(key, env.log[-1])
    elif method == 'random_mistar4':
        while not env.done:
            indices = torch.where(env.action_mask())[0].numpy()
            if not len(indices): break
            proposals = env.selection_rng.choice(indices, 4, replace=True)
            if env.done: break
            env.step_actions(proposals)
    else:
        literature_rollout(env, model, collect=False)
    report = env.problem.config.validate(instance, env.best)
    runtime = perf_counter() - tick
    assert report.feasible and report.objective >= env.initial_objective
    return dict(initial=env.initial_objective, objective=report.objective,
        improvement=(report.objective-env.initial_objective)/env.scale, runtime=runtime,
        overrun=max(0., runtime-budget), feasible=True, status=None,
        acceptance_cutoff=cutoff, setup_runtime=env.started-tick,
        feasible_decreases=sum(r['accepted'] and r['current_objective'] < r['before_current']-1e-7 for r in env.log),
        schedule=env.best.int().tolist(), transitions=env.log)


def summarize(rows):
    means, pairs = [], []
    for split in ('development', 'ood'):
        for method in METHODS:
            selected = [r for r in rows if r['split'] == split and r['method'] == method]
            means.append(dict(split=split, method=method, improvement=float(np.mean([r['improvement'] for r in selected])),
                runtime=float(np.mean([r['runtime'] for r in selected])),
                overruns=sum(r['overrun'] > 0 for r in selected),
                feasible_decreases=sum(r['feasible_decreases'] for r in selected)))
        contrasts = [('alns', 'uniform_anneal'), ('mistar_ppo', 'random_mistar4')]
        contrasts += [(f'{name}_ppo', f'{name}_{phase}') for name in MODELS for phase in ('short', 'replay', 'il')]
        contrasts += [(f'{name}_ppo', 'profile_heuristic') for name in MODELS]
        for a, b in contrasts:
            values = []
            for i in sorted({r['instance'] for r in rows if r['split'] == split}):
                quality = lambda method: np.mean([r['improvement'] for r in rows if r['split'] == split and r['instance'] == i and r['method'] == method])
                values.append(quality(a)-quality(b))
            pairs.append(dict(split=split, method=a, baseline=b, delta=bootstrap(values)))
    return dict(means=means, paired=pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('projects/scheduling/research/results/rl-literature-20261006'))
    parser.add_argument('--episodes', type=int, default=256)
    parser.add_argument('--train-budget', type=float, default=.8)
    parser.add_argument('--budget', type=float, default=1.2)
    args = parser.parse_args()
    if args.episodes < 8 or args.episodes % 4 or args.train_budget <= .05 or args.budget <= .05:
        raise ValueError('Episodes must be a multiple of four >=8; budgets must exceed 0.05 s')
    output = args.output; output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    seeds = [11, 29]; data_seed = 2026109700; data, manifest = {}, {}
    regimes = ('balanced', 'energy', 'periodic')
    for split, count, offset in [('train', 36, 0), ('development', 6, 100), ('ood', 12, 200)]:
        data[split] = [witness_instance(seed=data_seed+offset+i, regime=regimes[i%3],
            large=split == 'ood' or (split == 'train' and i >= 24)) for i in range(count)]
        manifest[split] = [dict(index=i, hash=instance_hash(inst), J=inst.J, T=inst.T) for i, (inst, _) in enumerate(data[split])]
    hashes = [r['hash'] for entries in manifest.values() for r in entries]
    assert len(hashes) == len(set(hashes))
    for old in output.parent.glob('rl-*/manifest.json'):
        assert not any(h in old.read_text() for h in hashes), old
    write(output/'manifest.json', manifest)
    torch.save({split: [(inst.__dict__, x) for inst, x in entries] for split, entries in data.items()}, output/'instances.pt')
    root = Path(__file__).resolve().parents[1]; sources = source_hashes()
    with tarfile.open(output/'source.tar.gz', 'w:gz') as archive:
        for path in sources: archive.add(root/path, arcname=path)
    input_hashes = {}
    for seed in seeds:
        for name in MODELS:
            for phase in ('il', 'ppo'):
                path = args.baseline/f'{name}-{phase}-{seed}.pt'
                input_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    plan = dict(seeds=seeds, episodes=args.episodes, methods=METHODS, budget=args.budget,
        train_budget=args.train_budget, max_train_steps=16, repair_budget=.05,
        data_seed=data_seed, train_small=24, train_large=12, development_count=6, ood_count=12,
        baseline=str(args.baseline.resolve()), input_checkpoint_hashes=input_hashes,
        continuation='PPO and teacher-actor replay; both start from the same 32-episode PPO checkpoint; optimizer reset',
        selection=f'Final {args.episodes}-episode checkpoint only; intermediate snapshots not evaluated or selected',
        ppo_epochs=3, minibatch=32, trace_lambda=1., entropy_coef=.001,
        replay='Teacher actor plus on-policy return critic; same nominal optimizer-step count as PPO before KL stopping',
        learning_rate=.0001, teacher_candidates_per_operator=6, teacher_topk=3,
        source_hashes=sources, source_archive_sha256=hashlib.sha256((output/'source.tar.gz').read_bytes()).hexdigest(),
        status='exploratory continuation on fresh planted synthetic data; no original-paper reproduction')
    write(output/'plan.json', plan)
    write(output/'environment.json', dict(python=platform.python_version(), torch=torch.__version__,
        numpy=np.__version__, threads=1, processor=platform.processor()))
    models = {}; checkpoint_names = []; resume_names = []
    for seed in seeds:
        tick = perf_counter(); examples, teacher_records = teacher(data['train'], seed, .05)
        write(output/f'teacher-{seed}.json', teacher_records)
        teacher_runtime = perf_counter()-tick
        print(f'teacher seed {seed}: {len(teacher_records)} instances in {teacher_runtime:.1f}s', flush=True)
        for name, cls in MODELS.items():
            assert examples[name]
            for phase, original in [('il', 'il'), ('short', 'ppo')]:
                target = output/f'{name}-{phase}-{seed}.pt'
                shutil.copyfile(args.baseline/f'{name}-{original}-{seed}.pt', target)
                models[(f'{name}_{phase}', seed)] = cls.load(target)
                checkpoint_names.append(target.name)
            for arm in ('ppo', 'replay'):
                model = cls.load(args.baseline/f'{name}-ppo-{seed}.pt').train()
                optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
                rng = np.random.default_rng(seed+1819); batch = []; learning = []
                tick = perf_counter()
                with gzip.open(output/f'training-{name}-{arm}-{seed}.jsonl.gz', 'wt') as stream:
                    for episode in range(args.episodes):
                        i = int(rng.integers(len(data['train']))); inst, initial = data['train'][i]
                        action_seed = seed*10000+1000+episode; torch.manual_seed(action_seed)
                        env = LiteratureSearchEnv(inst, initial, budget=args.train_budget, seed=action_seed, repair_budget=.05)
                        trajectory = literature_rollout(env, model, max_steps=16)
                        batch.append(trajectory)
                        record = dict(episode=episode, instance=i, objective=env.best_objective,
                            acceptance_cutoff=env.deadline-env.started, schedule=env.best.int().tolist(), transitions=env.log)
                        stream.write(json.dumps(record, allow_nan=False)+'\n')
                        if len(batch) == 4:
                            if arm == 'ppo':
                                metrics = ppo_batch_update(model, optimizer, batch, epochs=3, minibatch_size=32,
                                    trace_lambda=1., entropy_coef=.001, seed=seed+episode)
                            else:
                                metrics = replay_update(model, optimizer, examples[name], batch, seed=seed+episode)
                            entry = dict(update=episode, metrics=metrics)
                            stream.write(json.dumps(entry, allow_nan=False)+'\n'); learning.append(entry); batch=[]
                        if (episode+1) % 64 == 0:
                            snapshot = output/f'progress-{name}-{arm}-{seed}-{episode+1}.pt'
                            model.save(snapshot, metadata=dict(seed=seed, episodes=episode+1, arm=arm))
                            print(f'{name} {arm} seed {seed}: {episode+1}/{args.episodes}, elapsed {perf_counter()-tick:.1f}s', flush=True)
                target = output/f'{name}-{arm}-{seed}.pt'
                model.save(target, metadata=dict(seed=seed, episodes=args.episodes, prior_episodes=32,
                    labels=len(examples[name]), arm=arm, optimizer_reset=True))
                models[(f'{name}_{arm}', seed)] = model.eval(); checkpoint_names.append(target.name)
                resume = output/f'optimizer-{name}-{arm}-{seed}.pt'
                torch.save(dict(optimizer_state=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                    episode=args.episodes, data_rng=rng.bit_generator.state), resume)
                resume_names.append(resume.name)
                write(output/f'training-{name}-{arm}-{seed}.json', dict(episodes=args.episodes,
                    teacher_runtime=teacher_runtime, labels=len(examples[name]), online_runtime=perf_counter()-tick,
                    parameters=sum(p.numel() for p in model.parameters()), updates=learning))
    write(output/'checkpoint-hashes.json', {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in checkpoint_names})
    write(output/'optimizer-hashes.json', {name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in resume_names})
    rows = []
    with gzip.open(output/'evaluations.jsonl.gz', 'wt') as stream:
        for split in ('development', 'ood'):
            for i, (inst, initial) in enumerate(data[split]):
                for seed in seeds:
                    for method in np.random.default_rng(9822+seed+i+(10000 if split=='ood' else 0)).permutation(METHODS):
                        row = evaluate(str(method), models.get((str(method),seed)), inst, initial, seed, args.budget, .05)
                        row.update(split=split, instance=i, seed=seed, method=str(method))
                        rows.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush()
                print(f'evaluated {split} {i+1}/{len(data[split])}', flush=True)
    summary = summarize(rows); write(output/'summary.json', summary)
    tex = [r'\begin{table}[t]', r'\centering', r'\caption{Longer continuation on fresh synthetic instances at '+str(args.budget)+r' s: six development and twelve larger instances, two seeds. Final checkpoints only.}',
        r'\label{tab:literature-continuation}', r'\begin{tabular}{lrr}', r'\toprule', r'Method & Development & Larger instances \\', r'\midrule']
    for method in METHODS:
        values = [next(r['improvement'] for r in summary['means'] if r['split']==split and r['method']==method) for split in ('development','ood')]
        tex.append(f"{method.replace('_',' ')} & {values[0]:.6f} & {values[1]:.6f}"+r' \\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (output/'summary.tex').write_text('\n'.join(tex)+'\n')
    unchanged = all(hashlib.sha256((root/path).read_bytes()).hexdigest()==h for path,h in sources.items())
    write(output/'source-integrity.json', dict(unchanged=unchanged)); assert unchanged
    print(json.dumps(summary['means'],indent=2),flush=True)


if __name__ == '__main__': main()
