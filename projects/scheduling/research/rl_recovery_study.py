"""Fresh tighter-resource cold-start study of completion-aware recovery."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..rl_construct import construct
from ..rl_recovery import RecoveryPolicy, recover_construct
from .datasets import instance_hash, source_hashes, witness_instance
from .oracle import oracle_check


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def tight_instance(seed, regime, large, config):
    """Known-feasible, tighter family; witness is private generator evidence."""
    instance, witness = witness_instance(seed=seed, regime=regime, large=large)
    rng = np.random.default_rng(seed + 701)
    usage = instance.uso_p.double() @ witness.double()
    solar = usage + torch.tensor(rng.uniform(.15, 1.2, instance.T))
    first, last = [], []
    for row in witness.numpy():
        active = np.flatnonzero(row)
        first.append(max(0, int(active[0])-int(rng.integers(1, 5))))
        last.append(min(instance.T, int(active[-1])+1+int(rng.integers(1, 5))))
    instance = replace(instance, recurso_p=solar,
        min_statup=torch.full((instance.J,), 2), max_statup=torch.full((instance.J,), 3),
        min_cpu_time=torch.full((instance.J,), 2), max_cpu_time=torch.tensor(rng.integers(3, 7, instance.J)),
        win_min=torch.tensor(first), win_max=torch.tensor(last))
    if not config.validate(instance, witness).feasible:
        raise ValueError("Tight generator witness invalid")
    return instance


def supervise(model, optimizer, labels):
    if not labels:
        return
    loss = torch.stack([F.cross_entropy(model(torch.tensor(d["features"], dtype=torch.float32))[None], torch.tensor([d["index"]])) for d in labels]).mean()
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


def contrasts(rows):
    result = []
    for first, second in [("completion-rl", "completion-replay"), ("completion-uniform", "backtrack-uniform"), ("completion-uniform", "onepass-uniform")]:
        for split in ("all", "test", "ood"):
            ids = sorted({r["id"] for r in rows if split == "all" or r["split"] == split})
            delta = np.array([np.mean([r["utility"] for r in rows if r["id"] == identifier and r["method"] == first])-np.mean([r["utility"] for r in rows if r["id"] == identifier and r["method"] == second]) for identifier in ids])
            samples = np.random.default_rng(1106).choice(delta, size=(4000, len(delta)), replace=True).mean(1)
            result.append(dict(first=first, second=second, split=split, instances=len(ids), difference=float(delta.mean()), interval=np.quantile(samples, [.025, .975]).tolist()))
    return result


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = ONTSConfig(q=.2, bat_usage=.3, limite_inferior=.1)
    data, manifest = {}, []
    for split, count, large, offset in [("train", args.train_per_regime, False, 0), ("test", args.test_per_regime, False, 10000), ("ood", args.ood_per_regime, True, 20000)]:
        data[split] = []
        for index, regime in enumerate(("balanced", "energy", "periodic")):
            for n in range(count):
                instance = tight_instance(args.data_seed+offset+index*1000+n, regime, large, config)
                identifier = f"{split}-{regime}-{n}"
                data[split].append((identifier, instance))
                write(output / f"{identifier}.json", onts_instance_to_json_data(instance))
                manifest.append(dict(id=identifier, hash=instance_hash(instance), split=split, regime=regime, jobs=instance.J, horizon=instance.T, known_feasible=True))
    hashes = {r["hash"] for r in manifest}
    if len(hashes) != len(manifest):
        raise ValueError("Duplicate instances")
    previous = set()
    for path in [*Path(args.history_root).glob("**/manifest.json"), *Path(args.history_root).glob("**/protocol.json")]:
        prior = json.loads(path.read_text())
        if isinstance(prior, dict):
            previous.update(e["hash"] for e in prior.get("entries", prior.get("manifest", [])) if isinstance(e, dict) and "hash" in e)
    if previous & hashes:
        raise ValueError("Overlap with previous instances")
    sources = source_hashes()
    write(output / "protocol.json", dict(arguments=vars(args), config=asdict(config), manifest=manifest,
        source_hashes=sources, prior_hashes_checked=len(previous),
        generator="onts-tight-planted-v1: solar follows private witness load plus .15--1.2 W; narrower windows; two mandatory starts; reduced battery/discharge capacity",
        primary="completion-rl minus completion-replay pooled held-out utility; on-time success 1+normalized objective, failure zero; secondary recovery/completion contrasts",
        scope="Exploratory within-family size transfer; final checkpoints only; no witness or exact solver at inference; all arms share frozen NumPy profile backend"))
    root = Path(__file__).resolve().parents[1]
    for filename, digest in sources.items():
        target = output / "source" / filename; target.parent.mkdir(parents=True, exist_ok=True)
        content = (root / filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Source changed during freeze")
        target.write_bytes(content)
    print("Protocol and sources frozen; generating recovery teachers", flush=True)
    labels, teachers = [], []
    for index, (identifier, instance) in enumerate(data["train"]):
        rollouts = []
        for attempt in range(args.teacher_rollouts):
            row, _, _ = recover_construct(instance, mode="priority" if attempt == 0 else "uniform", seed=args.data_seed+index*100+attempt,
                config=config, budget=args.teacher_budget, record_decisions=True, max_decisions=args.eval_decisions)
            rollouts.append(row)
        successful = [r for r in rollouts if r["success"]]
        if successful:
            labels.extend(max(successful, key=lambda r: r["objective"])["decisions"])
        teachers.append(dict(id=identifier, rollouts=rollouts))
    write(output / "teachers.json", teachers)
    if not labels:
        raise ValueError("No successful training teacher; study cannot supply matched imitation")
    models, training = {}, []
    for seed in args.policy_seeds:
        torch.manual_seed(seed)
        model = RecoveryPolicy(); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        for epoch in range(args.teacher_epochs):
            for start in range(0, len(labels), 8):
                supervise(model, optimizer, labels[start:start+8])
        models[("imitation", seed)] = copy.deepcopy(model).eval()
        rl, replay = copy.deepcopy(model), copy.deepcopy(model)
        rl_optimizer = torch.optim.Adam(rl.parameters(), lr=.001)
        replay_optimizer = torch.optim.Adam(replay.parameters(), lr=.001)
        rng = np.random.default_rng(seed)
        for episode in range(args.episodes):
            identifier, instance = data["train"][int(rng.integers(len(data["train"])))]
            group, log_groups, entropy_groups = [], [], []
            for rollout in range(4):
                row, logs, entropies = recover_construct(instance, rl, seed=seed*100000+episode*4+rollout, config=config,
                    budget=args.train_budget, training=True, max_decisions=args.train_decisions)
                group.append(row); log_groups.append(logs); entropy_groups.append(entropies)
            rewards = np.array([r["reward"] for r in group]); losses = []
            for k, logs in enumerate(log_groups):
                if logs:
                    advantage = rewards[k] - (rewards.sum()-rewards[k])/3
                    losses.append(-advantage*torch.stack(logs).sum()-.001*torch.stack(entropy_groups[k]).mean())
            if losses:
                rl_optimizer.zero_grad(); torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(rl.parameters(), 1.); rl_optimizer.step()
                start = episode*8 % len(labels)
                supervise(replay, replay_optimizer, (labels+labels)[start:start+8])
            training.append(dict(seed=seed, episode=episode, id=identifier, rollouts=group, reward_std=float(rewards.std())))
            if (episode+1) % 8 == 0:
                print(f"seed {seed}: {episode+1}/{args.episodes}", flush=True)
        models[("rl", seed)] = rl.eval(); models[("replay", seed)] = replay.eval()
    write(output / "training.json", training)
    checkpoint_hashes = {}
    for (name, seed), model in models.items():
        path = output / f"{name}-{seed}.pt"; torch.save(model.state_dict(), path)
        checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / "checkpoints.json", checkpoint_hashes)
    print("Final checkpoints frozen; evaluating untouched instances", flush=True)
    evaluations, references = [], []
    methods = ["onepass-uniform", "backtrack-uniform", "completion-uniform", "completion-imitation", "completion-replay", "completion-rl"]
    for split in ("test", "ood"):
        for identifier, instance in data[split]:
            order_rng = np.random.default_rng(args.data_seed+len(evaluations))
            for seed in args.policy_seeds:
                for method in order_rng.permutation(methods).tolist():
                    if method == "onepass-uniform":
                        row, _, _ = construct(instance, None, mode="uniform", seed=seed, budget=args.eval_budget, config=config)
                    else:
                        arm = method.removeprefix("completion-")
                        row, _, _ = recover_construct(instance, models.get((arm, seed)), mode="uniform" if "uniform" in method else "sample", seed=seed,
                            budget=args.eval_budget, config=config, lookahead=method != "backtrack-uniform", max_decisions=args.eval_decisions)
                    valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], config)
                    if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-4):
                        raise ValueError("Independent oracle disagreement")
                    evaluations.append(dict(id=identifier, split=split, seed=seed, method=method, utility=row["reward"] if row["success"] else 0., **row))
            reference = ONTSMilp(instance, config).solve(time_limit=args.reference_budget, solver_threads=1)
            references.append(dict(id=identifier, status=reference.status, objective=reference.objective, upper_bound=reference.upper_bound,
                runtime=reference.runtime, schedule=reference.schedule.int().tolist() if reference.schedule is not None else None))
            print(f"evaluated {identifier}; reference {reference.status}", flush=True)
    write(output / "evaluations.json", evaluations); write(output / "references.json", references)
    paired = contrasts(evaluations); write(output / "contrasts.json", paired)
    lines = ["# Completion-aware recovery pilot", "", "Exploratory, fresh tighter planted family. No witness at inference. Feasibility includes all attempts; utility is zero on failure. Policy seeds are averaged within each instance for paired intervals.", "", "| Split | Method | On-time success | Mean utility | Mean backtracks | Mean rejected |", "|---|---|---:|---:|---:|---:|"]
    for split in ("test", "ood"):
        for method in methods:
            selected = [r for r in evaluations if r["method"] == method and r["split"] == split]
            lines.append(f"| {split} | {method} | {sum(r['success'] for r in selected)}/{len(selected)} | {np.mean([r['utility'] for r in selected]):.6f} | {np.mean([r.get('backtracks',0) for r in selected]):.2f} | {np.mean([r.get('rejected',0) for r in selected]):.2f} |")
    lines += ["", f"Teacher labels: {len(labels)}. Evaluation overruns: {sum(r['overrun'] > 0 for r in evaluations)}. Certified references: {sum(r['status']=='optimal' for r in references)}/{len(references)}.", "", "Paired utility differences; exploratory marginal intervals without multiplicity correction:", ""]
    for c in paired:
        lines.append(f"- {c['split']}: {c['first']} minus {c['second']}: {c['difference']:.6f}, interval [{c['interval'][0]:.6f}, {c['interval'][1]:.6f}], {c['instances']} instances.")
    lines += ["", "Completion probes provide necessary conditions, not a guarantee. Backtracking explores the restricted DP profile menu. All comparisons use this new study's instances and budget; preceding graph results are not causal controls. RL/replay share initialization and optimizer-step counts, not training data volume or runtime."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--backend-snapshot", required=True)
    parser.add_argument("--data-seed", type=int, default=2026101200)
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[11,29,47])
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--train-per-regime", type=int, default=3)
    parser.add_argument("--test-per-regime", type=int, default=2)
    parser.add_argument("--ood-per-regime", type=int, default=1)
    parser.add_argument("--teacher-rollouts", type=int, default=3)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--teacher-budget", type=float, default=.7)
    parser.add_argument("--train-budget", type=float, default=.35)
    parser.add_argument("--eval-budget", type=float, default=2.)
    parser.add_argument("--reference-budget", type=float, default=5.)
    parser.add_argument("--train-decisions", type=int, default=64)
    parser.add_argument("--eval-decisions", type=int, default=256)
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and name != "data_seed" and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
