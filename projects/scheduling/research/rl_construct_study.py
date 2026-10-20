"""Fresh, witness-hidden feasibility-first pilot; no held-out tuning."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..rl_construct import ConstructPolicy, construct
from ..rl_construct_graph import GraphConstructPolicy
from .datasets import instance_hash, source_hashes, witness_instance
from .oracle import oracle_check


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def supervise(model, optimizer, labels):
    if not labels:
        return None
    losses = [F.cross_entropy(model(d["graph"] if isinstance(model, GraphConstructPolicy) else torch.tensor(d["features"], dtype=torch.float32))[None], torch.tensor([d["index"]])) for d in labels]
    loss = torch.stack(losses).mean()
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()
    return float(loss.detach())


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    config = ONTSConfig()
    datasets, manifest = {}, []
    for split, count, large, offset in [("train", args.train_per_regime, False, 0), ("test", args.test_per_regime, False, 10000), ("ood", args.ood_per_regime, True, 20000)]:
        datasets[split] = []
        for r, regime in enumerate(("balanced", "energy", "periodic")):
            for n in range(count):
                instance, witness = witness_instance(seed=args.data_seed + offset + r * 1000 + n, regime=regime, large=large)
                identifier = f"{split}-{regime}-{n}"
                data = onts_instance_to_json_data(instance)
                write(output / f"{identifier}.json", data)
                # Witness is used only to establish known feasibility, then discarded.
                assert config.validate(instance, witness).feasible
                datasets[split].append((identifier, instance))
                manifest.append(dict(id=identifier, hash=instance_hash(instance), split=split, regime=regime, jobs=instance.J, horizon=instance.T, known_feasible=True))
    hashes = [r["hash"] for r in manifest]
    if len(set(hashes)) != len(hashes):
        raise ValueError("Overlapping instance content")
    # Check prior frozen manifests too, without making their instances available.
    previous_hashes = set()
    root = Path(__file__).parent / "results"
    for path in [*root.glob("**/manifest.json"), *root.glob("**/protocol.json")]:
        prior = json.loads(path.read_text())
        if isinstance(prior, dict):
            previous_hashes.update(e["hash"] for e in prior.get("entries", prior.get("manifest", [])) if isinstance(e, dict) and "hash" in e)
    if previous_hashes.intersection(hashes):
        raise ValueError("Instance already appears in a prior manifest")
    write(output / "protocol.json", dict(arguments=vars(args), source_hashes=source_hashes(),
          manifest=manifest, prior_manifest_hashes_checked=len(previous_hashes),
          primary_comparison="graph-rl minus graph-replay on on-time feasible terminal utility (1+normalized objective; failures zero), pooled held-out instances; separate success-rate and quality summaries; exploratory instance bootstrap",
          scope="Exploratory synthetic hybrid construction; witnesses unavailable at inference; final checkpoints only; exact references computed after training."))
    print("Protocol frozen; collecting training-only construction teachers", flush=True)
    labels, teacher_rows = [], []
    for index, (identifier, instance) in enumerate(datasets["train"]):
        proposals = []
        for attempt in range(args.teacher_rollouts):
            row, _, _ = construct(instance, None, mode="priority" if attempt == 0 else "uniform", seed=args.data_seed + index * 100 + attempt, budget=args.budget, record_decisions=True, record_graph="graph" in args.architectures)
            proposals.append(row)
        successes = [r for r in proposals if r["success"]]
        if successes:
            best = max(successes, key=lambda r: r["objective"])
            labels.extend(best["decisions"])
        teacher_rows.append(dict(id=identifier, rollouts=proposals))
    write(output / "teachers.json", teacher_rows)
    training, models = [], {}
    for architecture, seed in product(args.architectures, args.policy_seeds):
        torch.manual_seed(seed)
        warm = GraphConstructPolicy() if architecture == "graph" else ConstructPolicy()
        optimizer = torch.optim.Adam(warm.parameters(), lr=0.001)
        for _ in range(args.teacher_epochs):
            for start in range(0, len(labels), 8):
                supervise(warm, optimizer, labels[start:start+8])
        models[(f"{architecture}-imitation", seed)] = copy.deepcopy(warm).eval()
        rl, replay = copy.deepcopy(warm), copy.deepcopy(warm)
        rl_optimizer = torch.optim.Adam(rl.parameters(), lr=0.001)
        replay_optimizer = torch.optim.Adam(replay.parameters(), lr=0.001)
        rng = np.random.default_rng(seed)
        for episode in range(args.episodes):
            identifier, instance = datasets["train"][int(rng.integers(len(datasets["train"])))]
            # Grouped leave-one-out REINFORCE: same instance, independent rollouts.
            rows, log_groups, entropy_groups = [], [], []
            for rollout in range(args.group_size):
                row, logs, entropies = construct(instance, rl, seed=seed * 100000 + episode * args.group_size + rollout, budget=args.budget, training=True)
                rows.append(row); log_groups.append(logs); entropy_groups.append(entropies)
            rewards = np.array([r["reward"] for r in rows])
            losses = []
            for k, logs in enumerate(log_groups):
                if logs:
                    baseline = (rewards.sum() - rewards[k]) / (args.group_size - 1)
                    losses.append(-(rewards[k] - baseline) * torch.stack(logs).sum() - 0.001 * torch.stack(entropy_groups[k]).mean())
            if losses:
                rl_optimizer.zero_grad(); torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(rl.parameters(), 1.); rl_optimizer.step()
                start = (episode * 8) % max(1, len(labels))
                supervise(replay, replay_optimizer, (labels+labels)[start:start+8])
            training.append(dict(architecture=architecture, seed=seed, episode=episode, id=identifier, rollouts=rows, reward_std=float(rewards.std())))
            if (episode + 1) % 16 == 0:
                print(f"{architecture} seed {seed}: {episode+1}/{args.episodes} episodes", flush=True)
        models[(f"{architecture}-rl", seed)] = rl.eval()
        models[(f"{architecture}-replay", seed)] = replay.eval()
    write(output / "training.json", training)
    checkpoint_hashes = {}
    for (name, seed), model in models.items():
        path = output / f"{name}-{seed}.pt"
        torch.save(model.state_dict(), path)
        checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / "checkpoints.json", checkpoint_hashes)
    print("Final checkpoints frozen; evaluating untouched instances", flush=True)
    references, evaluations = [], []
    methods = ["uniform", "priority", *[f"{architecture}-{arm}" for architecture in args.architectures for arm in ("imitation", "replay", "rl")]]
    for split in ("test", "ood"):
        for identifier, instance in datasets[split]:
            for seed in args.policy_seeds:
                for method in methods:
                    row, _, _ = construct(instance, models.get((method, seed)), mode=method if method in {"uniform", "priority"} else "sample", seed=seed, budget=args.budget)
                    valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], config)
                    if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-5):
                        raise AssertionError("Independent oracle disagreement")
                    evaluations.append(dict(id=identifier, split=split, seed=seed, method=method, **row))
            result = ONTSMilp(instance, config).solve(time_limit=args.reference_budget, solver_threads=1)
            references.append(dict(id=identifier, status=result.status, objective=result.objective,
                upper_bound=result.upper_bound, runtime=result.runtime,
                schedule=result.schedule.int().tolist() if result.schedule is not None else None))
            print(f"evaluated {identifier}; reference {result.status}", flush=True)
    by_id = {r["id"]: r for r in references}
    for row in evaluations:
        reference = by_id[row["id"]]
        optimum = reference["objective"] if reference["status"] == "optimal" else None
        row["certified_gap"] = (max(0., optimum-row["objective"]) / max(1., abs(optimum))) if optimum is not None and row["success"] else None
        row["optimal"] = row["certified_gap"] is not None and row["certified_gap"] <= 1e-6
        row["on_time_utility"] = row["reward"] if row["success"] else 0.
    write(output / "references.json", references)
    write(output / "evaluations.json", evaluations)
    contrasts = []
    for first, second in [(f"{arch}-rl", f"{arch}-replay") for arch in args.architectures] + ([("graph-rl", "mlp-rl")] if set(args.architectures) == {"mlp", "graph"} else []):
        for split in ("all", "test", "ood"):
            ids = sorted({r["id"] for r in evaluations if split == "all" or r["split"] == split})
            differences = []
            for identifier in ids:
                means = [np.mean([r["on_time_utility"] for r in evaluations if r["id"] == identifier and r["method"] == method]) for method in (first, second)]
                differences.append(means[0]-means[1])
            difference = np.array(differences)
            rng = np.random.default_rng(20261011)
            sampled = rng.choice(difference, size=(4000, len(difference)), replace=True).mean(1)
            contrasts.append(dict(first=first, second=second, split=split, instances=len(ids), mean=float(difference.mean()), interval=np.quantile(sampled, [.025, .975]).tolist()))
    write(output / "contrasts.json", contrasts)
    lines = ["# Cold-start construction pilot", "", "Exploratory; planted instances, no witness at inference. Success means independently validated feasibility within budget. Feasibility is measured over all attempts. Gaps are conditional on success and certified references.", "", "| Split | Method | Success | Certified optimal | Mean certified gap |", "|---|---|---:|---:|---:|"]
    for split in ("test", "ood"):
        for method in methods:
            rows = [r for r in evaluations if r["split"] == split and r["method"] == method]
            gaps = [r["certified_gap"] for r in rows if r["certified_gap"] is not None]
            lines.append(f"| {split} | {method} | {sum(r['success'] for r in rows)}/{len(rows)} | {sum(r['optimal'] for r in rows)}/{len(rows)} | {np.mean(gaps):.4f} ({len(gaps)} pairs) |" if gaps else f"| {split} | {method} | {sum(r['success'] for r in rows)}/{len(rows)} | 0/{len(rows)} | unavailable |")
    lines += ["", f"Teacher decision labels: {len(labels)}. Evaluation overruns: {sum(r['overrun'] > 0 for r in evaluations)}.", "", "RL uses grouped leave-one-out REINFORCE after a shared imitation warm start. Replay receives one teacher update per RL update; optimizer steps are matched, training runtime is not. A restricted three-profile menu may dead-end even when a completion exists. Larger instances test unseen size within the same planted family; this does not establish generator-family transfer or a guarantee."]
    lines += ["", "Paired on-time terminal utility differences (1+normalized objective for success, zero for failure). Policy seeds are averaged within each instance. Intervals are exploratory marginal instance-bootstrap intervals; no multiplicity correction:", ""]
    for contrast in contrasts:
        lines.append(f"- {contrast['split']}: {contrast['first']} minus {contrast['second']}: {contrast['mean']:.6f}, interval [{contrast['interval'][0]:.6f}, {contrast['interval'][1]:.6f}], {contrast['instances']} instances.")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--architectures", nargs="+", choices=["mlp", "graph"], default=["mlp"])
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--policy-seeds", nargs="+", type=int, default=[11, 29])
    parser.add_argument("--data-seed", type=int, default=202610110)
    parser.add_argument("--train-per-regime", type=int, default=3)
    parser.add_argument("--test-per-regime", type=int, default=2)
    parser.add_argument("--ood-per-regime", type=int, default=1)
    parser.add_argument("--teacher-rollouts", type=int, default=4)
    parser.add_argument("--teacher-epochs", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--budget", type=float, default=2.)
    parser.add_argument("--reference-budget", type=float, default=5.)
    args = parser.parse_args()
    if args.group_size < 2 or any(getattr(args, key) < 1 for key in ["episodes", "train_per_regime", "test_per_regime", "ood_per_regime", "teacher_rollouts", "teacher_epochs"]):
        parser.error("Counts must be positive and group-size at least two")
    run(args)


if __name__ == "__main__":
    main()
