"""Fresh architecture comparison under a common completion/recovery backend."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..rl_construct import construct
from ..rl_recovery import recover_construct
from ..rl_architectures import ARCHITECTURES, ArchitecturePolicy
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
    loss = torch.stack([F.cross_entropy(model(d["observation"])[None], torch.tensor([d["index"]])) for d in labels]).mean()
    optimizer.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()


def contrasts(rows):
    result = []
    pairs = [(f"{a}-rl", f"{a}-replay") for a in ARCHITECTURES]
    pairs += [(f"{a}-rl", "mlp-rl") for a in ARCHITECTURES if a != "mlp"]
    for first, second in pairs:
        for split in ("all", "test", "ood"):
            ids = sorted({r["id"] for r in rows if split == "all" or r["split"] == split})
            delta = np.array([np.mean([r["utility"] for r in rows if r["id"] == identifier and r["method"] == first])-np.mean([r["utility"] for r in rows if r["id"] == identifier and r["method"] == second]) for identifier in ids])
            samples = np.random.default_rng(1106).choice(delta, size=(4000,len(delta)),replace=True).mean(1)
            result.append(dict(first=first,second=second,split=split,instances=len(ids),difference=float(delta.mean()),interval=np.quantile(samples,[.025,.975]).tolist()))
    return result


def milp_baseline(instance, config, budget):
    started = perf_counter(); problem = ONTSMilp(instance, config)
    remaining = budget-(perf_counter()-started)-min(.05,.05*budget)
    result = problem.solve(time_limit=remaining,solver_threads=1) if remaining > 0 else None
    schedule = result.schedule.int().tolist() if result is not None and result.schedule is not None else [[0]*instance.T for _ in range(instance.J)]
    feasible,objective,_ = oracle_check(onts_instance_to_json_data(instance),schedule,config)
    runtime = perf_counter()-started
    success = feasible and runtime <= budget
    return dict(schedule=schedule,feasible=feasible,objective=objective if feasible else None,success=success,runtime=runtime,
        overrun=max(0.,runtime-budget),utility=1+objective/max(1.,float(instance.priority.sum())*instance.T) if success else 0.,
        status=result.status if result else "setup_limit",upper_bound=result.upper_bound if result else None,
        time_to_first_feasible=None,first_feasible_observed=False)


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
        source_hashes=sources, prior_hashes_checked=len(previous), architectures=list(ARCHITECTURES),
        parameters={a:sum(p.numel() for p in ArchitecturePolicy(a,args.width).parameters()) for a in ARCHITECTURES},
        generator="onts-tight-planted-v1: solar follows private witness load plus .15--1.2 W; narrower windows; two mandatory starts; reduced battery/discharge capacity",
        primary="Within each architecture RL minus replay pooled held-out on-time utility. Secondary architecture and MILP rankings; exploratory intervals, no multiplicity correction.",
        scope="Four ONTS architecture recipes, identical raw observations, candidate decoder inputs, data and backend. Width matched, not parameter count or training time. Recurrent pointer is task-order-sensitive. Task-time GNN is not MILP variable-constraint branching. Final checkpoints only; fresh planted-family size transfer, no SOTA claim."))
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
                config=config, budget=args.teacher_budget, record_decisions=True, record_observation=True, max_decisions=args.eval_decisions)
            rollouts.append(row)
        successful = [r for r in rollouts if r["success"]]
        if successful:
            labels.extend(max(successful, key=lambda r: r["objective"])["decisions"])
        teachers.append(dict(id=identifier, rollouts=rollouts))
    write(output / "teachers.json", teachers)
    if not labels:
        raise ValueError("No successful training teacher; study cannot supply matched imitation")
    models, training = {}, []
    training_costs = []
    for architecture in ARCHITECTURES:
        for seed in args.policy_seeds:
            training_started = perf_counter()
            torch.manual_seed(seed)
            model = ArchitecturePolicy(architecture,args.width); optimizer = torch.optim.Adam(model.parameters(), lr=.001)
            for epoch in range(args.teacher_epochs):
                for start in range(0, len(labels), 8):
                    supervise(model, optimizer, labels[start:start+8])
            models[(f"{architecture}-imitation", seed)] = copy.deepcopy(model).eval()
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
                training.append(dict(architecture=architecture, seed=seed, episode=episode, id=identifier, rollouts=group, reward_std=float(rewards.std())))
                if (episode+1) % 8 == 0:
                    print(f"{architecture} seed {seed}: {episode+1}/{args.episodes}", flush=True)
            training_costs.append(dict(architecture=architecture,seed=seed,runtime=perf_counter()-training_started,updates=args.episodes,parameters=sum(p.numel() for p in model.parameters())))
            models[(f"{architecture}-rl", seed)] = rl.eval(); models[(f"{architecture}-replay", seed)] = replay.eval()
    write(output / "training_costs.json",training_costs)
    write(output / "training.json", training)
    checkpoint_hashes = {}
    for (name, seed), model in models.items():
        path = output / f"{name}-{seed}.pt"; torch.save(model.state_dict(), path)
        checkpoint_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write(output / "checkpoints.json", checkpoint_hashes)
    print("Final checkpoints frozen; evaluating untouched instances", flush=True)
    evaluations, references = [], []
    methods = ["completion-uniform"]+[f"{a}-{arm}" for a in ARCHITECTURES for arm in ("imitation","replay","rl")]
    solver_rows = []
    for split in ("test", "ood"):
        for identifier, instance in data[split]:
            order_rng = np.random.default_rng(args.data_seed+len(evaluations))
            for seed in args.policy_seeds:
                for method in order_rng.permutation(methods).tolist():
                    row, _, _ = recover_construct(instance,models.get((method,seed)),mode="uniform" if method == "completion-uniform" else "sample",seed=seed,
                        budget=args.eval_budget,config=config,max_decisions=args.eval_decisions)
                    valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], config)
                    if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-4):
                        raise ValueError("Independent oracle disagreement")
                    evaluations.append(dict(id=identifier, split=split, seed=seed, method=method, utility=row["reward"] if row["success"] else 0., **row))
            solver_rows.append(dict(id=identifier,split=split,method="milp",**milp_baseline(instance,config,args.eval_budget)))
            reference = ONTSMilp(instance, config).solve(time_limit=args.reference_budget, solver_threads=1)
            references.append(dict(id=identifier, status=reference.status, objective=reference.objective, upper_bound=reference.upper_bound,
                runtime=reference.runtime, schedule=reference.schedule.int().tolist() if reference.schedule is not None else None))
            print(f"evaluated {identifier}; reference {reference.status}", flush=True)
    write(output / "solver_evaluations.json",solver_rows)
    write(output / "evaluations.json", evaluations); write(output / "references.json", references)
    paired = contrasts(evaluations); write(output / "contrasts.json", paired)
    lines = ["# Architecture comparison pilot", "", "Exploratory, fresh tighter planted family. No witness at inference. Feasibility includes all attempts; utility is zero on failure. Policy seeds are averaged within each instance for paired intervals.", "", "| Split | Method | On-time success | Mean utility | Mean runtime | Optimal attempts |", "|---|---|---:|---:|---:|---:|"]
    for split in ("all", "test", "ood"):
        for method in methods+["milp"]:
            selected = [r for r in evaluations+solver_rows if r["method"] == method and (split == "all" or r["split"] == split)]
            lines.append(f"| {split} | {method} | {sum(r['success'] for r in selected)}/{len(selected)} | {np.mean([r['utility'] for r in selected]):.6f} | {np.mean([r['runtime'] for r in selected]):.3f} | {sum(r['success'] and next(ref for ref in references if ref['id']==r['id'])['status']=='optimal' and abs(r['objective']-next(ref for ref in references if ref['id']==r['id'])['objective'])<1e-4 for r in selected)}/{len(selected)} |")
    lines += ["", f"Teacher labels: {len(labels)}. Evaluation overruns: {sum(r['overrun'] > 0 for r in evaluations)}. Certified references: {sum(r['status']=='optimal' for r in references)}/{len(references)}.", "", "Paired utility differences; exploratory marginal intervals without multiplicity correction:", ""]
    for c in paired:
        lines.append(f"- {c['split']}: {c['first']} minus {c['second']}: {c['difference']:.6f}, interval [{c['interval'][0]:.6f}, {c['interval'][1]:.6f}], {c['instances']} instances.")
    lines += ["", "MILP timing includes model assembly, solving and independent output validation; cold start, no supplied witness. MILP is deterministic and evaluated once per instance, neural arms once per policy seed. Completion probes are necessary conditions. Identical raw observations and recovery menu; width and optimizer update counts matched, not parameter counts/training runtime. All architectures use the same shared teacher snapshots. These are architecture adaptations, not published-method replications. No multiplicity correction and too few instances to establish a winner. RL advantage requires positive RL-minus-replay evidence, not architecture ranking."]
    lines += ["", "Model parameters:", ""]
    for architecture in ARCHITECTURES:
        lines.append(f"- {architecture}: {sum(p.numel() for p in ArchitecturePolicy(architecture,args.width).parameters())} parameters.")
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--backend-snapshot", required=True)
    parser.add_argument("--data-seed", type=int, default=2026101500)
    parser.add_argument("--policy-seeds", type=int, nargs="+", default=[11,29,47])
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--train-per-regime", type=int, default=3)
    parser.add_argument("--test-per-regime", type=int, default=2)
    parser.add_argument("--ood-per-regime", type=int, default=1)
    parser.add_argument("--teacher-rollouts", type=int, default=3)
    parser.add_argument("--teacher-epochs", type=int, default=10)
    parser.add_argument("--teacher-budget", type=float, default=.7)
    parser.add_argument("--train-budget", type=float, default=.35)
    parser.add_argument("--eval-budget", type=float, default=2.)
    parser.add_argument("--reference-budget", type=float, default=5.)
    parser.add_argument("--train-decisions", type=int, default=64)
    parser.add_argument("--eval-decisions", type=int, default=4096)
    parser.add_argument("--width",type=int,default=32)
    args = parser.parse_args()
    if args.width<4 or args.width%4 or len(set(args.policy_seeds)) != len(args.policy_seeds):
        parser.error("Width must be divisible by four; seeds must be distinct")
    for name, value in vars(args).items():
        if isinstance(value, (int, float)) and name != "data_seed" and (not np.isfinite(value) or value <= 0):
            parser.error(f"{name} must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
