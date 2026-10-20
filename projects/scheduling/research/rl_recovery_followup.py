"""Fresh inference-only follow-up with memoization and a larger search cap."""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import onts_instance_to_json_data
from ..milp import ONTSConfig, ONTSMilp
from ..rl_construct import construct
from ..rl_recovery import RecoveryPolicy, recover_construct
from .datasets import instance_hash, source_hashes
from .oracle import oracle_check
from .rl_recovery_study import contrasts, tight_instance, write


def run(args):
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    source = Path(args.checkpoints).resolve()
    original = json.loads((source/"protocol.json").read_text())
    config = ONTSConfig(**original["config"])
    seeds = original["arguments"]["policy_seeds"]
    torch.set_num_threads(1)
    models, checkpoint_hashes = {}, {}
    original_hashes = json.loads((source/"checkpoints.json").read_text())
    for arm in ("imitation", "replay", "rl"):
        for seed in seeds:
            name = f"{arm}-{seed}.pt"; content = (source/name).read_bytes()
            if hashlib.sha256(content).hexdigest() != original_hashes[name]:
                raise ValueError("Checkpoint mismatch")
            (output/name).write_bytes(content); checkpoint_hashes[name] = original_hashes[name]
            model = RecoveryPolicy(); model.load_state_dict(torch.load(source/name, weights_only=True, map_location="cpu"))
            models[(arm, seed)] = model.eval()
    data, manifest = [], []
    for split, count, large, offset in [("test", args.test_per_regime, False, 0), ("ood", args.ood_per_regime, True, 10000)]:
        for r, regime in enumerate(("balanced", "energy", "periodic")):
            for index in range(count):
                instance = tight_instance(args.data_seed+offset+r*1000+index, regime, large, config)
                identifier = f"{split}-{regime}-{index}"
                write(output/f"{identifier}.json", onts_instance_to_json_data(instance))
                data.append((identifier, split, instance))
                manifest.append(dict(id=identifier, split=split, hash=instance_hash(instance), regime=regime, jobs=instance.J, horizon=instance.T))
    hashes = {r["hash"] for r in manifest}; previous = set()
    for path in [*Path(args.history_root).glob("**/manifest.json"), *Path(args.history_root).glob("**/protocol.json")]:
        prior = json.loads(path.read_text())
        if isinstance(prior, dict):
            previous.update(e["hash"] for e in prior.get("entries", prior.get("manifest", [])) if isinstance(e, dict) and "hash" in e)
    if len(hashes) != len(manifest) or hashes & previous:
        raise ValueError("Overlapping evaluation instances")
    sources = source_hashes()
    arguments = dict(vars(args), policy_seeds=seeds, train_budget=original["arguments"]["train_budget"], teacher_budget=original["arguments"]["teacher_budget"])
    write(output/"protocol.json", dict(arguments=arguments, config=asdict(config), manifest=manifest,
        source_hashes=sources, input_checkpoint_hashes=checkpoint_hashes, prior_hashes_checked=len(previous),
        primary="completion-rl minus completion-replay pooled on-time utility; exploratory instance bootstrap",
        scope="Inference-only larger decision cap and state memoization, unchanged previously selected final checkpoints. Fresh holdout, same tight planted family; no retraining or external-distribution claim."))
    root = Path(__file__).resolve().parents[1]
    for filename, digest in sources.items():
        content = (root/filename).read_bytes()
        if hashlib.sha256(content).hexdigest() != digest:
            raise ValueError("Source changed during freeze")
        target = output/"source"/filename; target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(content)
    write(output/"checkpoints.json", checkpoint_hashes)
    write(output/"training.json", []); write(output/"teachers.json", [])
    print("Follow-up sources, instances and unchanged checkpoints frozen", flush=True)
    methods = ["onepass-uniform", "backtrack-uniform", "completion-uniform", "completion-imitation", "completion-replay", "completion-rl"]
    rows, references = [], []
    for identifier, split, instance in data:
        rng = np.random.default_rng(args.data_seed+len(rows))
        for seed in seeds:
            for method in rng.permutation(methods).tolist():
                if method == "onepass-uniform":
                    row, _, _ = construct(instance, None, mode="uniform", seed=seed, budget=args.eval_budget, config=config)
                else:
                    arm = method.removeprefix("completion-")
                    row, _, _ = recover_construct(instance, models.get((arm,seed)), mode="uniform" if "uniform" in method else "sample", seed=seed,
                        budget=args.eval_budget, config=config, lookahead=method != "backtrack-uniform", max_decisions=args.eval_decisions)
                valid, objective, _ = oracle_check(onts_instance_to_json_data(instance), row["schedule"], config)
                if valid != row["feasible"] or (valid and abs(objective-row["objective"]) > 1e-4):
                    raise ValueError("Independent schedule disagreement")
                rows.append(dict(id=identifier, split=split, seed=seed, method=method, utility=row["reward"] if row["success"] else 0., **row))
        result = ONTSMilp(instance, config).solve(time_limit=args.reference_budget, solver_threads=1)
        references.append(dict(id=identifier, status=result.status, objective=result.objective, upper_bound=result.upper_bound,
            schedule=result.schedule.int().tolist() if result.schedule is not None else None, runtime=result.runtime))
        print(f"evaluated {identifier}; reference {result.status}", flush=True)
    write(output/"evaluations.json", rows); write(output/"references.json", references)
    paired = contrasts(rows); write(output/"contrasts.json", paired)
    lines = ["# Completion-aware recovery: memoized inference follow-up", "", "Unchanged final checkpoints; new untouched tighter-family instances. This is an inference intervention, not retrained recovery RL. Failures remain in utility/feasibility denominators.", "", "| Split | Method | On-time success | Mean utility | Mean runtime | Cap hits |", "|---|---|---:|---:|---:|---:|"]
    for split in ("test", "ood"):
        for method in methods:
            selected = [r for r in rows if r["method"] == method and r["split"] == split]
            lines.append(f"| {split} | {method} | {sum(r['success'] for r in selected)}/{len(selected)} | {np.mean([r['utility'] for r in selected]):.6f} | {np.mean([r['runtime'] for r in selected]):.3f} | {sum(r.get('sampled_decisions',0)>=args.eval_decisions for r in selected)} |")
    lines += ["", f"Evaluation overruns: {sum(r['overrun']>0 for r in rows)}. Certified references: {sum(r['status']=='optimal' for r in references)}/{len(references)}.", "", "Paired utility differences; exploratory marginal intervals:", ""]
    for c in paired:
        lines.append(f"- {c['split']}: {c['first']} minus {c['second']}: {c['difference']:.6f}, interval [{c['interval'][0]:.6f}, {c['interval'][1]:.6f}], {c['instances']} instances.")
    (output/"REPORT.md").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", required=True)
    parser.add_argument("--history-root", required=True)
    parser.add_argument("--backend-snapshot", required=True)
    parser.add_argument("--data-seed", type=int, default=2026101400)
    parser.add_argument("--test-per-regime", type=int, default=2)
    parser.add_argument("--ood-per-regime", type=int, default=1)
    parser.add_argument("--eval-budget", type=float, default=2.)
    parser.add_argument("--reference-budget", type=float, default=5.)
    parser.add_argument("--eval-decisions", type=int, default=4096)
    args = parser.parse_args()
    if any(not np.isfinite(value) or value <= 0 for key,value in vars(args).items() if isinstance(value,(int,float)) and key != "data_seed"):
        parser.error("Budgets and counts must be positive and finite")
    run(args)


if __name__ == "__main__":
    main()
