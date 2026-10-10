"""Run resumable ONTS correctness, training, validation and held-out benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import traceback
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ..environments.onts_env import load_onts_instance
from ..hybrid import solve_hybrid
from ..learning import train_neighborhood_policy
from ..milp import ONTSConfig, ONTSMilp
from ..neighborhood import NeighborhoodPolicy
from .datasets import build_suite, instance_hash, source_hashes
from .oracle import run_exhaustive_audit
from .protocol import (
    AblatedPolicy,
    append_record,
    environment_metadata,
    objectives_at_budget,
    read_jsonl,
)

DEFAULT_PLAN = {
    "format": "onts-study-plan-v1",
    "dataset_seed": 20261105,
    "audit_cases": 120,
    "train_per_regime": 4,
    "validation_per_regime": 2,
    "test_per_regime": 4,
    "ood_per_regime": 2,
    "training_seeds": [0, 1, 2],
    "search_seeds": [0, 1, 2],
    "budgets": [1.0, 3.0],
    "bundled_budgets": [1.0],
    "methods": ["exact", "random", "heuristic", "neural"],
    "ablation_methods": ["untrained", "no_future", "no_edges"],
    "initialization_time": 2.0,
    "reference_time": 3.0,
    "training": {
        "hidden": 32,
        "layers": 2,
        "rounds": 6,
        "epochs": 8,
        "repair_time": 0.35,
        "fraction": 0.4,
        "block_size": 4,
    },
    "validation_fractions": [0.2, 0.35, 0.5],
    "validation_budget": 2.0,
    "repair_time": 0.5,
    "init_fraction": 0.3,
    "block_size": 4,
    "torch_threads": 1,
    "config": asdict(ONTSConfig()),
    "claims": {
        "primary": "neural vs exact on cold track at largest synthetic budget",
        "secondary": "neural vs random and heuristic on common-incumbent track",
        "inference": "paired instance means over matched seeds; Holm-adjusted sign tests",
        "objective_conditioning": "objective comparisons require both feasible; separately report all failures",
        "ablations": "test-time interventions without retraining; not architecture superiority claims",
        "bundled": "external instances evaluated without training on them",
    },
}


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def open_study(output: Path, *, pilot=False, bundled=True, overrides=None):
    path = output / "plan.json"
    if path.exists():
        plan = json.loads(path.read_text())
        if plan["source_hashes"] != source_hashes():
            raise ValueError(
                "Source changed since plan freeze. Use a new output directory to avoid mixing implementations."
            )
        return plan, json.loads((output / "data" / "manifest.json").read_text())
    plan = json.loads(json.dumps(DEFAULT_PLAN))
    if pilot:
        plan.update(
            train_per_regime=2,
            validation_per_regime=1,
            test_per_regime=1,
            ood_per_regime=1,
            training_seeds=[0],
            search_seeds=[0],
            budgets=[1.0],
            bundled_budgets=[1.0],
            audit_cases=12,
            validation_fractions=[0.2, 0.5],
            validation_budget=1.0,
            reference_time=1.0,
        )
        plan["training"].update(
            rounds=3, epochs=3, hidden=16, layers=1, repair_time=0.2
        )
    if overrides:
        for key, value in overrides.items():
            if key not in plan:
                raise ValueError(f"Unknown plan option: {key}")
            if isinstance(value, dict) and isinstance(plan[key], dict):
                plan[key].update(value)
            else:
                plan[key] = value
    root = Path(__file__).resolve().parents[1]
    manifest = build_suite(
        output / "data",
        train_per_regime=plan["train_per_regime"],
        validation_per_regime=plan["validation_per_regime"],
        test_per_regime=plan["test_per_regime"],
        ood_per_regime=plan["ood_per_regime"],
        seed=plan["dataset_seed"],
        bundled_dir=root / "instances" / "examples" if bundled else None,
    )
    plan["source_hashes"] = source_hashes()
    plan["environment"] = environment_metadata()
    plan["manifest_hash"] = hashlib.sha256(
        (output / "data" / "manifest.json").read_bytes()
    ).hexdigest()
    plan["pilot"] = pilot
    write_json(path, plan)
    return plan, manifest


def verify_manifest(output: Path, plan: dict, manifest: dict):
    if (
        hashlib.sha256((output / "data" / "manifest.json").read_bytes()).hexdigest()
        != plan["manifest_hash"]
    ):
        raise ValueError("Manifest changed")
    hashes = set()
    for entry in manifest["entries"]:
        instance = load_onts_instance(entry["path"])
        digest = instance_hash(instance)
        if digest != entry["hash"] or digest in hashes:
            raise ValueError("Instance mutation or duplicate content detected")
        hashes.add(digest)


def prepare_references(output, plan, manifest):
    path = output / "references.jsonl"
    existing = {row["instance_id"] for row in read_jsonl(path)}
    for entry in manifest["entries"]:
        if entry["split"] == "train" or entry["id"] in existing:
            continue
        started = perf_counter()
        instance = load_onts_instance(entry["path"])
        p = ONTSMilp(instance, ONTSConfig(**plan["config"]))
        if entry["witness"] is not None:
            initial = torch.tensor(entry["witness"])
        else:
            initial = p.solve(
                proposal=torch.zeros(instance.J, instance.T),
                time_limit=plan["initialization_time"],
            ).schedule
        result = p.solve(time_limit=plan["reference_time"], incumbent=initial)
        row = {
            "instance_id": entry["id"],
            "split": entry["split"],
            "hash": entry["hash"],
            "initial_schedule": initial.int().tolist() if initial is not None else None,
            "initial_objective": p.config.validate(instance, initial).objective
            if initial is not None
            else None,
            "reference_objective": result.objective,
            "upper_bound": result.upper_bound,
            "status": result.status,
            "setup_runtime": perf_counter() - started,
            "reference_schedule": result.schedule.int().tolist()
            if result.schedule is not None
            else None,
        }
        append_record(path, row)
        print(
            f"reference {entry['id']} initial={row['initial_objective']} reference={result.objective}",
            flush=True,
        )


def train(output, plan, manifest):
    entries = [e for e in manifest["entries"] if e["split"] == "train"]
    instances = [load_onts_instance(e["path"]) for e in entries]
    schedules = [torch.tensor(e["witness"]) for e in entries]
    for seed in plan["training_seeds"]:
        path = output / "models" / f"policy_{seed}.pt"
        if path.exists():
            continue
        model, report = train_neighborhood_policy(
            instances,
            seed=seed,
            config=ONTSConfig(**plan["config"]),
            initial_schedules=schedules,
            **plan["training"],
        )
        report.update(
            config=plan["config"],
            training_hashes=[e["hash"] for e in entries],
            source_hashes=plan["source_hashes"],
            instances=[e["id"] for e in entries],
        )
        write_json(path.with_suffix(".json"), report)
        if not report["trained"]:
            raise RuntimeError(f"No improving teacher repairs for training seed {seed}")
        path.parent.mkdir(parents=True, exist_ok=True)
        model.save(path, metadata=report)
        print(
            f"training seed={seed} improving_repairs={report['improving_repairs']} loss={report['loss'][-1]:.4f}",
            flush=True,
        )


def load_policies(output, plan, manifest):
    models = {
        seed: NeighborhoodPolicy.load(output / "models" / f"policy_{seed}.pt")
        for seed in plan["training_seeds"]
    }
    test_hashes = {e["hash"] for e in manifest["entries"] if e["split"] != "train"}
    for model in models.values():
        if test_hashes.intersection(model.training_metadata["training_hashes"]):
            raise ValueError("Training/evaluation content overlap")
        if model.training_metadata["config"] != plan["config"]:
            raise ValueError("Checkpoint model conventions mismatch")
    return models


def validate_search(output, plan, manifest, policies, references):
    path = output / "validation.jsonl"
    rows = read_jsonl(path)
    finished = {(r["instance_id"], r["method"], r["fraction"]) for r in rows}
    for entry in manifest["entries"]:
        if entry["split"] != "validation":
            continue
        instance = load_onts_instance(entry["path"])
        initial = torch.tensor(references[entry["id"]]["initial_schedule"])
        for method in ("random", "heuristic", "neural"):
            for fraction in plan["validation_fractions"]:
                if (entry["id"], method, fraction) in finished:
                    continue
                result = solve_hybrid(
                    instance,
                    selector=method,
                    policy=policies[plan["training_seeds"][0]],
                    time_limit=plan["validation_budget"],
                    repair_time=plan["repair_time"],
                    neighborhood_fraction=fraction,
                    block_size=plan["block_size"],
                    initial_schedule=initial,
                    initial_exact=False,
                    exploration=0.2,
                )
                available = objectives_at_budget(
                    result.to_dict(), plan["validation_budget"]
                )
                row = {
                    "instance_id": entry["id"],
                    "method": method,
                    "fraction": fraction,
                    "score": available / references[entry["id"]]["initial_objective"]
                    if available is not None
                    else 0.0,
                    "objective_at_budget": available,
                    "runtime": result.runtime,
                }
                rows.append(row)
                append_record(path, row)
    selected = {}
    for method in ("random", "heuristic", "neural"):
        scores = {
            fraction: np.mean(
                [
                    r["score"]
                    for r in rows
                    if r["method"] == method and r["fraction"] == fraction
                ]
            )
            for fraction in plan["validation_fractions"]
        }
        selected[method] = max(scores, key=lambda f: (scores[f], -f))
    write_json(
        output / "selection.json",
        {
            "fractions": selected,
            "criterion": "mean validated objective / common initial objective on validation only",
            "checkpoint_selection": "seed 0 fixed; all training seeds used in testing",
        },
    )
    print(f"validation selection {selected}", flush=True)
    return selected


def benchmark(output, plan, manifest, policies, references, fractions):
    path = output / "runs.jsonl"
    rows = read_jsonl(path)
    completed = {row["run_id"] for row in rows}
    tasks = []
    for entry in manifest["entries"]:
        if entry["split"] not in ("test", "ood", "bundled"):
            continue
        budgets = (
            plan["bundled_budgets"] if entry["split"] == "bundled" else plan["budgets"]
        )
        for budget in budgets:
            for seed in plan["search_seeds"]:
                for track in ("cold", "common"):
                    if track == "common" and (
                        entry["split"] == "bundled" or budget != max(budgets)
                    ):
                        continue
                    if (
                        track == "common"
                        and references[entry["id"]]["initial_schedule"] is None
                    ):
                        continue
                    methods = plan["methods"]
                    if track == "common" and entry["split"] != "bundled":
                        methods = methods + plan["ablation_methods"]
                    if entry["split"] == "bundled" and track == "common":
                        # External cold evaluation plus common-incumbent diagnosis.
                        methods = ["exact", "random", "heuristic", "neural"]
                    for method in methods:
                        run_id = f"{entry['id']}|{track}|{budget}|{seed}|{method}"
                        tasks.append((run_id, entry, budget, seed, track, method))
    # Randomized order prevents systematic late-run thermal/order advantages.
    rng = np.random.default_rng(plan["dataset_seed"] + 99)
    rng.shuffle(tasks)
    for index, (run_id, entry, budget, seed, track, method) in enumerate(tasks):
        if run_id in completed:
            continue
        instance = load_onts_instance(entry["path"])
        policy_seed = plan["training_seeds"][seed % len(plan["training_seeds"])]
        policy = policies[policy_seed]
        selector = method
        if method in ("no_future", "no_edges"):
            policy = AblatedPolicy(
                policy, no_future=method == "no_future", no_edges=method == "no_edges"
            )
            selector = "neural"
        elif method == "untrained":
            torch.manual_seed(policy_seed + 4000)
            policy = NeighborhoodPolicy(
                hidden=plan["training"]["hidden"], layers=plan["training"]["layers"]
            )
            selector = "neural"
        initial = (
            torch.tensor(references[entry["id"]]["initial_schedule"])
            if track == "common"
            else None
        )
        row = {
            "run_id": run_id,
            "instance_id": entry["id"],
            "instance_hash": entry["hash"],
            "split": entry["split"],
            "regime": entry["regime"],
            "jobs": entry["jobs"],
            "horizon": entry["horizon"],
            "method": method,
            "budget": budget,
            "track": track,
            "search_seed": seed,
            "training_seed": policy_seed,
            "fraction": fractions.get(selector, 0.35),
        }
        try:
            result = solve_hybrid(
                instance,
                selector=selector,
                policy=policy,
                time_limit=budget,
                repair_time=plan["repair_time"],
                init_fraction=plan["init_fraction"],
                neighborhood_fraction=row["fraction"],
                block_size=plan["block_size"],
                initial_schedule=initial,
                initial_exact=track == "cold",
                seed=seed,
                config=ONTSConfig(**plan["config"]),
            )
            payload = result.to_dict()
            row.update(payload)
            row["returned_objective"] = row["objective"]
            row["returned_feasible"] = row["feasible"]
            # Do not credit schedules becoming available after the fixed deadline.
            row["objective"] = objectives_at_budget(payload, budget)
            row["feasible"] = row["objective"] is not None
            row["overrun"] = max(0, result.runtime - budget)
            if result.schedule is not None:
                validation = ONTSConfig(**plan["config"]).validate(
                    instance, result.schedule
                )
                if not validation.feasible:
                    raise RuntimeError(
                        f"Invalid returned solution: {validation.violations}"
                    )
            row["validation_passed"] = True
        except (ValueError, RuntimeError, MemoryError) as exc:
            row.update(
                error=f"{type(exc).__name__}: {exc}",
                traceback=traceback.format_exc(),
                feasible=False,
                objective=None,
            )
        append_record(path, row)
        print(
            f"run {index + 1}/{len(tasks)} {run_id} objective={row['objective']} feasible={row['feasible']}",
            flush=True,
        )
    return len(tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=[
            "all",
            "prepare",
            "audit",
            "train",
            "references",
            "validate",
            "benchmark",
            "report",
        ],
        default="all",
    )
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--plan-overrides",
        type=Path,
        help="JSON plan overrides, only when creating a new study",
    )
    parser.add_argument("--no-bundled", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    overrides = (
        json.loads(args.plan_overrides.read_text()) if args.plan_overrides else None
    )
    plan, manifest = open_study(
        args.output, pilot=args.pilot, bundled=not args.no_bundled, overrides=overrides
    )
    verify_manifest(args.output, plan, manifest)
    if args.phase == "prepare":
        print(
            f"Frozen plan and {len(manifest['entries'])} instance identities",
            flush=True,
        )
        return
    if args.phase in ("all", "audit"):
        path = args.output / "audit.json"
        audit = (
            json.loads(path.read_text())
            if path.exists()
            else run_exhaustive_audit(plan["audit_cases"])
        )
        write_json(path, audit)
        if not audit["all_passed"]:
            raise RuntimeError("Correctness audit failed; study stopped")
        print(f"audit {audit['schedules']} schedules passed", flush=True)
    if args.phase in ("all", "train"):
        train(args.output, plan, manifest)
    if args.phase in ("all", "references"):
        prepare_references(args.output, plan, manifest)
    if args.phase in ("all", "validate", "benchmark"):
        models = load_policies(args.output, plan, manifest)
        references = {
            row["instance_id"]: row
            for row in read_jsonl(args.output / "references.jsonl")
        }
        if args.phase in ("all", "validate"):
            fractions = validate_search(args.output, plan, manifest, models, references)
        else:
            fractions = json.loads((args.output / "selection.json").read_text())[
                "fractions"
            ]
        if args.phase in ("all", "benchmark"):
            expected = benchmark(
                args.output, plan, manifest, models, references, fractions
            )
            write_json(
                args.output / "completion.json",
                {
                    "expected_runs": expected,
                    "completed_runs": len(read_jsonl(args.output / "runs.jsonl")),
                },
            )
    if args.phase in ("all", "report"):
        from .report import generate_report

        generate_report(args.output)


if __name__ == "__main__":
    main()
