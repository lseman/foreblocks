"""Fresh, bounded exploratory comparison for the semantic neighborhood ranker.

One training/search seed and common incumbents: an engineering pilot, not a
confirmatory state-of-the-art evaluation. Resume rejects changed source/data.
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from ..environments.onts_env import load_onts_instance, onts_instance_to_json_data
from ..hybrid import solve_hybrid
from ..learning import train_neighborhood_policy
from ..milp import ONTSConfig
from ..neighborhood import NeighborhoodPolicy
from ..ranked_search import solve_ranked
from ..ranking import NeighborhoodRanker
from ..ranking_learning import train_neighborhood_ranker
from .datasets import build_suite, instance_hash, source_hashes
from .oracle import oracle_check
from .protocol import append_record, objectives_at_budget, read_jsonl
from .statistics import holm_adjust, paired_comparison


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    config = ONTSConfig()
    if (output / "plan.json").exists():
        plan = json.loads((output / "plan.json").read_text())
        manifest = json.loads((output / "data/manifest.json").read_text())
        if source_hashes() != plan["source_hashes"]:
            raise RuntimeError("Pilot source changed: use a new output directory")
    else:
        manifest = build_suite(
            output / "data",
            seed=915731,
            train_per_regime=3,
            validation_per_regime=1,
            test_per_regime=2,
            ood_per_regime=1,
        )
        plan = {
            "format": "onts-ranker-pilot-v1",
            "exploratory": True,
            "seed": 0,
            "source_hashes": source_hashes(),
            "config": asdict(config),
            "budget": 3.0,
            "repair_time": 0.3,
            "validation_budget": 1.5,
            "fractions": [0.2, 0.35, 0.5],
            "candidate_count": 8,
            "methods": [
                "exact",
                "random",
                "heuristic",
                "neural",
                "uniform",
                "bandit",
                "ranked",
                "adaptive",
                "untrained",
                "no_semantics",
                "no_memory",
            ],
            "training": {
                "hidden": 32,
                "layers": 2,
                "rounds": 4,
                "epochs": 12,
                "repair_time": 0.15,
                "fraction": 0.35,
            },
            "scope": "One seed; common planted incumbents; inference-only interventions; no bundled or cold evidence",
        }
        prior = output.parent / "local-study/data/manifest.json"
        if prior.exists():
            old = {e["hash"] for e in json.loads(prior.read_text())["entries"]}
            if old & {e["hash"] for e in manifest["entries"]}:
                raise RuntimeError("Fresh pilot overlaps completed study")
        save(output / "plan.json", plan)
        with tarfile.open(output / "frozen-source.tar.gz", "w:gz") as archive:
            root = Path(__file__).resolve().parents[1]
            for relative in plan["source_hashes"]:
                archive.add(
                    root / relative, arcname=str(Path("projects/scheduling") / relative)
                )
    for entry in manifest["entries"]:
        if instance_hash(load_onts_instance(entry["path"])) != entry["hash"]:
            raise RuntimeError("Pilot data hash changed")
    training = [e for e in manifest["entries"] if e["split"] == "train"]
    instances = [load_onts_instance(e["path"]) for e in training]
    schedules = [torch.tensor(e["witness"]) for e in training]
    if not (output / "ranker.pt").exists():
        ranker, metadata = train_neighborhood_ranker(
            instances, initial_schedules=schedules, config=config, **plan["training"]
        )
        save(output / "ranker-training.json", metadata)
        if not metadata["trained"]:
            raise RuntimeError("No informative ranking outcomes; checkpoint not saved")
        ranker.save(output / "ranker.pt", metadata=metadata)
        print(
            f"ranker training: {metadata['repair_outcomes']} outcomes, {metadata['zero_gain_outcomes']} zero-gain, {metadata['runtime']:.2f}s",
            flush=True,
        )
    ranker = NeighborhoodRanker.load(output / "ranker.pt")
    if not (output / "imitation.pt").exists():
        imitation, metadata = train_neighborhood_policy(
            instances, initial_schedules=schedules, config=config, **plan["training"]
        )
        save(output / "imitation-training.json", metadata)
        imitation.save(output / "imitation.pt", metadata=metadata)
    imitation = NeighborhoodPolicy.load(output / "imitation.pt")
    if not (output / "selection.json").exists():
        validation = [e for e in manifest["entries"] if e["split"] == "validation"]
        for fraction in plan["fractions"]:
            for entry in validation:
                result = solve_ranked(
                    load_onts_instance(entry["path"]),
                    policy=ranker,
                    initial_schedule=torch.tensor(entry["witness"]),
                    time_limit=plan["validation_budget"],
                    repair_time=plan["repair_time"],
                    fraction=fraction,
                )
                append_record(
                    output / "validation.jsonl",
                    {
                        "fraction": fraction,
                        "instance_id": entry["id"],
                        "objective": objectives_at_budget(
                            result.to_dict(), plan["validation_budget"]
                        ),
                    },
                )
        records = read_jsonl(output / "validation.jsonl")
        # Require complete feasible observations; ties prefer the smaller fraction.
        means = {
            f: float(
                np.mean(
                    [
                        r["objective"] if r["objective"] is not None else 0
                        for r in records
                        if r["fraction"] == f
                    ]
                )
            )
            for f in plan["fractions"]
        }
        selected = max(plan["fractions"], key=lambda f: means[f])
        save(
            output / "selection.json", {"fraction": selected, "validation_means": means}
        )
    fraction = json.loads((output / "selection.json").read_text())["fraction"]
    heldout = [e for e in manifest["entries"] if e["split"] in ("test", "ood")]
    train_hashes = set(ranker.training_metadata["training_hashes"])
    if train_hashes & {e["hash"] for e in heldout}:
        raise RuntimeError("Training/evaluation overlap")
    completed = {r["run_id"] for r in read_jsonl(output / "runs.jsonl")}
    tasks = [(e, method) for e in heldout for method in plan["methods"]]
    rng = np.random.default_rng(71)
    for index in rng.permutation(len(tasks)):
        entry, method = tasks[int(index)]
        run_id = f"{entry['id']}|{method}"
        if run_id in completed:
            continue
        instance = load_onts_instance(entry["path"])
        initial = torch.tensor(entry["witness"])
        if method in ("exact", "random", "heuristic", "neural"):
            result = solve_hybrid(
                instance,
                selector=method,
                policy=imitation,
                initial_schedule=initial,
                initial_exact=False,
                time_limit=plan["budget"],
                repair_time=plan["repair_time"],
                neighborhood_fraction={
                    "random": 0.35,
                    "heuristic": 0.5,
                    "neural": 0.2,
                }.get(method, 0.35),
            )
        else:
            policy = ranker
            mode = method
            if method in ("no_semantics", "no_memory"):
                policy = NeighborhoodRanker(
                    hidden=32,
                    layers=2,
                    semantics=method != "no_semantics",
                    memory=method != "no_memory",
                )
                policy.load_state_dict(ranker.state_dict())
                mode = "ranked"
            elif method == "untrained":
                torch.manual_seed(4000)
                policy = NeighborhoodRanker(hidden=32, layers=2)
                mode = "ranked"
            result = solve_ranked(
                instance,
                mode=mode,
                policy=policy,
                initial_schedule=initial,
                time_limit=plan["budget"],
                repair_time=plan["repair_time"],
                fraction=fraction,
            )
        payload = result.to_dict()
        feasible, objective, _ = oracle_check(
            onts_instance_to_json_data(instance), payload["schedule"], config
        )
        if not feasible or not np.isclose(objective, payload["objective"], atol=1e-4):
            raise RuntimeError("Independent oracle rejected output")
        row = {
            "run_id": run_id,
            "instance_id": entry["id"],
            "instance_hash": entry["hash"],
            "method": method,
            "split": entry["split"],
            "track": "common",
            "budget": plan["budget"],
            "search_seed": 0,
            **payload,
            "returned_objective": payload["objective"],
            "returned_feasible": payload["feasible"],
            "validation_passed": True,
            "overrun": max(0, result.runtime - plan["budget"]),
        }
        row["objective"] = objectives_at_budget(payload, plan["budget"])
        row["feasible"] = row["objective"] is not None
        append_record(output / "runs.jsonl", row)
        print(
            f"run {len(read_jsonl(output / 'runs.jsonl'))}/{len(tasks)} {run_id}: {row['objective']}",
            flush=True,
        )
    rows = read_jsonl(output / "runs.jsonl")
    # Statistics helper expects the historical name 'neural'; use a local copy.
    comparison_rows = [
        dict(
            row,
            method="neural"
            if row["method"] == "ranked"
            else "imitation"
            if row["method"] == "neural"
            else row["method"],
        )
        for row in rows
    ]
    comparisons = [
        paired_comparison(
            comparison_rows,
            method="neural",
            baseline=baseline,
            track="common",
            budget=plan["budget"],
            split=split,
        )
        for split in ("test", "ood")
        for baseline in [
            "exact",
            "random",
            "heuristic",
            "imitation",
            "uniform",
            "bandit",
            "adaptive",
            "untrained",
            "no_semantics",
            "no_memory",
        ]
    ]
    for comparison, p_value in zip(
        comparisons, holm_adjust([c["sign_test_p"] for c in comparisons])
    ):
        comparison["holm_adjusted_p"] = p_value
    save(
        output / "analysis.json",
        {
            "exploratory": True,
            "comparisons": comparisons,
            "completed_runs": len(rows),
            "expected_runs": len(tasks),
            "all_independent_oracle_checks_passed": True,
        },
    )
    text = [
        "# Semantic ranker pilot",
        "",
        f"Completed {len(rows)}/{len(tasks)} runs. Independent output checks passed.",
        "One seed; common planted incumbents; fresh synthetic identities. No SOTA or confirmatory claim.",
        "Inference-only semantics/memory interventions are not retrained architecture ablations.",
        "All pool methods share the validation-selected fraction; legacy fractions are fixed from the previous study.",
        "The ranker receives more teacher calls than imitation; training costs and all zero-gain outcomes are saved.",
        "",
        "| Split | Comparator | Ranker change (%) | Marginal 95% CI | Adjusted p |",
        "|---|---|---:|---|---:|",
    ]
    for c in comparisons:
        effect = c["relative_objective_change_percent"]
        text.append(
            f"| {c['split']} | {c['baseline']} | {effect['mean']:.3f} | [{effect['ci95'][0]:.3f}, {effect['ci95'][1]:.3f}] | {c['holm_adjusted_p']:.4f} |"
            if effect
            else f"| {c['split']} | {c['baseline']} | unavailable | | |"
        )
    (output / "REPORT.md").write_text("\n".join(text) + "\n")
    print((output / "REPORT.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
