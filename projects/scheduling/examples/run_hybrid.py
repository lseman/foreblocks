"""Solve or benchmark ONTS with exact, random, heuristic, or learned LNS."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

# Supports both python -m and direct invocation from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.scheduling.classical import solve_classical
from projects.scheduling.environments.onts_env import load_onts_instance
from projects.scheduling.hybrid import solve_hybrid
from projects.scheduling.job_profile import solve_profile
from projects.scheduling.local_search import solve_local_search
from projects.scheduling.milp import ONTSConfig
from projects.scheduling.neighborhood import NeighborhoodPolicy
from projects.scheduling.profile_portfolio import solve_profile_portfolio
from projects.scheduling.ranked_search import solve_ranked
from projects.scheduling.ranking import NeighborhoodRanker
from projects.scheduling.research.datasets import instance_hash
from projects.scheduling.research.protocol import objectives_at_budget
from projects.scheduling.trust_region import SolutionPredictor, solve_predict_search


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--instance", type=Path)
    source.add_argument("--instance-dir", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[
            "exact",
            "random",
            "heuristic",
            "neural",
            "ranked",
            "uniform",
            "bandit",
            "adaptive",
            "lp_guided",
            "rins",
            "local_branch",
            "adaptive_local_branch",
            "classical_portfolio",
            "classical_portfolio_plus",
            "profile",
            "profile_fallback",
            "profile_portfolio",
            "local_search",
            "predict_search",
            "predict_search_polish",
        ],
        default=["heuristic"],
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--ranker-checkpoint", type=Path)
    parser.add_argument("--predictor-checkpoint", type=Path)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--time-limit", type=float, default=30)
    parser.add_argument("--repair-time", type=float, default=2)
    parser.add_argument("--init-fraction", type=float, default=0.3)
    parser.add_argument("--fraction", type=float, default=0.25)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--strict-final-run", action="store_true")
    parser.add_argument(
        "--config", type=Path, help="JSON ONTSConfig battery/model parameters"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    torch.set_num_threads(args.threads)
    os.environ["ONTS_SOLVER_THREADS"] = str(args.threads)
    if "neural" in args.methods and args.checkpoint is None:
        parser.error("--checkpoint is required for neural selection")
    if (
        any(method in args.methods for method in ("ranked", "adaptive"))
        and args.ranker_checkpoint is None
    ):
        parser.error("--ranker-checkpoint is required for ranked/adaptive selection")
    if (
        any(method.startswith("predict_search") for method in args.methods)
        and args.predictor_checkpoint is None
    ):
        parser.error("--predictor-checkpoint is required for predict_search")
    predictor = (
        SolutionPredictor.load(args.predictor_checkpoint, device=args.device)
        if args.predictor_checkpoint
        else None
    )
    if predictor is not None and not predictor.training_metadata.get("trained"):
        parser.error("Predictor checkpoint has no training data")
    ranker = (
        NeighborhoodRanker.load(args.ranker_checkpoint, device=args.device)
        if args.ranker_checkpoint
        else None
    )
    if ranker is not None and not ranker.training_metadata.get("trained", False):
        parser.error("Ranker checkpoint has no informative outcome training")
    policy = (
        NeighborhoodPolicy.load(args.checkpoint, device=args.device)
        if args.checkpoint
        else None
    )
    if policy is not None and not getattr(policy, "training_metadata", {}).get(
        "trained", False
    ):
        parser.error(
            "Checkpoint has no improving-repair training; train on harder instances"
        )
    paths = (
        [args.instance] if args.instance else sorted(args.instance_dir.glob("*.json"))
    )
    if not paths:
        parser.error("No JSON instances found")
    config_values = json.loads(args.config.read_text()) if args.config else {}
    if args.strict_final_run:
        config_values["allow_truncated_final_run"] = False
    config = ONTSConfig(**config_values)
    if policy is not None and policy.training_metadata.get("config") != asdict(config):
        parser.error("Checkpoint and evaluation model parameters differ")
    if predictor is not None and predictor.training_metadata.get("config") != asdict(
        config
    ):
        parser.error("Predictor and evaluation model parameters differ")
    if ranker is not None and ranker.training_metadata.get("config") != asdict(config):
        parser.error("Ranker and evaluation model parameters differ")
    rows = []
    for path in paths:
        instance = load_onts_instance(path)
        if ranker is not None and instance_hash(
            instance
        ) in ranker.training_metadata.get("training_hashes", []):
            parser.error("Evaluation instance overlaps ranker training content")
        if predictor is not None and instance_hash(
            instance
        ) in predictor.training_metadata.get("training_hashes", []):
            parser.error("Evaluation instance overlaps predictor training content")
        for seed in args.seeds:
            for method in args.methods:
                if method in ("profile", "profile_fallback"):
                    result = solve_profile(
                        instance,
                        time_limit=args.time_limit,
                        init_fraction=args.init_fraction,
                        fallback_time=args.repair_time
                        if method == "profile_fallback"
                        else 0.0,
                        seed=seed,
                        config=config,
                    )
                elif method == "profile_portfolio":
                    # Settings selected on development data (HEURISTICS.md).
                    result = solve_profile_portfolio(
                        instance,
                        controller="stagnation",
                        patience=10,
                        job_selection="rules",
                        time_limit=args.time_limit,
                        repair_time=args.repair_time,
                        fraction=args.fraction,
                        candidate_count=args.candidates,
                        init_fraction=args.init_fraction,
                        seed=seed,
                        config=config,
                    )
                elif method == "local_search":
                    result = solve_local_search(
                        instance, time_limit=args.time_limit, seed=seed, config=config
                    )
                elif method.startswith("predict_search"):
                    result = solve_predict_search(
                        instance,
                        predictor,
                        time_limit=args.time_limit,
                        polish=method == "predict_search_polish",
                        seed=seed,
                        config=config,
                    )
                elif method in (
                    "lp_guided",
                    "rins",
                    "local_branch",
                    "adaptive_local_branch",
                    "classical_portfolio",
                    "classical_portfolio_plus",
                ):
                    result = solve_classical(
                        instance,
                        mode={
                            "classical_portfolio": "portfolio",
                            "classical_portfolio_plus": "portfolio_plus",
                        }.get(method, method),
                        time_limit=args.time_limit,
                        repair_time=args.repair_time,
                        init_fraction=args.init_fraction,
                        fraction=args.fraction,
                        seed=seed,
                        config=config,
                    )
                elif method in ("ranked", "uniform", "bandit", "adaptive"):
                    result = solve_ranked(
                        instance,
                        mode=method,
                        policy=ranker,
                        time_limit=args.time_limit,
                        repair_time=args.repair_time,
                        init_fraction=args.init_fraction,
                        fraction=args.fraction,
                        candidate_count=args.candidates,
                        block_size=args.block_size,
                        seed=seed,
                        config=config,
                    )
                else:
                    result = solve_hybrid(
                        instance,
                        selector=method,
                        policy=policy,
                        time_limit=args.time_limit,
                        repair_time=args.repair_time,
                        init_fraction=args.init_fraction,
                        neighborhood_fraction=args.fraction,
                        block_size=args.block_size,
                        seed=seed,
                        config=config,
                    )
                row = {
                    "instance": str(path.resolve()),
                    "method": method,
                    "seed": seed,
                    **result.to_dict(),
                }
                row["returned_objective"] = row["objective"]
                row["returned_feasible"] = row["feasible"]
                row["objective"] = objectives_at_budget(row, args.time_limit)
                row["feasible"] = row["objective"] is not None
                row["overrun"] = max(0.0, result.runtime - args.time_limit)
                rows.append(row)
                print(
                    f"{path.name} {method} seed={seed}: {result.status} objective={result.objective} gap={result.relative_gap}"
                )
    summaries = {}
    for method in args.methods:
        group = [row for row in rows if row["method"] == method]
        feasible = [row for row in group if row["feasible"]]
        summaries[method] = {
            "runs": len(group),
            "feasible_rate": len(feasible) / len(group),
            "mean_runtime": sum(row["runtime"] for row in group) / len(group),
            "mean_feasible_objective": sum(row["objective"] for row in feasible)
            / len(feasible)
            if feasible
            else None,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "onts-benchmark-v1",
        "config": asdict(config),
        "parameters": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "summary": summaries,
        "runs": rows,
    }
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
