"""Train a semantic complete-neighborhood ranker on a training-only JSON split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from projects.scheduling.environments.onts_env import load_onts_instance
from projects.scheduling.milp import ONTSConfig
from projects.scheduling.ranking_learning import train_neighborhood_ranker


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--repair-time", type=float, default=0.25)
    parser.add_argument("--init-time", type=float, default=0.5)
    parser.add_argument("--fraction", type=float, default=0.35)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-semantics", action="store_true")
    parser.add_argument("--no-memory", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    paths = sorted(args.instance_dir.glob("*.json"))
    if not paths:
        parser.error("No JSON training instances found")
    config = ONTSConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    policy, metadata = train_neighborhood_ranker(
        [load_onts_instance(path) for path in paths],
        config=config,
        rounds=args.rounds,
        epochs=args.epochs,
        repair_time=args.repair_time,
        init_time=args.init_time,
        fraction=args.fraction,
        candidate_count=args.candidates,
        block_size=args.block_size,
        hidden=args.hidden,
        layers=args.layers,
        seed=args.seed,
        device=args.device,
        semantics=not args.no_semantics,
        memory=not args.no_memory,
    )
    metadata["instances"] = [str(path.resolve()) for path in paths]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not metadata["trained"]:
        parser.exit(
            2,
            "No comparative gain signal found; diagnostics saved, no trained checkpoint.\n",
        )
    policy.save(args.output, metadata=metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "outcomes"}, indent=2))


if __name__ == "__main__":
    main()
