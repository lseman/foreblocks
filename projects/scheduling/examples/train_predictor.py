"""Train a contrastive solution predictor for predict-and-search."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.scheduling.environments.onts_env import load_onts_instance
from projects.scheduling.milp import ONTSConfig
from projects.scheduling.research.datasets import instance_hash
from projects.scheduling.trust_region import train_predictor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--collect-budget", type=float, default=6.0)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    paths = sorted(args.instance_dir.glob("*.json"))
    if not paths:
        parser.error("No JSON training instances found")
    config = ONTSConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    instances = [load_onts_instance(path) for path in paths]
    model, metadata = train_predictor(
        instances,
        config=config,
        epochs=args.epochs,
        hidden=args.hidden,
        layers=args.layers,
        collect_budget=args.collect_budget,
        seed=args.seed,
        device=args.device,
    )
    metadata.update(
        instances=[str(path.resolve()) for path in paths],
        training_hashes=[instance_hash(instance) for instance in instances],
        config=asdict(config),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    if not metadata["trained"]:
        parser.exit(2, "No training solutions collected; no checkpoint saved.\n")
    model.save(args.output, metadata=metadata)
    print(json.dumps({k: v for k, v in metadata.items() if k != "loss"}, indent=2))


if __name__ == "__main__":
    main()
