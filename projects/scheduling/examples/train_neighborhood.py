"""Train the coefficient-aware neighborhood policy on solver improvements."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from projects.scheduling.environments.onts_env import load_onts_instance
from projects.scheduling.learning import train_neighborhood_policy
from projects.scheduling.milp import ONTSConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-dir", type=Path, required=True, help="Training split only"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--init-time", type=float, default=0.5)
    parser.add_argument("--repair-time", type=float, default=1)
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--block-size", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    paths = sorted(args.instance_dir.glob("*.json"))
    if not paths:
        parser.error("No JSON training instances found")
    config = ONTSConfig(**(json.loads(args.config.read_text()) if args.config else {}))
    policy, report = train_neighborhood_policy(
        [load_onts_instance(p) for p in paths],
        hidden=args.hidden,
        layers=args.layers,
        rounds=args.rounds,
        epochs=args.epochs,
        init_time=args.init_time,
        repair_time=args.repair_time,
        fraction=args.fraction,
        block_size=args.block_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        config=config,
    )
    report["instances"] = [str(path.resolve()) for path in paths]
    report["config"] = asdict(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["trained"]:
        print(json.dumps(report, indent=2))
        parser.exit(
            2,
            "No improving repairs found. Use harder instances, more rounds, or a shorter initialization budget. No checkpoint saved.\n",
        )
    policy.save(args.output, metadata=report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
