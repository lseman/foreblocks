#!/usr/bin/env python3

from __future__ import annotations

import csv
import sys
from pathlib import Path

DEFAULT_RUN_DIR = Path("/data/dev/foreblocks/estudo_rebeca/run_hidro_50_5")


def _load_mse(metrics_path: Path) -> float:
    with metrics_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        row = next(reader, None)
    if row is None:
        raise ValueError(f"Empty metrics file: {metrics_path}")
    if "mse" not in row:
        raise ValueError(f"Missing 'mse' column in: {metrics_path}")
    return float(row["mse"])


def collect_runs(run_dir: Path) -> list[tuple[float, str, Path]]:
    ranked_runs: list[tuple[float, str, Path]] = []
    for subrun_dir in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        metrics_path = subrun_dir / "darts_final_metrics.csv"
        if not metrics_path.is_file():
            continue
        mse = _load_mse(metrics_path)
        ranked_runs.append((mse, subrun_dir.name, metrics_path))
    return ranked_runs


def main() -> int:
    run_dir = Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    if not run_dir.is_dir():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        return 1

    ranked_runs = collect_runs(run_dir)
    if not ranked_runs:
        print(
            f"No runs with darts_final_metrics.csv found in {run_dir}", file=sys.stderr
        )
        return 1

    ranked_runs.sort(key=lambda item: item[0])

    print(f"Top 5 runs by final MSE in {run_dir}")
    print("rank,run,mse,metrics_csv")
    for rank, (mse, run_name, metrics_path) in enumerate(ranked_runs[:5], start=1):
        print(f"{rank},{run_name},{mse:.12f},{metrics_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
