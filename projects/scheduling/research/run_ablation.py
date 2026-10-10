"""Historical exploratory runner, not a controlled component study.

Use research.portfolio_ablation for the paired study with observed timestamps.

Runs profile_only and no_surplus variants on the same 40 bundled-like
instances as the v2 confirmatory study (1 seed each, to save time).
Results are saved as CSV for table generation.

Run from foreblocks root:
    .venv/bin/python -m projects.scheduling.research.run_ablation [options]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

# Must be imported relative to the research package.
# When run as -m projects.scheduling.research.run_ablation, these resolve correctly.
try:
    from ..profile_portfolio import solve_profile_portfolio
    from .protocol import objectives_at_budget
    from .realistic import realistic_instance
except ImportError:
    # Fallback for direct execution.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from profile_portfolio import solve_profile_portfolio
    from research.protocol import objectives_at_budget
    from research.realistic import realistic_instance

COUNT = 40
SEED = 101  # single seed for speed; v2 confirmatory used {101, 211}
BUDGET = 30.0
BASE = 61_000_000

# Shared controller parameters (same as v2 confirmatory).
CONTROLLER = dict(
    controller="stagnation",
    patience=10,
    job_selection="rules",
    fraction=0.2,
)

VARIANTS = {
    "profile_only": dict(
        **CONTROLLER,
        repair_time=1e-6,  # effectively disables MILP repairs
        polish_after_feasible=None,
        surplus_weight=0.0,
    ),
    "no_surplus": dict(
        **CONTROLLER,
        repair_time=1.0,
        polish_after_feasible=0.5,  # same early stop as v2
        surplus_weight=0.0,  # no surplus discount
    ),
}


def run(task):
    index, variant = task
    os.environ["ONTS_SOLVER_THREADS"] = "1"
    torch.set_num_threads(1)
    instance = realistic_instance(BASE + index)
    try:
        result = solve_profile_portfolio(
            instance,
            time_limit=BUDGET,
            seed=SEED,
            **VARIANTS[variant],
        )
        payload = result.to_dict()
        credited = objectives_at_budget(payload, BUDGET)
        return dict(
            instance=index,
            method=variant,
            J=instance.J,
            T=instance.T,
            objective=credited,
            feasible=credited is not None,
            runtime=payload["runtime"],
            time_to_first_feasible=payload.get("time_to_first_feasible"),
        )
    except Exception as exc:
        return dict(
            instance=index,
            method=variant,
            J=instance.J,
            T=instance.T,
            objective=None,
            feasible=False,
            runtime=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: stdout)")
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    tasks = [(i, v) for i in range(COUNT) for v in VARIANTS]
    print(f"Running {len(tasks)} ablation tasks...")

    with ProcessPoolExecutor(args.workers) as pool:
        rows = list(pool.map(run, tasks))

    # Write CSV
    header = ["instance", "method", "J", "T", "objective", "feasible",
              "runtime", "time_to_first_feasible"]
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Wrote {len(rows)} rows to {args.output}")
    else:
        import io
        writer = csv.DictWriter(sys.stdout, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    for v in VARIANTS:
        vr = [r for r in rows if r["method"] == v]
        feas = sum(1 for r in vr if r.get("feasible"))
        objs = [r["objective"] for r in vr if r.get("objective")]
        if objs:
            print(f"{v}: feasible={feas}/{len(vr)}, mean_obj={np.mean(objs):.1f}")
        else:
            print(f"{v}: no objectives")


if __name__ == "__main__":
    main()
