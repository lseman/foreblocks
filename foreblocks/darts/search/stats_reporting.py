from typing import Any, Dict, List

import numpy as np


def save_json(path: str, obj: Any) -> None:
    import json
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def save_csv(path: str, header, rows) -> None:
    import csv
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def mean_std(xs) -> tuple[float, float]:
    xs = list(xs)
    if not xs:
        return 0.0, 0.0
    m = float(np.mean(xs))
    s = float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0
    return m, s


def lpt_estimate(
    work_times, workers, overhead_per_task: float = 0.0, fixed_overhead: float = 0.0
) -> float:
    """
    Greedy LPT bin packing estimate of wall time with `workers`.
    Adds overhead_per_task to each task and fixed_overhead once.
    """
    work = [float(t) + float(overhead_per_task) for t in work_times]
    if not work:
        return float(fixed_overhead)
    w = max(1, int(workers))
    bins = [0.0 for _ in range(w)]
    for t in sorted(work, reverse=True):
        i = int(np.argmin(bins))
        bins[i] += t
    return float(max(bins) + float(fixed_overhead))


def append_whatif_estimates(
    *,
    phase: str,
    run_id: str,
    work_times: List[float],
    parallelism_levels: List[int],
    overhead_per_task: float,
    fixed_overhead: float,
    whatif_rows: List[List[Any]],
) -> List[Dict[str, float]]:
    estimates = []
    for w in parallelism_levels:
        est = lpt_estimate(
            work_times,
            workers=w,
            overhead_per_task=overhead_per_task,
            fixed_overhead=fixed_overhead,
        )
        estimates.append(
            {"phase": phase, "workers": int(w), "est_wall_time_sec": float(est)}
        )
        whatif_rows.append([run_id, phase, int(w), float(est)])
    return estimates
