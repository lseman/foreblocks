"""Paired per-instance inference; repeated seeds are not independent samples."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_mean(values, *, seed=0, resamples=10000):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Provide finite independent instance-level observations")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(
        axis=1
    )
    return {
        "mean": float(values.mean()),
        "ci95": [float(v) for v in np.quantile(means, [0.025, 0.975])],
        "instances": len(values),
        "resamples": resamples,
    }


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    if np.any((pvalues < 0) | (pvalues > 1)):
        raise ValueError("p-values must be in [0,1]")
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def paired_comparison(rows, *, method, baseline, track, budget, seed=0, split=None):
    selected = [
        r
        for r in rows
        if r["track"] == track
        and r["budget"] == budget
        and r["method"] in (method, baseline)
        and (split is None or r["split"] == split)
    ]
    grouped = {}
    for row in selected:
        if row.get("error"):
            continue
        grouped.setdefault((row["instance_id"], row["search_seed"]), {})[
            row["method"]
        ] = row
    pairs = {}
    outcomes = {
        "both_feasible": 0,
        "method_only_feasible": 0,
        "baseline_only_feasible": 0,
        "neither_feasible": 0,
        "missing_pair": 0,
    }
    for (instance_id, _), pair in grouped.items():
        if method not in pair or baseline not in pair:
            outcomes["missing_pair"] += 1
            continue
        a, b = pair[method], pair[baseline]
        if a["feasible"] and b["feasible"]:
            outcomes["both_feasible"] += 1
            delta = (
                100 * (a["objective"] - b["objective"]) / max(abs(b["objective"]), 1e-9)
            )
            pairs.setdefault(instance_id, []).append(delta)
        elif a["feasible"]:
            outcomes["method_only_feasible"] += 1
        elif b["feasible"]:
            outcomes["baseline_only_feasible"] += 1
        else:
            outcomes["neither_feasible"] += 1
    means = np.array([np.mean(pairs[k]) for k in sorted(pairs)])
    wins = int(np.sum(means > 1e-8))
    losses = int(np.sum(means < -1e-8))
    ties = len(means) - wins - losses
    sign_p = float(binomtest(wins, wins + losses, 0.5).pvalue) if wins + losses else 1.0
    return {
        "method": method,
        "baseline": baseline,
        "track": track,
        "budget": budget,
        "split": split,
        "paired_feasibility": outcomes,
        "relative_objective_change_percent": bootstrap_mean(means, seed=seed)
        if len(means)
        else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_p": sign_p,
        "unit": "instance means over matched seeds; objective comparison conditional on both feasible",
    }


def normalized_auc(history, *, budget, reference):
    """Integrate observed incumbent/reference; truncate overrunning events.

    Before the first feasible schedule the score is zero. Reference is a common
    best-known feasible objective, explicitly not assumed to be an optimum.
    """
    if budget <= 0 or reference <= 0:
        raise ValueError("Positive budget/reference required")
    previous_time = 0.0
    score = 0.0
    area = 0.0
    for event in sorted(history, key=lambda item: item["time"]):
        time = max(previous_time, min(float(event["time"]), budget))
        area += (time - previous_time) * score
        previous_time = time
        if event["time"] <= budget and event.get("objective") is not None:
            score = max(score, float(event["objective"]) / reference)
        if time >= budget:
            break
    area += (budget - previous_time) * score
    return float(area / budget)
