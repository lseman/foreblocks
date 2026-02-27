from typing import Any, Dict, List, Tuple

import numpy as np

from ..scoring import normalize_metric_value


def candidate_diversity_bonus(selected_ops: List[str], all_ops: List[str]) -> float:
    """Small bounded score bonus for operation-set diversity."""
    if not selected_ops:
        return 0.0

    unique_non_identity = sorted({op for op in selected_ops if op != "Identity"})
    if not unique_non_identity:
        return 0.0

    max_non_identity = max(1, len(all_ops) - 1)
    ratio = min(len(unique_non_identity) / max_non_identity, 1.0)
    return 0.12 * ratio


def candidate_signature(candidate: Dict[str, Any]) -> Tuple[Any, ...]:
    ops = tuple(sorted(candidate.get("selected_ops", [])))
    return (
        ops,
        int(candidate.get("hidden_dim", 0)),
        int(candidate.get("num_cells", 0)),
        int(candidate.get("num_nodes", 0)),
    )


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep best-scoring candidate for each architecture signature."""
    best_by_sig: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
    for cand in candidates:
        sig = candidate_signature(cand)
        prev = best_by_sig.get(sig)
        if prev is None or float(cand.get("score", 0.0)) > float(
            prev.get("score", 0.0)
        ):
            best_by_sig[sig] = cand
    return list(best_by_sig.values())


def normalize_metric_for_pool(metric: str, value: float) -> float:
    """Apply the same per-metric transform used by zero-cost scoring."""
    return normalize_metric_value(metric, float(value))


def rescore_candidates_poolwise(candidates: List[Dict[str, Any]]) -> None:
    """Rescore candidates using pool-wise z-scored metrics for balanced weighting."""
    if not candidates:
        return

    first_metrics = candidates[0].get("metrics", {})
    cfg = first_metrics.get("config") if isinstance(first_metrics, dict) else None
    weights = getattr(cfg, "weights", None)
    if not isinstance(weights, dict) or not weights:
        return

    metric_keys = list(weights.keys())

    transformed_by_metric: Dict[str, List[float]] = {k: [] for k in metric_keys}
    for cand in candidates:
        raw_metrics = cand.get("metrics", {}).get("metrics", {})
        for metric in metric_keys:
            raw_val = raw_metrics.get(metric, 0.0)
            tval = normalize_metric_for_pool(metric, float(raw_val))
            transformed_by_metric[metric].append(tval)

    z_by_metric: Dict[str, List[float]] = {}
    for metric, vals in transformed_by_metric.items():
        arr = np.asarray(vals, dtype=np.float64)
        mu = float(np.mean(arr)) if arr.size else 0.0
        sd = float(np.std(arr)) if arr.size > 1 else 0.0
        if not np.isfinite(sd) or sd < 1e-12:
            z = np.zeros_like(arr)
        else:
            z = (arr - mu) / sd
            z = np.clip(z, -3.0, 3.0)
        z_by_metric[metric] = z.tolist()

    denom = float(sum(abs(float(w)) for w in weights.values()))
    denom = max(denom, 1.0)

    for i, cand in enumerate(candidates):
        pool_score = 0.0
        for metric, weight in weights.items():
            pool_score += float(weight) * float(z_by_metric[metric][i])
        pool_score /= denom

        cand["aggregate_score_pool"] = float(pool_score)
        cand["score"] = float(pool_score + float(cand.get("diversity_bonus", 0.0)))
