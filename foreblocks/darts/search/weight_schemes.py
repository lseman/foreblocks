"""Weight-scheme generation and stability utilities for zero-cost ablations."""

from typing import Dict

import numpy as np


def weights_uniform(base: Dict[str, float]) -> Dict[str, float]:
    return {k: (1.0 if v >= 0 else -1.0) for k, v in base.items()}


def weights_family_subsets(base: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    grad = ["grasp", "fisher", "snip", "jacobian", "sensitivity"]
    act = ["naswot", "activation_diversity", "zennas"]
    complexity = ["params", "flops", "conditioning"]
    syn = ["synflow"]
    return {
        "subset_grad": {k: base[k] for k in grad if k in base},
        "subset_act": {k: base[k] for k in act if k in base},
        "subset_complexity": {k: base[k] for k in complexity if k in base},
        "subset_synflow": {k: base[k] for k in syn if k in base},
        "subset_pos_only": {k: v for k, v in base.items() if v > 0},
        "subset_no_penalties": {
            k: v
            for k, v in base.items()
            if k not in {"params", "flops", "conditioning"}
        },
    }


def weights_leave_one_out(base: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    out = {}
    for key in list(base.keys()):
        w = dict(base)
        w.pop(key, None)
        out[f"loo_minus_{key}"] = w
    return out


def sample_random_weights_around(
    base: Dict[str, float], sigma: float, seed: int
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    out = {}
    for key, w0 in base.items():
        scale = max(abs(w0), 1e-6)
        out[key] = float(rng.normal(w0, sigma * scale))
    return out


def build_weight_schemes(
    base_weights: Dict[str, float],
    n_random: int = 20,
    random_sigma: float = 0.25,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    schemes: Dict[str, Dict[str, float]] = {}
    schemes["baseline"] = dict(base_weights)
    schemes["uniform"] = weights_uniform(base_weights)
    schemes.update(weights_family_subsets(base_weights))
    schemes.update(weights_leave_one_out(base_weights))
    for i in range(n_random):
        schemes[f"rand_{i:02d}"] = sample_random_weights_around(
            base_weights, random_sigma, seed + i
        )
    return schemes


def ranks_desc(scores: np.ndarray) -> np.ndarray:
    """Ranks with 1 = best."""
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


def spearman_from_scores(a: np.ndarray, b: np.ndarray) -> float:
    ra = ranks_desc(a).astype(np.float64)
    rb = ranks_desc(b).astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra**2).sum()) * np.sqrt((rb**2).sum())
    return float((ra * rb).sum() / denom) if denom > 0 else 0.0


def topk_overlap_from_scores(a: np.ndarray, b: np.ndarray, k: int) -> float:
    ra = ranks_desc(a)
    rb = ranks_desc(b)
    top_a = set(np.where(ra <= k)[0].tolist())
    top_b = set(np.where(rb <= k)[0].tolist())
    return float(len(top_a & top_b) / max(k, 1))
