"""Shared metric normalization and scoring utilities for DARTS search."""

from typing import Dict

import numpy as np


def normalize_metric_value(metric: str, value: float) -> float:
    """Normalize metric values on comparable scales while preserving sign."""
    # SynFlow is expected as a raw positive score; apply exactly one log transform here.
    if metric in {"synflow", "params", "flops", "zennas", "activation_diversity"}:
        normalized = np.log1p(max(float(value), 0.0))
        return 0.0 if np.isnan(normalized) else float(normalized)

    if metric in {"naswot", "jacobian"}:
        normalized = np.sign(value) * np.log1p(abs(float(value)))
        return 0.0 if np.isnan(normalized) else float(normalized)

    if metric == "conditioning":
        return float(min(max(float(value), 0.0), 30.0) / 30.0)

    return float(value)


def score_from_metrics(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Compute weighted aggregate score from raw metrics."""
    total_score = 0.0
    total_weight = 0.0
    consumed_weight_keys = set()

    for metric, value in metrics.items():
        weight_key = metric
        if weight_key not in weights:
            if metric == "activation_diversity" and "zennas" in weights:
                weight_key = "zennas"
            elif metric == "zennas" and "activation_diversity" in weights:
                weight_key = "activation_diversity"
            else:
                continue
        if weight_key in consumed_weight_keys:
            continue
        consumed_weight_keys.add(weight_key)

        weight = float(weights[weight_key])
        if not np.isfinite(value):
            continue

        normalized = normalize_metric_value(metric, float(value))
        total_score += normalized * weight
        total_weight += abs(weight)

    return float(total_score / max(total_weight, 1.0))
