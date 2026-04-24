"""Candidate configuration helpers for DARTS search.

These helpers isolate random search-space sampling and operation-family
selection logic away from the central DARTSTrainer orchestrator.
"""

from __future__ import annotations

import math
import random
from typing import Any

from ..config import DEFAULT_OP_FAMILIES


def normalize_op_families(
    op_families: dict[str, list[str]] | None,
    allowed_ops: list[str],
) -> dict[str, list[str]]:
    """Filter the op-family map to the current allowed operation set."""
    family_source = op_families or DEFAULT_OP_FAMILIES
    allowed_set = set(allowed_ops)
    normalized: dict[str, list[str]] = {}

    for family_name, ops in family_source.items():
        filtered = [op for op in ops if op in allowed_set]
        if filtered or str(family_name).lower() == "ssm":
            normalized[str(family_name).lower()] = filtered

    if not normalized:
        normalized = {"mlp": [op for op in allowed_ops if op != "Identity"]}

    return normalized


def _resolve_candidate_families(
    rng,
    allowed_ops: list[str],
    op_families: dict[str, list[str]],
    family_range: tuple[int, int],
) -> tuple[dict[str, list[str]], list[str], list[str]]:
    allowed_set = set(allowed_ops)
    family_space: dict[str, list[str]] = {}
    for family_name, ops in op_families.items():
        filtered = [op for op in ops if op in allowed_set]
        if filtered:
            family_space[family_name] = filtered

    family_names = list(family_space.keys())
    if not family_names:
        fallback_ops = [op for op in allowed_ops if op != "Identity"]
        return {"mlp": fallback_ops}, ["mlp"], ["mlp"]

    min_families = min(family_range[0], len(family_names))
    max_families = min(family_range[1], len(family_names))
    if max_families < min_families:
        max_families = min_families

    selected_families = rng.sample(
        family_names, k=rng.randint(min_families, max_families)
    )
    op_families = [name for name in selected_families if family_space.get(name)]

    if not op_families:
        non_ssm = [name for name, ops in family_space.items() if ops]
        if non_ssm:
            fallback = rng.choice(non_ssm)
            if fallback not in selected_families:
                selected_families.append(fallback)
            op_families = [fallback]

    return family_space, selected_families, op_families


def _select_family_operations(
    rng,
    *,
    family_space: dict[str, list[str]],
    op_families: list[str],
    allowed_ops: list[str],
    require_identity: bool,
    min_ops: int,
    max_ops: int | None,
) -> tuple[list[str], dict[str, list[str]]]:
    guaranteed_ops: list[str] = []
    family_choices: dict[str, list[str]] = {}

    for family_name in op_families:
        family_pool = [
            op for op in family_space.get(family_name, []) if op != "Identity"
        ]
        if not family_pool:
            family_pool = list(family_space.get(family_name, []))
        if not family_pool:
            continue
        chosen_op = rng.choice(family_pool)
        guaranteed_ops.append(chosen_op)
        family_choices[family_name] = [chosen_op]

    guaranteed_ops = list(dict.fromkeys(guaranteed_ops))
    ops_no_id = list(
        dict.fromkeys(
            op
            for family_name in op_families
            for op in family_space.get(family_name, [])
            if op != "Identity"
        )
    )
    if not ops_no_id:
        ops_no_id = [op for op in allowed_ops if op != "Identity"]

    max_ops_local = min(
        max_ops or len(ops_no_id) + (1 if require_identity else 0),
        len(ops_no_id) + (1 if require_identity else 0),
    )
    min_required = len(guaranteed_ops) + (1 if require_identity else 0)
    min_ops_local = max(min_ops, min_required)
    if max_ops_local < min_ops_local:
        max_ops_local = min_ops_local

    n_ops = rng.randint(min_ops_local, max_ops_local)
    non_identity_target = max(0, n_ops - (1 if require_identity else 0))
    extra_pool = [op for op in ops_no_id if op not in guaranteed_ops]
    extra_count = min(
        len(extra_pool), max(0, non_identity_target - len(guaranteed_ops))
    )
    extra_ops = rng.sample(extra_pool, k=extra_count) if extra_count > 0 else []
    selected_non_identity = guaranteed_ops + extra_ops

    if not selected_non_identity and ops_no_id:
        selected_non_identity = [rng.choice(ops_no_id)]

    selected_ops = (
        ["Identity"] + selected_non_identity
        if require_identity
        else selected_non_identity
    )
    return selected_ops, family_choices


def make_candidate_config(
    rng,
    allowed_ops: list[str],
    hidden_dim_choices: list[int],
    cell_range: tuple[int, int],
    node_range: tuple[int, int],
    *,
    op_families: dict[str, list[str]] | None = None,
    family_range: tuple[int, int] = (1, 3),
    min_ops: int = 2,
    max_ops: int | None = None,
    require_identity: bool = True,
    edge_to_op_target: float = 1.0,
    edge_to_op_max_ratio: float = 1.8,
    arch_modes: list[str] | None = None,
    attention_variants: list[str] | None = None,
    ffn_variants: list[str] | None = None,
) -> dict[str, Any]:
    rng = rng or random
    op_families = normalize_op_families(op_families, allowed_ops)
    family_space, selected_families, op_families = _resolve_candidate_families(
        rng, allowed_ops, op_families, family_range
    )
    selected_ops, family_choices = _select_family_operations(
        rng,
        family_space=family_space,
        op_families=op_families,
        allowed_ops=allowed_ops,
        require_identity=require_identity,
        min_ops=min_ops,
        max_ops=max_ops,
    )

    node_candidates = list(range(int(node_range[0]), int(node_range[1]) + 1))
    op_budget = max(len(selected_ops) - (1 if require_identity else 0), 1)
    desired_edges = max(1, int(round(op_budget * max(float(edge_to_op_target), 0.2))))
    max_edges = max(
        3, int(math.ceil(op_budget * max(float(edge_to_op_max_ratio), 1.0)))
    )
    feasible = [
        n for n in node_candidates if (n * (n - 1) // 2) <= max_edges
    ] or node_candidates
    weights = [
        max(
            math.exp(-abs((n * (n - 1) // 2) - desired_edges) / max(desired_edges, 1))
            / (1.0 + 0.08 * float(n * (n - 1) // 2)),
            1e-6,
        )
        for n in feasible
    ]
    try:
        num_nodes = rng.choices(feasible, weights=weights, k=1)[0]
    except Exception:
        num_nodes = rng.choice(feasible)

    arch_mode = rng.choice(arch_modes) if arch_modes else "encoder_decoder"
    transformer_self_attention_type = (
        rng.choice(attention_variants) if attention_variants else "auto"
    )
    transformer_ffn_variant = (
        rng.choice(ffn_variants) if "attention" in selected_families else "auto"
    )
    family_choices = dict(family_choices)
    if "attention" in selected_families:
        family_choices["attention_variant"] = [transformer_self_attention_type]
        family_choices["attention_ffn"] = [transformer_ffn_variant]

    return {
        "selected_ops": selected_ops,
        "selected_families": selected_families,
        "family_choices": family_choices,
        "hidden_dim": rng.choice(hidden_dim_choices),
        "num_cells": rng.randint(cell_range[0], cell_range[1]),
        "num_nodes": int(num_nodes),
        "arch_mode": arch_mode,
        "transformer_self_attention_type": transformer_self_attention_type,
        "transformer_ffn_variant": transformer_ffn_variant,
        "transformer_use_moe": transformer_ffn_variant == "moe",
    }
