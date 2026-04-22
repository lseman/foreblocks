"""
Robustness w.r.t. the initial operator-pool selection.

Instead of evaluating candidates from a single fixed operator set, this module
samples many random operator-pool subsets, runs constrained candidate
generation inside each pool, and aggregates the stability of selected
architectures across pools.

Public API
----------
- :func:`robust_initial_pool_over_op_pools`
"""

from __future__ import annotations

import concurrent.futures
import random
from typing import Any

import numpy as np

from .candidate_scoring import candidate_signature
from .scoring import score_from_metrics
from .weight_schemes import build_weight_schemes

# module-level alias kept for readability
_sig_from_cfg = candidate_signature


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def robust_initial_pool_over_op_pools(
    trainer,
    *,
    val_loader,
    # outer robustness (op-pool perturbations)
    n_pools: int = 25,
    pool_size_range: tuple = (4, 10),
    pool_seed: int = 0,
    # inner candidate sampling (per pool)
    num_candidates: int = 30,
    top_k: int = 10,
    max_samples: int = 32,
    num_batches: int = 1,
    seed: int = 0,
    max_workers: int | None = None,
    # optional: weight-scheme robustness inside each pool
    use_weight_schemes: bool = False,
    n_random: int = 50,
    random_sigma: float = 0.25,
    robustness_mode: str = "topk_freq",  # "topk_freq" | "avg_rank" | "worst_rank"
    topk_ref: int | None = None,
    # candidate knobs
    min_ops: int = 2,
    max_ops: int | None = None,
    cell_range: tuple = (1, 2),
    node_range: tuple = (2, 4),
    hidden_dim_choices: list[int] | None = None,
    require_identity: bool = True,
) -> dict[str, Any]:
    """
    Evaluate robustness of architecture selection w.r.t. the operator pool.

    Samples *n_pools* random subsets of ``trainer.all_ops``, runs constrained
    zero-cost candidate search within each pool, and aggregates how often an
    architecture signature ends up in the top-*k* across pools.

    Args:
        trainer:          :class:`~darts.trainer.DARTSTrainer` instance.
        val_loader:       Validation DataLoader used by zero-cost metrics.
        n_pools:          Number of distinct operator-pool subsets to sample.
        pool_size_range:  ``(min, max)`` number of ops per pool (Identity optional).
        pool_seed:        Random seed for pool sampling order.
        num_candidates:   Architecture candidates to evaluate within each pool.
        top_k:            Pool-level top-k threshold.
        max_samples:      Zero-cost evaluation sample budget.
        num_batches:      Batches per zero-cost evaluation.
        seed:             Random seed for candidate generation.
        max_workers:      Thread-pool size for parallel candidate evaluation.
        use_weight_schemes: Also perturb scoring weights inside each pool.
        n_random:         Random weight-scheme count (when ``use_weight_schemes``).
        random_sigma:     Std-dev for random weight perturbations.
        robustness_mode:  Ranking strategy: ``"topk_freq"``, ``"avg_rank"``,
                          or ``"worst_rank"``.
        topk_ref:         Override top-k reference count for reporting.
        min_ops / max_ops: Candidate op-count bounds.
        cell_range:       ``(min, max)`` cells per candidate.
        node_range:       ``(min, max)`` nodes per cell.
        hidden_dim_choices: Allowed hidden dimensions (defaults to ``trainer.hidden_dims``).
        require_identity: Always include the Identity operation.

    Returns:
        Dict with keys ``selected``, ``robustness_table``, ``pool_results``,
        ``op_pools``, ``config``.
    """
    if hidden_dim_choices is None:
        hidden_dim_choices = list(trainer.hidden_dims)
    if max_ops is None:
        max_ops = len(trainer.all_ops)
    if topk_ref is None:
        topk_ref = top_k

    # ── Sample operator pools ─────────────────────────────────────────────
    py_rng_pools = random.Random(pool_seed)
    base_ops: list[str] = list(trainer.all_ops)

    def _sample_pool() -> list[str]:
        ops_no_id = [op for op in base_ops if op != "Identity"]
        lo, hi = pool_size_range
        k = py_rng_pools.randint(
            max(1, lo - (1 if require_identity else 0)),
            max(1, hi - (1 if require_identity else 0)),
        )
        picked = py_rng_pools.sample(ops_no_id, k=min(k, len(ops_no_id)))
        return ["Identity"] + picked if require_identity else picked

    op_pools: list[list[str]] = []
    seen: set = set()
    while len(op_pools) < n_pools:
        p = tuple(_sample_pool())
        if p not in seen:
            seen.add(p)
            op_pools.append(list(p))

    # ── Per-pool evaluation ────────────────────────────────────────────────
    all_pool_results: list[dict] = []
    # signature → accumulator
    agg: dict[str, dict] = {}

    for pool_idx, allowed_ops in enumerate(op_pools):
        py_rng = random.Random(seed + pool_idx)

        def _make_config() -> dict[str, Any]:
            return trainer._make_candidate_config(
                py_rng,
                allowed_ops,
                hidden_dim_choices,
                cell_range,
                node_range,
                min_ops=min_ops,
                max_ops=max_ops,
                require_identity=require_identity,
            )

        def _eval_one(candidate_id: int) -> dict[str, Any]:  # noqa: ARG001
            cfg = _make_config()
            try:
                model = trainer._build_candidate_model(cfg)
                out = trainer.evaluate_zero_cost_metrics_raw(
                    model=model,
                    dataloader=val_loader,
                    max_samples=max_samples,
                    num_batches=num_batches,
                )
                raw = out.get("raw_metrics") or {}
                if not raw:
                    return {"success": False, "error": "empty_raw_metrics"}
                return {
                    "success": True,
                    **cfg,
                    "raw_metrics": raw,
                    "base_weights": dict(out.get("base_weights", {})),
                }
            except Exception as exc:
                return {"success": False, "error": str(exc)}

        # Run inner evaluation (parallel)
        candidates: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [ex.submit(_eval_one, i) for i in range(num_candidates)]
            for fut in concurrent.futures.as_completed(futs):
                r = fut.result()
                if r.get("success", False):
                    candidates.append(r)

        if not candidates:
            all_pool_results.append(
                {
                    "pool_idx": pool_idx,
                    "allowed_ops": allowed_ops,
                    "success": False,
                    "error": "all_candidates_failed",
                }
            )
            continue

        # ── Scoring ───────────────────────────────────────────────────────
        base_weights = dict(candidates[0].get("base_weights", {}))

        if use_weight_schemes:
            schemes = build_weight_schemes(
                base_weights=base_weights,
                n_random=n_random,
                random_sigma=random_sigma,
                seed=seed + pool_idx,
            )
            scheme_names = list(schemes.keys())
            baseline_name = "baseline" if "baseline" in schemes else scheme_names[0]
            weights_for_pool = schemes[baseline_name]
        else:
            scheme_names = ["baseline"]
            weights_for_pool = base_weights

        scores = np.array(
            [
                score_from_metrics(c["raw_metrics"], weights_for_pool)
                for c in candidates
            ],
            dtype=np.float64,
        )

        order = np.argsort(-scores, kind="mergesort")
        k_eff = min(top_k, len(candidates))
        top_idx = order[:k_eff].tolist()

        pool_top: list[dict] = []
        for local_rank, i in enumerate(top_idx, start=1):
            c = candidates[i]
            sig = _sig_from_cfg(c)
            entry = {
                "pool_idx": pool_idx,
                "allowed_ops": allowed_ops,
                "signature": sig,
                "baseline_score": float(scores[i]),
                "rank_in_pool": int(local_rank),
                "selected_ops": list(c["selected_ops"]),
                "hidden_dim": int(c["hidden_dim"]),
                "num_cells": int(c["num_cells"]),
                "num_nodes": int(c["num_nodes"]),
            }
            pool_top.append(entry)

            st = agg.setdefault(
                sig,
                {"count_in_topk": 0, "ranks": [], "scores": [], "example": entry},
            )
            st["count_in_topk"] += 1
            st["ranks"].append(entry["rank_in_pool"])
            st["scores"].append(entry["baseline_score"])

        all_pool_results.append(
            {
                "pool_idx": pool_idx,
                "allowed_ops": allowed_ops,
                "success": True,
                "topk": pool_top,
                "scheme_names": scheme_names,
            }
        )

    # ── Aggregate robustness across pools ─────────────────────────────────
    table: list[dict] = []
    for sig, st in agg.items():
        ranks = np.array(st["ranks"], dtype=np.float64)
        scores_arr = np.array(st["scores"], dtype=np.float64)
        table.append(
            {
                "signature": sig,
                "topk_freq": int(st["count_in_topk"]),
                "avg_rank": float(ranks.mean()) if len(ranks) else float("inf"),
                "worst_rank": int(ranks.max()) if len(ranks) else 10**9,
                "avg_score": float(scores_arr.mean())
                if len(scores_arr)
                else float("-inf"),
                "score_std": float(scores_arr.std(ddof=0)) if len(scores_arr) else 0.0,
                "example_selected_ops": st["example"]["selected_ops"],
                "example_hidden_dim": st["example"]["hidden_dim"],
                "example_num_cells": st["example"]["num_cells"],
                "example_num_nodes": st["example"]["num_nodes"],
            }
        )

    if not table:
        raise RuntimeError("No successful pools produced any top-k entries.")

    if robustness_mode == "topk_freq":
        table.sort(key=lambda r: (r["topk_freq"], r["avg_score"]), reverse=True)
    elif robustness_mode == "avg_rank":
        table.sort(key=lambda r: (r["avg_rank"], -r["avg_score"]))
    elif robustness_mode == "worst_rank":
        table.sort(key=lambda r: (r["worst_rank"], -r["avg_score"]))
    else:
        raise ValueError(
            f"Unknown robustness_mode='{robustness_mode}'. Use 'topk_freq', 'avg_rank', or 'worst_rank'."
        )

    selected = table[: min(top_k, len(table))]

    return {
        "selected": selected,
        "robustness_table": table,
        "pool_results": all_pool_results,
        "op_pools": op_pools,
        "config": {
            "n_pools": n_pools,
            "pool_size_range": pool_size_range,
            "pool_seed": pool_seed,
            "num_candidates": num_candidates,
            "top_k": top_k,
            "max_samples": max_samples,
            "num_batches": num_batches,
            "seed": seed,
            "use_weight_schemes": use_weight_schemes,
            "n_random": n_random,
            "random_sigma": random_sigma,
            "robustness_mode": robustness_mode,
            "topk_ref": topk_ref,
            "min_ops": min_ops,
            "max_ops": max_ops,
            "cell_range": cell_range,
            "node_range": node_range,
            "hidden_dim_choices": hidden_dim_choices,
            "require_identity": require_identity,
        },
    }
