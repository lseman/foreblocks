"""
Zero-cost weight-scheme ablation search.

Evaluates raw zero-cost metrics *once* per candidate, then re-scores each
candidate under many weighting schemes (baseline, uniform, leave-one-out,
random perturbations) to quantify how sensitive the candidate ranking is to
the choice of metric weights.

Public entry-point: :func:`ablation_weight_search`.
"""

from __future__ import annotations

import concurrent.futures
import os
import threading
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from .scoring import score_from_metrics
from .weight_schemes import (
    build_weight_schemes,
    ranks_desc as _ranks_desc,
    spearman_from_scores as _spearman,
    topk_overlap_from_scores as _topk_overlap,
)


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def ablation_weight_search(
    trainer,
    train_loader,
    val_loader,
    test_loader=None,
    *,
    num_candidates: int = 20,
    max_samples: int = 32,
    num_batches: int = 1,
    top_k: int = 5,
    max_workers: int | None = None,
    n_random: int = 50,
    random_sigma: float = 0.25,
    seed: int = 0,
    save_dir: str = ".",
    save_prefix: str = "zc_weight_ablation",
) -> dict[str, Any]:
    """
    Run a lightweight ablation study that evaluates raw zero-cost metrics
    ONCE per candidate, then re-scores candidates under many weight schemes.

    Produces:
    * pandas tables and matplotlib plots saved to *save_dir*
    * the full raw data needed for paper tables / reproducibility

    Args:
        trainer:        :class:`~darts.trainer.DARTSTrainer` instance.
        train_loader:   (unused, kept for API symmetry).
        val_loader:     Validation DataLoader used for zero-cost evaluation.
        test_loader:    (unused, kept for API symmetry).
        num_candidates: Number of random architectures to sample.
        max_samples:    Max samples per zero-cost evaluation.
        num_batches:    Batches per zero-cost evaluation.
        top_k:          Top-k used for overlap statistics.
        max_workers:    Thread-pool workers (``None`` = auto).
        n_random:       Random weight perturbation schemes.
        random_sigma:   Std-dev of random weight perturbations.
        seed:           Global random seed.
        save_dir:       Output directory for tables/plots.
        save_prefix:    File-name prefix for saved artefacts.

    Returns:
        Dict with keys: ``candidates``, ``scheme_names``, ``schemes``,
        ``score_matrix``, ``rank_matrix``, ``tables``, ``random_analysis``,
        ``artifacts``, ``config``.
    """
    import random as _random

    import pandas as pd

    os.makedirs(save_dir, exist_ok=True)

    print("Weight ablation search (zero-cost weighting).")
    print(
        f"  candidates={num_candidates} | max_samples={max_samples} | "
        f"num_batches={num_batches}"
    )
    print(f"  top_k={top_k} | n_random={n_random} | sigma={random_sigma} | seed={seed}")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Phase 1: generate candidates + compute raw metrics (parallel)
    # ------------------------------------------------------------------
    def _eval_one(candidate_id: int) -> dict[str, Any]:
        cfg = trainer._make_candidate_config(
            _random,
            trainer.all_ops,
            trainer.hidden_dims,
            (1, 2),
            (2, 4),
            min_ops=2,
            max_ops=len(trainer.all_ops),
            require_identity=True,
        )
        model = trainer._build_candidate_model(cfg)
        out = trainer.evaluate_zero_cost_metrics_raw(
            model=model,
            dataloader=val_loader,
            max_samples=max_samples,
            num_batches=num_batches,
        )
        return {
            "candidate_id": candidate_id,
            "success": True,
            **cfg,
            "raw_metrics": out["raw_metrics"],
            "success_rates": out.get("success_rates", {}),
            "errors": out.get("errors", {}),
            "base_weights": out.get("base_weights", {}),
        }

    candidates: list[dict[str, Any]] = []
    lock = threading.Lock()
    done_count = 0

    def _cb(fut):
        nonlocal done_count
        r = fut.result()
        with lock:
            done_count += 1
            status = "ok" if r.get("success") else "fail"
            print(
                f"  [{done_count:>3}/{num_candidates}] {status}  "
                f"id={r.get('candidate_id', -1)}"
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_eval_one, i) for i in range(num_candidates)]
        for f in futs:
            f.add_done_callback(_cb)
        for f in concurrent.futures.as_completed(futs):
            r = f.result()
            if r.get("success"):
                candidates.append(r)

    if not candidates:
        raise RuntimeError("All candidates failed in raw zero-cost evaluation.")

    print(f"\nRaw eval done: {len(candidates)}/{num_candidates} successful.")
    print("-" * 70)

    # ------------------------------------------------------------------
    # Phase 2: build schemes and score matrix
    # ------------------------------------------------------------------
    base_weights = dict(candidates[0].get("base_weights", {}))
    schemes = build_weight_schemes(
        base_weights=base_weights,
        n_random=n_random,
        random_sigma=random_sigma,
        seed=seed,
    )
    scheme_names = list(schemes.keys())
    N, S = len(candidates), len(scheme_names)

    score_mat = np.zeros((N, S), dtype=np.float64)
    for i, c in enumerate(candidates):
        for j, name in enumerate(scheme_names):
            score_mat[i, j] = score_from_metrics(c["raw_metrics"], schemes[name])

    for i, c in enumerate(candidates):
        c["scheme_scores"] = {scheme_names[j]: float(score_mat[i, j]) for j in range(S)}

    # ------------------------------------------------------------------
    # Phase 3: build analysis tables
    # ------------------------------------------------------------------
    baseline_idx = scheme_names.index("baseline") if "baseline" in scheme_names else 0
    baseline_scores = score_mat[:, baseline_idx]

    # Stability table
    stability_rows = [
        {
            "scheme": name,
            "spearman_vs_baseline": _spearman(baseline_scores, score_mat[:, j]),
            "topk_overlap_vs_baseline": _topk_overlap(
                baseline_scores, score_mat[:, j], top_k
            ),
        }
        for j, name in enumerate(scheme_names)
    ]
    df_stability = pd.DataFrame(stability_rows).sort_values(
        by=["spearman_vs_baseline", "topk_overlap_vs_baseline"], ascending=False
    )

    # Top-candidate summary
    baseline_rank = _ranks_desc(baseline_scores)
    order = np.argsort(baseline_rank)
    top_show = min(10, N)
    cand_rows = []
    for idx in order[:top_show]:
        c = candidates[idx]
        row = {
            "candidate_id": c["candidate_id"],
            "baseline_rank": int(baseline_rank[idx]),
            "hidden_dim": c["hidden_dim"],
            "num_cells": c["num_cells"],
            "num_nodes": c["num_nodes"],
            "num_ops": len(c["selected_ops"]),
            "selected_ops": ", ".join(c["selected_ops"]),
            "baseline_score": float(score_mat[idx, baseline_idx]),
        }
        if "uniform" in scheme_names:
            row["uniform_score"] = float(score_mat[idx, scheme_names.index("uniform")])
        cand_rows.append(row)
    df_top = pd.DataFrame(cand_rows)

    # LOO importance
    loo_names = [n for n in scheme_names if n.startswith("loo_minus_")]
    loo_rows = []
    for n in loo_names:
        j = scheme_names.index(n)
        loo_rows.append(
            {
                "metric_removed": n.replace("loo_minus_", ""),
                "spearman_vs_baseline": _spearman(baseline_scores, score_mat[:, j]),
                "topk_overlap_vs_baseline": _topk_overlap(
                    baseline_scores, score_mat[:, j], top_k
                ),
            }
        )
    df_loo = pd.DataFrame(loo_rows)
    if len(df_loo):
        df_loo["importance_spearman_drop"] = 1.0 - df_loo["spearman_vs_baseline"]
        df_loo["importance_topk_drop"] = 1.0 - df_loo["topk_overlap_vs_baseline"]
        df_loo = df_loo.sort_values(
            by=["importance_spearman_drop", "importance_topk_drop"], ascending=False
        )

    # Random-weight analysis
    rand_names = [n for n in scheme_names if n.startswith("rand_")]
    baseline_winner_idx = int(np.argmax(baseline_scores))
    winner_ranks = (
        np.array(
            [
                int(
                    _ranks_desc(score_mat[:, scheme_names.index(rn)])[
                        baseline_winner_idx
                    ]
                )
                for rn in rand_names
            ],
            dtype=np.int64,
        )
        if rand_names
        else np.array([], dtype=np.int64)
    )
    topk_freq = np.zeros(N, dtype=np.int64)
    for rn in rand_names:
        j = scheme_names.index(rn)
        topk_ids = np.where(_ranks_desc(score_mat[:, j]) <= top_k)[0]
        topk_freq[topk_ids] += 1

    # ------------------------------------------------------------------
    # Phase 4: plots
    # ------------------------------------------------------------------
    _make_plots(
        save_dir=save_dir,
        save_prefix=save_prefix,
        candidates=candidates,
        scheme_names=scheme_names,
        score_mat=score_mat,
        order=order,
        top_show=top_show,
        rand_names=rand_names,
        winner_ranks=winner_ranks,
        topk_freq=topk_freq,
        df_loo=df_loo,
        top_k=top_k,
    )

    # ------------------------------------------------------------------
    # Phase 5: persist tables
    # ------------------------------------------------------------------
    p_stab = os.path.join(save_dir, f"{save_prefix}_stability.csv")
    p_top = os.path.join(save_dir, f"{save_prefix}_top_candidates.csv")
    p_loo = os.path.join(save_dir, f"{save_prefix}_loo_importance.csv")
    df_stability.to_csv(p_stab, index=False)
    df_top.to_csv(p_top, index=False)
    print(f"Tables saved: {p_stab}, {p_top}")
    if len(df_loo):
        df_loo.to_csv(p_loo, index=False)
        print(f"  {p_loo}")

    return {
        "candidates": candidates,
        "scheme_names": scheme_names,
        "schemes": schemes,
        "score_matrix": score_mat,
        "rank_matrix": np.column_stack(
            [_ranks_desc(score_mat[:, j]) for j in range(S)]
        ),
        "tables": {
            "stability": df_stability,
            "top_candidates": df_top,
            "loo_importance": df_loo if len(df_loo) else None,
        },
        "random_analysis": {
            "baseline_winner_candidate_id": int(
                candidates[baseline_winner_idx]["candidate_id"]
            ),
            "baseline_winner_rank_under_random": winner_ranks.tolist(),
            "topk_frequency_under_random": topk_freq.tolist(),
        },
        "artifacts": {
            "rank_heatmap": os.path.join(save_dir, f"{save_prefix}_rank_heatmap.png"),
            "winner_rank_hist": os.path.join(
                save_dir, f"{save_prefix}_winner_rank_hist.png"
            ),
            "topk_freq": os.path.join(save_dir, f"{save_prefix}_topk_freq.png"),
            "loo_importance": os.path.join(
                save_dir, f"{save_prefix}_loo_importance.png"
            ),
            "stability_csv": p_stab,
            "top_candidates_csv": p_top,
            "loo_csv": p_loo if len(df_loo) else None,
        },
        "config": {
            "num_candidates": num_candidates,
            "max_samples": max_samples,
            "num_batches": num_batches,
            "top_k": top_k,
            "n_random": n_random,
            "random_sigma": random_sigma,
            "seed": seed,
        },
    }


# ---------------------------------------------------------------------------
# Internal: plot helpers
# ---------------------------------------------------------------------------


def _make_plots(
    *,
    save_dir,
    save_prefix,
    candidates,
    scheme_names,
    score_mat,
    order,
    top_show,
    rand_names,
    winner_ranks,
    topk_freq,
    df_loo,
    top_k,
) -> None:
    """Generate and save the four standard ablation plots."""

    def _savefig(path):
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {path}")

    N, S = score_mat.shape
    ranks_mat = np.column_stack([_ranks_desc(score_mat[:, j]) for j in range(S)])

    # 1. Rank heatmap
    scheme_keep = [
        s
        for s in [
            "baseline",
            "uniform",
            "subset_grad",
            "subset_act",
            "subset_complexity",
            "subset_no_penalties",
            "subset_pos_only",
        ]
        if s in scheme_names
    ] + rand_names[: min(6, len(rand_names))]
    scheme_keep = list(dict.fromkeys(scheme_keep))
    keep_idx = [scheme_names.index(s) for s in scheme_keep]
    sub_ranks = ranks_mat[order[:top_show]][:, keep_idx]

    plt.figure(figsize=(max(8, 0.8 * len(keep_idx)), max(4, 0.5 * top_show)))
    plt.imshow(sub_ranks, aspect="auto")
    plt.xticks(range(len(keep_idx)), scheme_keep, rotation=45, ha="right")
    plt.yticks(
        range(top_show),
        [int(candidates[i]["candidate_id"]) for i in order[:top_show]],
    )
    plt.xlabel("Weight scheme")
    plt.ylabel("Candidate id (baseline top)")
    plt.title("Candidate ranks across weight schemes (lower is better)")
    _savefig(os.path.join(save_dir, f"{save_prefix}_rank_heatmap.png"))

    # 2. Baseline-winner rank under random weights
    if len(winner_ranks):
        plt.figure(figsize=(8, 4))
        plt.hist(
            winner_ranks,
            bins=min(20, max(5, int(np.sqrt(len(winner_ranks))))),
            edgecolor="black",
        )
        plt.xlabel("Rank of baseline winner under random weights (1=best)")
        plt.ylabel("Count")
        plt.title("Baseline-winner rank under random weight perturbations")
        _savefig(os.path.join(save_dir, f"{save_prefix}_winner_rank_hist.png"))

    # 3. Top-k frequency
    if rand_names:
        freq_order = np.argsort(-topk_freq)
        show = min(10, N)
        plt.figure(figsize=(10, 4))
        plt.bar(range(show), topk_freq[freq_order[:show]])
        plt.xticks(
            range(show),
            [int(candidates[i]["candidate_id"]) for i in freq_order[:show]],
        )
        plt.xlabel("Candidate id")
        plt.ylabel(f"Times in top-{top_k} across {len(rand_names)} random schemes")
        plt.title(f"Top-{top_k} frequency under random weight perturbations")
        _savefig(os.path.join(save_dir, f"{save_prefix}_topk_freq.png"))

    # 4. LOO importance
    if len(df_loo):
        show = min(12, len(df_loo))
        df_plot = df_loo.head(show)
        plt.figure(figsize=(10, 4))
        plt.bar(range(show), df_plot["importance_spearman_drop"].to_numpy())
        plt.xticks(
            range(show), df_plot["metric_removed"].tolist(), rotation=45, ha="right"
        )
        plt.xlabel("Removed metric")
        plt.ylabel("Importance (1 – Spearman vs baseline)")
        plt.title("Leave-one-out importance of each zero-cost metric")
        _savefig(os.path.join(save_dir, f"{save_prefix}_loo_importance.png"))
