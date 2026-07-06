"""
Multi-fidelity architecture search pipeline.

Phases:
1. Parallel zero-cost evaluation of ``num_candidates`` random architectures.
2. Select top-*k* candidates by aggregate score.
3. Short DARTS training + architecture derivation for each top candidate.
4. Select the best derived model by validation loss.
5. Full final training of the best model.

Public entry-point: :func:`run_multi_fidelity_search`.
"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime
import logging
import os
import time
from typing import Any

import torch

from ..utils.training import reset_model_parameters
from .candidate_scoring import rescore_candidates_poolwise
from .stats_reporting import append_whatif_estimates, mean_std, save_csv, save_json


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def _p3_csv_rows(run_id, cid, cand, t_total, t_search, t_derive, val_loss):
    """Return a pair of CSV rows (phase1 info + phase3 timings) for a candidate."""
    base = [
        run_id,
        None,
        cid,
        cand.get("score", 0.0),
        cand.get("hidden_dim"),
        len(cand.get("selected_ops", [])),
    ]
    p1_row = base + [cand.get("phase1_dt", 0.0), "", "", "", ""]
    p3_row = [
        run_id,
        "phase3",
        cid,
        cand.get("score", 0.0),
        cand.get("hidden_dim"),
        len(cand.get("selected_ops", [])),
    ] + ["", t_total, t_search, t_derive, float(val_loss)]
    p1_row[1] = "phase1"
    return [p1_row, p3_row]


def _build_sys_info(*, run_id, parallelism_levels, max_workers, **config_kwargs):
    return {
        "run_id": run_id,
        "timestamp_local": datetime.datetime.now().isoformat(),
        "cpu_count_os": os.cpu_count(),
        "torch_num_threads": torch.get_num_threads()
        if hasattr(torch, "get_num_threads")
        else None,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        "cuda_device_name": torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else None,
        "parallelism_levels": list(map(int, parallelism_levels)),
        "max_workers_used": max_workers,
        "config": config_kwargs,
    }


def _build_stats_payload(
    *, sys_info, phase_summary, phase1_benchmark_results, top_candidates, best_candidate
):
    top_table = [
        {
            "rank": i + 1,
            "candidate_id": int(c.get("candidate_id", -1)),
            "score": float(c.get("score", 0.0)),
            "hidden_dim": c.get("hidden_dim"),
            "num_ops": int(len(c.get("selected_ops", []))),
            "arch": f"{c.get('num_cells')}x{c.get('num_nodes')}",
            "phase1_dt": float(c.get("phase1_dt", 0.0)),
        }
        for i, c in enumerate(top_candidates)
    ]
    return {
        "system": sys_info,
        "phase_summary": phase_summary,
        "phase1_benchmark_results": phase1_benchmark_results,
        "top_candidates": top_table,
        "best_candidate": {
            "candidate_id": int(best_candidate["candidate"].get("candidate_id", -1)),
            "val_loss": float(best_candidate["val_loss"]),
            "score": float(best_candidate["candidate"].get("score", 0.0)),
            "hidden_dim": best_candidate["candidate"].get("hidden_dim"),
            "selected_ops": list(best_candidate["candidate"].get("selected_ops", [])),
        },
    }


def _persist_stats(
    *,
    out_base,
    run_id,
    stats_payload,
    per_candidate_rows,
    whatif_rows,
    bench_rows,
    logger,
):
    save_json(os.path.join(out_base, "stats.json"), stats_payload)
    save_csv(
        os.path.join(out_base, "per_candidate.csv"),
        header=[
            "run_id",
            "phase",
            "candidate_id",
            "score",
            "hidden_dim",
            "num_ops",
            "phase1_dt_sec",
            "phase3_total_dt_sec",
            "phase3_train_dt_sec",
            "phase3_derive_eval_dt_sec",
            "phase3_val_loss",
        ],
        rows=per_candidate_rows,
    )
    save_csv(
        os.path.join(out_base, "whatif_parallelism.csv"),
        header=["run_id", "phase", "workers", "est_wall_time_sec"],
        rows=whatif_rows,
    )
    if bench_rows:
        save_csv(
            os.path.join(out_base, "phase1_benchmark.csv"),
            header=[
                "run_id",
                "workers",
                "ncand",
                "wall_time_sec",
                "task_mean_sec",
                "task_std_sec",
            ],
            rows=bench_rows,
        )
    logger.info(f"Stats saved to: {out_base}")


# ---------------------------------------------------------------------------
# Bilevel LR Sensitivity Sweep
# ---------------------------------------------------------------------------


