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


def _resolve_phase3_rung_epochs(
    *,
    search_epochs: int,
    min_epoch_budget: int,
    reduction_factor: int,
    explicit,
) -> list[int]:
    """Build monotonically increasing ASHA rung budgets ending at search_epochs."""
    max_epochs = max(1, int(search_epochs))
    if explicit:
        rung_epochs = sorted({
            max(1, min(max_epochs, int(v))) for v in explicit if int(v) > 0
        })
        if not rung_epochs:
            return [max_epochs]
        if rung_epochs[-1] != max_epochs:
            rung_epochs.append(max_epochs)
        return rung_epochs

    budgets: list[int] = []
    cur = max(1, min(int(min_epoch_budget), max_epochs))
    while cur < max_epochs:
        budgets.append(int(cur))
        nxt = max(cur + 1, int(cur * reduction_factor))
        if nxt >= max_epochs:
            break
        cur = nxt
    budgets.append(max_epochs)
    return sorted({int(x) for x in budgets})


def _run_phase1_benchmark(
    *, trainer, val_loader, max_samples, workers, n_candidates, run_id, logger
):
    """Run phase-1 timing benchmark at multiple worker counts."""
    results, rows = [], []
    logger.info(
        f"Phase 1 benchmark: workers={workers}, candidates_per_run={n_candidates}"
    )
    for w in workers:
        t0 = time.perf_counter()
        task_times = []

        def _task(cid):
            r = trainer._evaluate_search_candidate(
                candidate_id=cid,
                val_loader=val_loader,
                max_samples=max_samples,
                num_batches=1,
                include_timing=True,
            )
            return float(r.get("phase1_dt", 0.0))

        with concurrent.futures.ThreadPoolExecutor(max_workers=w) as ex:
            for dt in ex.map(_task, range(n_candidates)):
                task_times.append(dt)

        wall = time.perf_counter() - t0
        m, s = mean_std(task_times)
        results.append({
            "workers": w,
            "ncand": n_candidates,
            "wall_time_sec": wall,
            "task_mean_sec": m,
            "task_std_sec": s,
        })
        rows.append([run_id, w, n_candidates, wall, m, s])
        logger.info(
            f"[P1 bench] workers={w}: wall={wall:.3f}s mean={m:.3f}s std={s:.3f}s"
        )
    return results, rows


