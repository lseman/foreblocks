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
from typing import Any, Dict, List, Optional

import torch

from ..utils.training import reset_model_parameters
from .stats_reporting import (
    append_whatif_estimates,
    mean_std,
    save_csv,
    save_json,
)

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def run_multi_fidelity_search(
    trainer,
    train_loader,
    val_loader,
    test_loader,
    *,
    num_candidates: int = 10,
    search_epochs: int = 10,
    final_epochs: int = 100,
    max_samples: int = 32,
    top_k: int = 5,
    max_workers: Optional[int] = None,
    collect_stats: bool = False,
    parallelism_levels=None,
    est_overhead_per_task: float = 0.0,
    est_fixed_overhead_phase1: float = 0.0,
    est_fixed_overhead_phase3: float = 0.0,
    benchmark_phase1_workers=None,
    benchmark_phase1_candidates: Optional[int] = None,
    stats_dir: str = "search_stats",
    run_name: Optional[str] = None,
    logger=None,
    retrain_final_from_scratch: bool = True,
    discrete_arch_threshold: float = 0.3,
    use_amp: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Run the complete multi-fidelity DARTS search pipeline.

    Args:
        trainer:          :class:`~darts.trainer.DARTSTrainer` instance.
        train_loader:     Training DataLoader.
        val_loader:       Validation DataLoader.
        test_loader:      Test DataLoader used in phase 5.
        num_candidates:   Total random architectures to evaluate (phase 1).
        search_epochs:    DARTS epochs for each top-k candidate (phase 3).
        final_epochs:     Final training epochs (phase 5).
        max_samples:      Zero-cost evaluation sample budget.
        top_k:            Candidates to advance from phase 1 to phase 3.
        max_workers:      Thread-pool size for phase 1 (``None`` = auto).
        collect_stats:    Save JSON/CSV timing artefacts.
        parallelism_levels: Worker counts for what-if estimates.
        stats_dir:        Output directory when ``collect_stats=True``.
        run_name:         Human-readable run label (auto-generated if ``None``).
        logger:           Optional Python logger (defaults to ``NASLogger``).
        retrain_final_from_scratch: Re-initialise model weights before phase 5.
        discrete_arch_threshold: Threshold for ``derive_discrete_architecture``.

    Returns:
        Dict with keys: ``final_model``, ``candidates``, ``top_candidates``,
        ``trained_candidates``, ``best_candidate``, ``final_results``,
        ``search_config``, and optionally ``stats``.
    """
    # ── Setup ──────────────────────────────────────────────────────────────
    if logger is None:
        logger = logging.getLogger("NASLogger")

    if parallelism_levels is None:
        cpu = os.cpu_count() or 8
        parallelism_levels = sorted(set([1, 2, 4, 8, cpu]))
    else:
        parallelism_levels = list(parallelism_levels)

    if benchmark_phase1_workers is not None:
        benchmark_phase1_workers = list(benchmark_phase1_workers)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or f"multifidelity_{ts}"
    out_base = os.path.join(stats_dir, run_id) if collect_stats else None
    if out_base is not None:
        os.makedirs(out_base, exist_ok=True)

    sys_info = _build_sys_info(
        run_id=run_id,
        parallelism_levels=parallelism_levels,
        max_workers=max_workers,
        num_candidates=num_candidates,
        search_epochs=search_epochs,
        final_epochs=final_epochs,
        max_samples=max_samples,
        top_k=top_k,
    )

    logger.info("Starting multi-fidelity DARTS search")
    logger.info(
        f"candidates={num_candidates}, search_epochs={search_epochs}, "
        f"final_epochs={final_epochs}, top_k={top_k}, max_samples={max_samples}, "
        f"max_workers={max_workers or 'auto'}"
    )

    phase_summary: Dict[str, Any] = {}
    per_candidate_rows: List[List] = []
    whatif_rows: List[List] = []
    bench_rows: List[List] = []

    # ── (Optional) Phase 1 benchmark across worker counts ─────────────────
    phase1_benchmark_results: List[Dict] = []
    if collect_stats and benchmark_phase1_workers:
        phase1_benchmark_results, bench_rows = _run_phase1_benchmark(
            trainer=trainer,
            val_loader=val_loader,
            max_samples=max_samples,
            workers=benchmark_phase1_workers,
            n_candidates=int(benchmark_phase1_candidates or min(num_candidates, 20)),
            run_id=run_id,
            logger=logger,
        )

    # ── Phase 1: parallel zero-cost evaluation ─────────────────────────────
    logger.info("Phase 1: generating + zero-cost evaluating candidates (parallel)")
    phase1_task_times: List[float] = []

    def _generate_and_eval(cid: int) -> Dict[str, Any]:
        return trainer._evaluate_search_candidate(
            candidate_id=cid,
            val_loader=val_loader,
            max_samples=max_samples,
            num_batches=1,
            include_timing=True,
        )

    def _on_phase1(r, completed):
        if r.get("success"):
            phase1_task_times.append(float(r.get("phase1_dt", 0.0)))
            logger.info(
                f"[P1] {completed}/{num_candidates} id={r.get('candidate_id')} "
                f"score={r.get('score', 0.0):.4f} ops={len(r.get('selected_ops', []))} "
                f"dt={r.get('phase1_dt', 0.0):.3f}s"
            )
        else:
            logger.info(f"[P1] {completed}/{num_candidates} failed")

    t_p1_0 = time.perf_counter()
    candidates = trainer._run_parallel_candidate_collection(
        num_candidates=num_candidates,
        candidate_fn=_generate_and_eval,
        max_workers=max_workers,
        on_result=_on_phase1,
        error_log_fn=lambda e: logger.warning(f"[P1] future error: {e}"),
    )
    t_p1 = time.perf_counter() - t_p1_0

    # -- Sequential fallback if parallel returned nothing --
    if not candidates and num_candidates > 0:
        logger.warning("Phase 1 parallel returned 0 results, retrying sequentially.")
        candidates, extra_t = _sequential_fallback(
            trainer, num_candidates, _generate_and_eval, phase1_task_times, logger
        )
        t_p1 += extra_t
        if not candidates:
            raise RuntimeError("Phase 1 produced zero successful candidates.")

    p1_mean, p1_std = mean_std(phase1_task_times)
    phase_summary["phase1"] = {
        "wall_time_sec": float(t_p1),
        "num_success": len(candidates),
        "num_total": num_candidates,
        "task_mean_sec": p1_mean,
        "task_std_sec": p1_std,
        "task_min_sec": float(min(phase1_task_times)) if phase1_task_times else 0.0,
        "task_max_sec": float(max(phase1_task_times)) if phase1_task_times else 0.0,
    }
    logger.info(f"Phase 1 done: {len(candidates)}/{num_candidates} (wall={t_p1:.3f}s)")

    if collect_stats:
        whatif = append_whatif_estimates(
            phase="phase1",
            run_id=run_id,
            work_times=phase1_task_times,
            parallelism_levels=parallelism_levels,
            overhead_per_task=est_overhead_per_task,
            fixed_overhead=est_fixed_overhead_phase1,
            whatif_rows=whatif_rows,
        )
        phase_summary["phase1"]["whatif_estimates"] = whatif

    # ── Phase 2: select top-k ─────────────────────────────────────────────
    logger.info(f"Phase 2: selecting top-{top_k} candidates")
    t_p2_0 = time.perf_counter()
    top_candidates = trainer._select_top_candidates(candidates, top_k)
    top_k_eff = len(top_candidates)
    t_p2 = time.perf_counter() - t_p2_0

    phase_summary["phase2"] = {"wall_time_sec": float(t_p2), "top_k_eff": top_k_eff}
    for i, c in enumerate(top_candidates):
        logger.info(
            f"[P2] {i + 1}: score={c['score']:.4f} "
            f"ops={len(c.get('selected_ops', []))} "
            f"hidden={c.get('hidden_dim')}"
        )
    if top_k_eff == 0:
        raise RuntimeError("Phase 2 selected zero candidates.")

    # ── Phase 3: short DARTS training for each top candidate ──────────────
    logger.info(
        f"Phase 3: training {top_k_eff} candidates ({search_epochs} epochs each)"
    )
    t_p3_0 = time.perf_counter()

    trained_candidates: List[Dict] = []
    trained_non_derived: List[Dict] = []
    p3_task_times: List[float] = []
    p3_search_times: List[float] = []
    p3_derive_times: List[float] = []

    for i, cand in enumerate(top_candidates):
        cid = cand.get("candidate_id", -1)
        logger.info(f"[P3] training {i + 1}/{top_k_eff} (id={cid})")

        t_c0 = time.perf_counter()

        t_s0 = time.perf_counter()
        search_results = trainer.train_darts_model(
            model=cand["model"],
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=search_epochs,
            use_swa=False,
            use_amp=use_amp,
        )
        t_search = time.perf_counter() - t_s0

        trained_non_derived.append(
            {
                "model": copy.deepcopy(search_results["model"]),
                "val_loss": search_results["best_val_loss"],
                "candidate": cand,
                "search_results": search_results,
            }
        )

        t_d0 = time.perf_counter()
        derived = trainer.derive_final_architecture(search_results["model"])
        val_loss = trainer._evaluate_model(derived, val_loader)
        t_derive = time.perf_counter() - t_d0
        t_total = time.perf_counter() - t_c0

        p3_task_times.append(float(t_total))
        p3_search_times.append(float(t_search))
        p3_derive_times.append(float(t_derive))

        trained_candidates.append(
            {
                "model": derived,
                "val_loss": float(val_loss),
                "candidate": cand,
                "search_results": search_results,
            }
        )
        logger.info(
            f"[P3] id={cid} val_loss={val_loss:.6f} "
            f"dt_total={t_total:.3f}s (train={t_search:.3f}s "
            f"derive+eval={t_derive:.3f}s)"
        )

        if collect_stats:
            per_candidate_rows += _p3_csv_rows(
                run_id, cid, cand, t_total, t_search, t_derive, val_loss
            )

    t_p3 = time.perf_counter() - t_p3_0
    p3_m, p3_s = mean_std(p3_task_times)
    p3_tm, p3_ts = mean_std(p3_search_times)
    p3_dm, p3_ds = mean_std(p3_derive_times)
    phase_summary["phase3"] = {
        "wall_time_sec": float(t_p3),
        "task_mean_sec": p3_m,
        "task_std_sec": p3_s,
        "train_mean_sec": p3_tm,
        "train_std_sec": p3_ts,
        "derive_eval_mean_sec": p3_dm,
        "derive_eval_std_sec": p3_ds,
    }
    if collect_stats:
        whatif = append_whatif_estimates(
            phase="phase3",
            run_id=run_id,
            work_times=p3_task_times,
            parallelism_levels=parallelism_levels,
            overhead_per_task=est_overhead_per_task,
            fixed_overhead=est_fixed_overhead_phase3,
            whatif_rows=whatif_rows,
        )
        phase_summary["phase3"]["whatif_estimates"] = whatif

    # ── Phase 4: select best candidate ────────────────────────────────────
    logger.info("Phase 4: selecting best candidate")
    t_p4_0 = time.perf_counter()
    if not trained_candidates:
        raise RuntimeError("Phase 3 produced zero trained candidates.")
    best_candidate = min(trained_candidates, key=lambda x: x["val_loss"])
    t_p4 = time.perf_counter() - t_p4_0
    phase_summary["phase4"] = {"wall_time_sec": float(t_p4)}
    logger.info(
        f"[P4] best val_loss={best_candidate['val_loss']:.6f} "
        f"ops={best_candidate['candidate'].get('selected_ops')}"
    )

    # ── Phase 5: full final training ──────────────────────────────────────
    logger.info("Phase 5: training final model")
    t_p5_0 = time.perf_counter()

    final_model = copy.deepcopy(best_candidate["model"])
    final_conf = getattr(final_model, "get_config", lambda: {})()
    final_discrete_arch: Dict = {}
    if hasattr(final_model, "derive_discrete_architecture"):
        try:
            final_discrete_arch = final_model.derive_discrete_architecture(
                threshold=discrete_arch_threshold
            )
        except Exception as exc:
            logger.warning(f"[P5] discrete arch derivation failed: {exc}")

    modules_reset = 0
    if retrain_final_from_scratch:
        modules_reset = reset_model_parameters(final_model)
        logger.info(f"[P5] re-initialised {modules_reset} modules")

    final_results = trainer.train_final_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=final_epochs,
        learning_rate=5e-4,
        weight_decay=1e-5,
        use_amp=use_amp,
    )

    # Quick training-curve plot
    from ..evaluation.plotting import plot_training_curve

    curve_path = (
        os.path.join(out_base, "final_model_training.pdf")
        if out_base
        else "final_model_training.pdf"
    )
    plot_training_curve(
        final_results["train_losses"],
        final_results["val_losses"],
        title="Final Model Training Progress",
        save_path=curve_path,
    )

    t_p5 = time.perf_counter() - t_p5_0
    phase_summary["phase5"] = {"wall_time_sec": float(t_p5)}

    # ── Build stats & persist ─────────────────────────────────────────────
    total_wall = sum(
        phase_summary[p]["wall_time_sec"]
        for p in ["phase1", "phase2", "phase3", "phase4", "phase5"]
    )
    phase_summary["total"] = {"wall_time_sec": float(total_wall)}

    stats_payload = _build_stats_payload(
        sys_info=sys_info,
        phase_summary=phase_summary,
        phase1_benchmark_results=phase1_benchmark_results,
        top_candidates=top_candidates,
        best_candidate=best_candidate,
    )

    if collect_stats and out_base:
        _persist_stats(
            out_base=out_base,
            run_id=run_id,
            stats_payload=stats_payload,
            per_candidate_rows=per_candidate_rows,
            whatif_rows=whatif_rows,
            bench_rows=bench_rows,
            logger=logger,
        )

    logger.info(
        "Phase wall-times (s): "
        + ", ".join(
            f"{p}={phase_summary[p]['wall_time_sec']:.3f}"
            for p in ["phase1", "phase2", "phase3", "phase4", "phase5"]
        )
        + f" | total={total_wall:.3f}"
    )

    # ── Return search summary ─────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "final_model": final_results["model"],
        "candidates": candidates,
        "top_candidates": top_candidates,
        "trained_candidates": trained_candidates,
        "best_candidate": best_candidate,
        "final_results": final_results,
        "final_config": final_conf,
        "final_discrete_architecture": final_discrete_arch,
        "search_config": {
            "num_candidates": num_candidates,
            "search_epochs": search_epochs,
            "final_epochs": final_epochs,
            "top_k": top_k_eff,
            "max_samples": max_samples,
            "max_workers": max_workers,
            "retrain_final_from_scratch": bool(retrain_final_from_scratch),
            "discrete_arch_threshold": float(discrete_arch_threshold),
        },
        "trained_non_derived_candidates": trained_non_derived,
        "final_reset_modules": int(modules_reset),
    }
    if collect_stats:
        summary["stats"] = stats_payload
        summary["stats_dir"] = out_base

    trainer.search_history.append(summary)
    trainer.final_model = final_results["model"]

    logger.info("Multi-fidelity search completed")
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sequential_fallback(trainer, num_candidates, eval_fn, task_times, logger):
    """Sequential retry when parallel phase-1 returns nothing."""
    seq_t0 = time.perf_counter()
    found = []
    for cid in range(num_candidates):
        try:
            res = eval_fn(cid)
            if res.get("success"):
                found.append(res)
                task_times.append(float(res.get("phase1_dt", 0.0)))
                logger.info(f"[P1 fallback] id={res.get('candidate_id')} ok")
            else:
                logger.warning(f"[P1 fallback] id={cid} success=False")
        except Exception as exc:
            logger.warning(f"[P1 fallback] id={cid} failed: {exc}")
    return found, time.perf_counter() - seq_t0


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
        results.append(
            {
                "workers": w,
                "ncand": n_candidates,
                "wall_time_sec": wall,
                "task_mean_sec": m,
                "task_std_sec": s,
            }
        )
        rows.append([run_id, w, n_candidates, wall, m, s])
        logger.info(
            f"[P1 bench] workers={w}: wall={wall:.3f}s mean={m:.3f}s std={s:.3f}s"
        )
    return results, rows


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


def bilevel_lr_sensitivity(
    trainer,
    model_factory,
    train_loader,
    val_loader,
    *,
    model_lrs=(1e-4, 3e-4, 1e-3, 3e-3),
    arch_lrs=(3e-4, 1e-3, 3e-3, 1e-2),
    seeds=(0, 1, 2),
    epochs: int = 30,
    save_csv_path: Optional[str] = None,
):
    """
    Grid-search over (``model_lr``, ``arch_lr``, ``seed``) configurations.

    For each combination, a fresh model is created via ``model_factory()``,
    trained with DARTS bilevel optimisation, then the derived architecture is
    evaluated on ``val_loader``.

    Args:
        trainer:        :class:`~darts.trainer.DARTSTrainer` instance.
        model_factory:  Callable ``() -> model`` (no args) that returns a
                        freshly-initialised model placed on the correct device.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        model_lrs:      Iterable of model learning rates to sweep.
        arch_lrs:       Iterable of architecture learning rates to sweep.
        seeds:          Iterable of random seeds to average across.
        epochs:         DARTS training epochs per configuration.
        save_csv_path:  If provided, write a CSV summary to this path.

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``model_lr``, ``arch_lr``, ``seed``, ``best_val_loss_mixed``,
        ``val_loss_derived``, ``train_time_s``, ``health_score``,
        ``avg_identity_dominance``.
    """
    import random as _random

    import numpy as np
    import pandas as pd

    results = []

    for mlr in model_lrs:
        for alr in arch_lrs:
            for s in seeds:
                torch.manual_seed(s)
                np.random.seed(s)
                _random.seed(s)

                model = model_factory().to(trainer.device)
                out = trainer.train_darts_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    model_learning_rate=mlr,
                    arch_learning_rate=alr,
                    use_bilevel_optimization=True,
                    verbose=False,
                )

                derived = trainer.derive_final_architecture(out["model"])
                derived_val = trainer._evaluate_model(derived, val_loader)

                health_last = None
                if out.get("diversity_scores"):
                    health_last = out["diversity_scores"][-1]

                results.append(
                    {
                        "model_lr": mlr,
                        "arch_lr": alr,
                        "seed": s,
                        "best_val_loss_mixed": float(out["best_val_loss"]),
                        "val_loss_derived": float(derived_val),
                        "train_time_s": float(out["training_time"]),
                        "health_score": None
                        if not health_last
                        else float(health_last["health_score"]),
                        "avg_identity_dominance": None
                        if not health_last
                        else float(health_last["avg_identity_dominance"]),
                    }
                )

    df = pd.DataFrame(results)
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df
