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

from ..config import DARTSTrainConfig
from ..utils.training import reset_model_parameters
from .candidate_scoring import rescore_candidates_poolwise
from .phase_utils import _resolve_phase3_rung_epochs, _run_phase1_benchmark
from .stats import _build_stats_payload, _build_sys_info, _p3_csv_rows, _persist_stats
from .stats_reporting import append_whatif_estimates, mean_std, save_csv, save_json


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
    final_patience: int | None = None,
    max_samples: int = 32,
    top_k: int = 5,
    max_workers: int | None = None,
    collect_stats: bool = False,
    parallelism_levels=None,
    est_overhead_per_task: float = 0.0,
    est_fixed_overhead_phase1: float = 0.0,
    est_fixed_overhead_phase3: float = 0.0,
    benchmark_phase1_workers=None,
    benchmark_phase1_candidates: int | None = None,
    stats_dir: str = "search_stats",
    run_name: str | None = None,
    logger=None,
    retrain_final_from_scratch: bool = True,
    discrete_arch_threshold: float = 0.3,
    use_amp: bool = False,
    phase1_progress: bool = False,
    **kwargs,
) -> dict[str, Any]:
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
        final_patience:   Early-stopping patience for phase-5 final training.
                          ``None`` derives ~20% of ``final_epochs`` (min 5).
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
        phase1_progress:  If True, show a tqdm bar as phase-1 candidates finish.

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
        parallelism_levels = sorted({1, 2, 4, 8, cpu})
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

    effective_phase1_workers = max_workers
    if effective_phase1_workers is None and str(
        getattr(trainer, "device", "")
    ).startswith("cuda"):
        # A single GPU is the bottleneck for zero-cost metrics. Running several
        # candidate threads only queues them behind the shared CUDA lock and
        # makes the run look stuck when one candidate is slow.
        effective_phase1_workers = 1

    phase1_rescore_mode = str(kwargs.pop("phase1_rescore_mode", "pool")).lower()
    phase3_reduction_factor = max(2, int(kwargs.pop("phase3_reduction_factor", 2)))
    phase3_min_epoch_budget = max(1, int(kwargs.pop("phase3_min_epoch_budget", 2)))
    phase3_rung_epochs = kwargs.pop("phase3_rung_epochs", None)
    phase3_verbose = bool(kwargs.pop("phase3_verbose", True))
    final_train_max_batches = kwargs.pop("final_train_max_batches", None)
    final_val_max_batches = kwargs.pop("final_val_max_batches", None)
    final_test_max_batches = kwargs.pop("final_test_max_batches", None)
    final_use_swa = bool(kwargs.pop("final_use_swa", False))
    final_compile = bool(kwargs.pop("final_compile", False))

    # Training-config kwargs forwarded to every train_darts_model call in phase 3.
    train_kwargs: dict[str, Any] = {
        k: kwargs.pop(k)
        for k in (
            "op_gdas",
            "moe_balance_weight",
            "transformer_exploration_weight",
            "beta_darts_weight",
            "arch_grad_ema_beta",
            "hessian_penalty_weight",
            "state_mix_ortho_reg_weight",
            "edge_diversity_weight",
            "edge_usage_balance_weight",
            "edge_identity_cap",
            "edge_identity_cap_weight",
            "max_train_batches",
            "max_val_batches",
        )
        if k in kwargs
    }
    phase3_train_max_batches = kwargs.pop("phase3_train_max_batches", None)
    phase3_val_max_batches = kwargs.pop("phase3_val_max_batches", None)
    if phase3_train_max_batches is not None:
        train_kwargs["max_train_batches"] = int(phase3_train_max_batches)
    if phase3_val_max_batches is not None:
        train_kwargs["max_val_batches"] = int(phase3_val_max_batches)

    phase_summary: dict[str, Any] = {}
    per_candidate_rows: list[list] = []
    whatif_rows: list[list] = []
    bench_rows: list[list] = []

    # ── (Optional) Phase 1 benchmark across worker counts ─────────────────
    phase1_benchmark_results: list[dict] = []
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
    phase1_msg = "Phase 1: generating + zero-cost evaluating candidates (parallel)"
    print(f"\n=== {phase1_msg} ===")
    logger.info(phase1_msg)
    phase1_task_times: list[float] = []

    def _make_threadsafe_eval_loader(base_loader):
        """Create a per-task eval loader to avoid sharing DataLoader iterators across threads."""
        bs = getattr(base_loader, "batch_size", None)
        if bs is None:
            bs = 32
        return torch.utils.data.DataLoader(
            base_loader.dataset,
            batch_size=bs,
            shuffle=False,
            pin_memory=bool(getattr(base_loader, "pin_memory", False)),
            num_workers=0,
            persistent_workers=False,
            drop_last=False,
        )

    def _generate_and_eval(cid: int) -> dict[str, Any]:
        local_val_loader = _make_threadsafe_eval_loader(val_loader)
        return trainer._evaluate_search_candidate(
            candidate_id=cid,
            val_loader=local_val_loader,
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
        max_workers=effective_phase1_workers,
        on_result=_on_phase1,
        progress=phase1_progress,
        progress_desc="Phase 1 candidates",
        error_log_fn=lambda e: logger.warning(
            f"[P1] future error ({type(e).__name__}): {e!r}"
        ),
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
    if phase1_rescore_mode == "pool":
        for cand in candidates:
            cand["score_raw"] = float(cand.get("score", 0.0))
        rescore_candidates_poolwise(candidates)
        logger.info("[P1] applied pool-relative candidate rescoring")
        phase_summary["phase1"]["rescore_mode"] = "pool"
    else:
        phase_summary["phase1"]["rescore_mode"] = "fixed"
    phase1_done = f"Phase 1 done: {len(candidates)}/{num_candidates} (wall={t_p1:.3f}s)"
    print(phase1_done)
    logger.info(phase1_done)

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
    phase2_msg = f"Phase 2: selecting top-{top_k} candidates"
    print(f"\n=== {phase2_msg} ===")
    logger.info(phase2_msg)
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

    # ── Phase 3: ASHA-style short DARTS training for promoted candidates ──
    rung_epochs = _resolve_phase3_rung_epochs(
        search_epochs=search_epochs,
        min_epoch_budget=phase3_min_epoch_budget,
        reduction_factor=phase3_reduction_factor,
        explicit=phase3_rung_epochs,
    )
    phase3_msg = (
        f"Phase 3: ASHA training {top_k_eff} candidates across rungs {rung_epochs}"
    )
    print(f"\n=== {phase3_msg} ===")
    logger.info(phase3_msg)
    t_p3_0 = time.perf_counter()

    trained_candidates: list[dict] = []
    trained_non_derived: list[dict] = []
    p3_task_times: list[float] = []
    p3_search_times: list[float] = []
    p3_derive_times: list[float] = []
    asha_states = [
        {
            "candidate": cand,
            "model": cand["model"],
            "epochs_trained": 0,
            "search_results": None,
            "mixed_val_loss": float("inf"),
            "promotion_score": float("inf"),
            "round_history": [],
        }
        for cand in top_candidates
    ]
    phase3_rounds: list[dict[str, Any]] = []

    for rung_idx, rung_epoch in enumerate(rung_epochs):
        if not asha_states:
            break

        rung_msg = (
            f"[P3][rung {rung_idx + 1}/{len(rung_epochs)}] "
            f"budget={rung_epoch} active={len(asha_states)}"
        )
        print(rung_msg)
        logger.info(rung_msg)
        round_rows: list[dict[str, Any]] = []

        for state_idx, state in enumerate(asha_states):
            cand = state["candidate"]
            cid = cand.get("candidate_id", -1)
            delta_epochs = int(rung_epoch) - int(state["epochs_trained"])
            if delta_epochs <= 0:
                continue

            train_msg = (
                f"[P3] training rung {rung_idx + 1} candidate "
                f"{state_idx + 1}/{len(asha_states)} (id={cid}) for +{delta_epochs} epochs "
                f"| arch_mode={cand.get('arch_mode', 'encoder_decoder')} "
                f"| families={cand.get('selected_families', [])} "
                f"| attn={cand.get('transformer_self_attention_type', 'auto')} "
                f"| ffn={cand.get('transformer_ffn_variant', 'swiglu')}"
            )
            print(train_msg)
            logger.info(train_msg)

            t_c0 = time.perf_counter()
            t_s0 = time.perf_counter()
            search_results = trainer.train_darts_model(
                model=state["model"],
                train_loader=train_loader,
                val_loader=val_loader,
                train_config=DARTSTrainConfig(
                    epochs=delta_epochs,
                    use_swa=False,
                    use_amp=use_amp,
                    verbose=phase3_verbose,
                    **train_kwargs,
                ),
                # Phase 3 only reads best_val_loss; skip the extra final-metrics
                # validation pass on every candidate×rung call.
                compute_metrics=False,
            )
            t_search = time.perf_counter() - t_s0
            t_total = time.perf_counter() - t_c0

            state["model"] = search_results["model"]
            state["search_results"] = search_results
            state["epochs_trained"] = int(rung_epoch)
            state["mixed_val_loss"] = float(search_results["best_val_loss"])
            state["promotion_score"] = float(search_results["best_val_loss"])
            state["round_history"].append({
                "rung": int(rung_idx + 1),
                "epoch_budget": int(rung_epoch),
                "delta_epochs": int(delta_epochs),
                "mixed_val_loss": float(search_results["best_val_loss"]),
                "train_time_sec": float(t_search),
                "total_time_sec": float(t_total),
            })
            round_rows.append({
                "candidate_id": cid,
                "epoch_budget": int(rung_epoch),
                "mixed_val_loss": float(search_results["best_val_loss"]),
                "train_time_sec": float(t_search),
                "total_time_sec": float(t_total),
            })
            result_msg = (
                f"[P3] completed rung {rung_idx + 1} candidate id={cid} "
                f"| mixed_val_loss={float(search_results['best_val_loss']):.6f} "
                f"| train_time={t_search:.3f}s total={t_total:.3f}s"
            )
            print(result_msg)
            logger.info(result_msg)
            p3_task_times.append(float(t_total))
            p3_search_times.append(float(t_search))

        asha_states.sort(
            key=lambda s: (
                float(s.get("promotion_score", float("inf"))),
                -float(s["candidate"].get("score", 0.0)),
            )
        )

        if rung_idx < len(rung_epochs) - 1 and len(asha_states) > 1:
            keep = max(
                1,
                (len(asha_states) + phase3_reduction_factor - 1)
                // phase3_reduction_factor,
            )
            promoted_ids = [
                s["candidate"].get("candidate_id", -1) for s in asha_states[:keep]
            ]
            promote_msg = (
                f"[P3][rung {rung_idx + 1}] promoting {keep}/{len(asha_states)} "
                f"candidates: {promoted_ids}"
            )
            print(promote_msg)
            logger.info(promote_msg)
            phase3_rounds.append({
                "rung_index": int(rung_idx + 1),
                "epoch_budget": int(rung_epoch),
                "num_candidates": len(round_rows),
                "num_promoted": int(keep),
                "results": round_rows,
                "promoted_candidate_ids": promoted_ids,
            })
            asha_states = asha_states[:keep]
        else:
            phase3_rounds.append({
                "rung_index": int(rung_idx + 1),
                "epoch_budget": int(rung_epoch),
                "num_candidates": len(round_rows),
                "num_promoted": len(round_rows),
                "results": round_rows,
                "promoted_candidate_ids": [
                    s["candidate"].get("candidate_id", -1) for s in asha_states
                ],
            })

    for state in asha_states:
        cand = state["candidate"]
        cid = cand.get("candidate_id", -1)
        if state["search_results"] is None:
            continue

        trained_non_derived.append({
            "model": copy.deepcopy(state["search_results"]["model"]),
            "val_loss": float(state["mixed_val_loss"]),
            "candidate": cand,
            "search_results": state["search_results"],
            "phase3_round_history": list(state["round_history"]),
            "epochs_trained": int(state["epochs_trained"]),
        })

        t_d0 = time.perf_counter()
        derived = trainer.derive_final_architecture(state["search_results"]["model"])
        val_loss = trainer._evaluate_model(derived, val_loader)
        t_derive = time.perf_counter() - t_d0
        p3_derive_times.append(float(t_derive))

        trained_candidates.append({
            "model": derived,
            "val_loss": float(val_loss),
            "candidate": cand,
            "search_results": state["search_results"],
            "phase3_round_history": list(state["round_history"]),
            "epochs_trained": int(state["epochs_trained"]),
            "mixed_val_loss": float(state["mixed_val_loss"]),
        })
        finalist_msg = (
            f"[P3][finalist] id={cid} derived_val_loss={val_loss:.6f} "
            f"mixed_val_loss={state['mixed_val_loss']:.6f} "
            f"epochs={state['epochs_trained']} derive={t_derive:.3f}s"
        )
        print(finalist_msg)
        logger.info(finalist_msg)

        if collect_stats:
            total_train = float(
                sum(x.get("total_time_sec", 0.0) for x in state["round_history"])
            )
            total_search = float(
                sum(x.get("train_time_sec", 0.0) for x in state["round_history"])
            )
            per_candidate_rows += _p3_csv_rows(
                run_id, cid, cand, total_train, total_search, t_derive, val_loss
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
        "rung_epochs": [int(x) for x in rung_epochs],
        "reduction_factor": int(phase3_reduction_factor),
        "max_train_batches": train_kwargs.get("max_train_batches"),
        "max_val_batches": train_kwargs.get("max_val_batches"),
        "rounds": phase3_rounds,
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
    phase4_msg = "Phase 4: selecting best candidate"
    print(f"\n=== {phase4_msg} ===")
    logger.info(phase4_msg)
    t_p4_0 = time.perf_counter()
    if not trained_candidates:
        raise RuntimeError("Phase 3 produced zero trained candidates.")
    best_candidate = min(trained_candidates, key=lambda x: x["val_loss"])
    t_p4 = time.perf_counter() - t_p4_0
    phase_summary["phase4"] = {"wall_time_sec": float(t_p4)}
    logger.info(
        f"[P4] best val_loss={best_candidate['val_loss']:.6f} "
        f"ops={best_candidate['candidate'].get('selected_ops')} "
        f"families={best_candidate['candidate'].get('selected_families', [])}"
    )

    # ── Phase 5: full final training ──────────────────────────────────────
    phase5_msg = "Phase 5: training final model"
    print(f"\n=== {phase5_msg} ===")
    logger.info(phase5_msg)
    t_p5_0 = time.perf_counter()

    final_model = copy.deepcopy(best_candidate["model"])
    final_conf = getattr(final_model, "get_config", lambda: {})()
    final_discrete_arch: dict = {}
    if hasattr(final_model, "derive_discrete_architecture"):
        try:
            final_discrete_arch = final_model.derive_discrete_architecture(
                threshold=discrete_arch_threshold
            )
        except Exception as exc:
            logger.warning(f"[P5] discrete arch derivation failed: {exc}")

    # ── Phase 5: print selected architecture (deduped via ArchitectureInspector) ──
    from ..architecture.inspector import ArchitectureInspector

    inspector = ArchitectureInspector(final_model)
    summary = inspector.summary()
    sel = best_candidate.get("candidate", {})

    p5_lines: list[str] = [
        "[P5] Selected architecture before final training:",
        f"[P5]   candidate_id={sel.get('candidate_id', 'N/A')} | arch_mode={summary.get('arch_mode', 'N/A')} | "
        f"hidden_dim={sel.get('hidden_dim', summary.get('hidden_dim', 'N/A'))} | "
        f"cells={summary.get('cells', 'N/A')} | nodes={summary.get('nodes', 'N/A')}",
        f"[P5]   selected_ops={sel.get('selected_ops', 'N/A')}",
        f"[P5]   selected_families={sel.get('selected_families', 'N/A')}",
        f"[P5]   transformer={inspector.transformer_summary()}",
        f"[P5]   normalization={summary.get('normalization', 'unknown')}",
        f"[P5]   {summary.get('decomposition_encoder', '')}",
        f"[P5]   {summary.get('decomposition_decoder', '')}",
        f"[P5]   encoder={summary.get('encoder_type', 'N/A')}",
        f"[P5]   decoder={summary.get('decoder_type', 'N/A')}",
        f"[P5]   cross_attention={summary.get('cross_attention', 'N/A')}",
        f"[P5]   cross_position={summary.get('cross_position', 'N/A')}",
        f"[P5]   decoder_style={summary.get('decoder_style', 'N/A')}",
        f"[P5]   decoder_query={summary.get('decoder_query', 'N/A')}",
    ]
    enc_sa = summary.get("attention_encoder")
    dec_sa = summary.get("attention_decoder")
    if enc_sa and enc_sa not in {"not used", "not applicable"}:
        p5_lines.append(f"[P5]   encoder_self_attention={enc_sa}")
        p5_lines.append(f"[P5]   encoder_attention_position={summary.get('attention_position_encoder', 'N/A')}")
        p5_lines.append(f"[P5]   encoder_tokenizer={summary.get('tokenizer', 'N/A')}")
        p5_lines.append(f"[P5]   encoder_ffn={summary.get('ffn_encoder', 'N/A')}")
    if dec_sa and dec_sa not in {"not used", "not applicable"}:
        p5_lines.append(f"[P5]   decoder_self_attention={dec_sa}")
        p5_lines.append(f"[P5]   decoder_attention_position={summary.get('attention_position_decoder', 'N/A')}")
        p5_lines.append(f"[P5]   decoder_ffn={summary.get('ffn_decoder', 'N/A')}")
    for cell_line in summary.get("cells", []):
        p5_lines.append(f"[P5]   {cell_line}")
    if final_discrete_arch:
        for k in sorted(final_discrete_arch.keys()):
            v = final_discrete_arch[k]
            if k in {"encoder", "decoder"} and isinstance(v, dict) and "type" in v:
                v_pretty = dict(v)
                if str(v_pretty.get("type", "")).startswith("op_"):
                    resolved = summary.get(f"{k}_type", v_pretty["type"])
                    v_pretty["type"] = resolved
                p5_lines.append(f"[P5]   {k}={v_pretty}")
            else:
                p5_lines.append(f"[P5]   {k}={v}")
    print("\n" + "\n".join(p5_lines))

    for line in p5_lines:
        logger.info(line)

    modules_reset = 0
    if retrain_final_from_scratch:
        modules_reset = reset_model_parameters(final_model)
        logger.info(f"[P5] re-initialised {modules_reset} modules")

    # Early-stopping patience for final training. Default scales with
    # final_epochs (~20%, min 5) so it actually fires on a converged run
    # instead of always running all epochs.
    if final_patience is None:
        resolved_patience = max(5, int(round(final_epochs * 0.2)))
    else:
        resolved_patience = int(final_patience)

    final_results = trainer.train_final_model(
        model=final_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=final_epochs,
        learning_rate=5e-4,
        weight_decay=1e-5,
        patience=resolved_patience,
        use_amp=use_amp,
        use_swa=final_use_swa,
        max_train_batches=(
            None if final_train_max_batches is None else int(final_train_max_batches)
        ),
        max_val_batches=(
            None if final_val_max_batches is None else int(final_val_max_batches)
        ),
        max_test_batches=(
            None if final_test_max_batches is None else int(final_test_max_batches)
        ),
        compile_model=final_compile,
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
    summary: dict[str, Any] = {
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
            "phase1_rescore_mode": phase1_rescore_mode,
            "phase3_reduction_factor": int(phase3_reduction_factor),
            "phase3_min_epoch_budget": int(phase3_min_epoch_budget),
            "phase3_rung_epochs": [int(x) for x in rung_epochs],
            "phase3_verbose": bool(phase3_verbose),
            "final_train_max_batches": (
                None
                if final_train_max_batches is None
                else int(final_train_max_batches)
            ),
            "final_val_max_batches": (
                None if final_val_max_batches is None else int(final_val_max_batches)
            ),
            "final_test_max_batches": (
                None if final_test_max_batches is None else int(final_test_max_batches)
            ),
            "final_use_swa": bool(final_use_swa),
            "final_compile": bool(final_compile),
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

