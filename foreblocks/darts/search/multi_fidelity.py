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
from .candidate_scoring import rescore_candidates_poolwise
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

    phase1_rescore_mode = str(kwargs.pop("phase1_rescore_mode", "pool")).lower()
    phase3_reduction_factor = max(2, int(kwargs.pop("phase3_reduction_factor", 2)))
    phase3_min_epoch_budget = max(1, int(kwargs.pop("phase3_min_epoch_budget", 2)))
    phase3_rung_epochs = kwargs.pop("phase3_rung_epochs", None)
    phase3_verbose = bool(kwargs.pop("phase3_verbose", True))

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
    phase1_msg = "Phase 1: generating + zero-cost evaluating candidates (parallel)"
    print(f"\n=== {phase1_msg} ===")
    logger.info(phase1_msg)
    phase1_task_times: List[float] = []

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

    def _generate_and_eval(cid: int) -> Dict[str, Any]:
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
        max_workers=max_workers,
        on_result=_on_phase1,
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

    trained_candidates: List[Dict] = []
    trained_non_derived: List[Dict] = []
    p3_task_times: List[float] = []
    p3_search_times: List[float] = []
    p3_derive_times: List[float] = []
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
    phase3_rounds: List[Dict[str, Any]] = []

    for rung_idx, rung_epoch in enumerate(rung_epochs):
        if not asha_states:
            break

        rung_msg = (
            f"[P3][rung {rung_idx + 1}/{len(rung_epochs)}] "
            f"budget={rung_epoch} active={len(asha_states)}"
        )
        print(rung_msg)
        logger.info(rung_msg)
        round_rows: List[Dict[str, Any]] = []

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
                epochs=delta_epochs,
                use_swa=False,
                use_amp=use_amp,
                verbose=phase3_verbose,
            )
            t_search = time.perf_counter() - t_s0
            t_total = time.perf_counter() - t_c0

            state["model"] = search_results["model"]
            state["search_results"] = search_results
            state["epochs_trained"] = int(rung_epoch)
            state["mixed_val_loss"] = float(search_results["best_val_loss"])
            state["promotion_score"] = float(search_results["best_val_loss"])
            state["round_history"].append(
                {
                    "rung": int(rung_idx + 1),
                    "epoch_budget": int(rung_epoch),
                    "delta_epochs": int(delta_epochs),
                    "mixed_val_loss": float(search_results["best_val_loss"]),
                    "train_time_sec": float(t_search),
                    "total_time_sec": float(t_total),
                }
            )
            round_rows.append(
                {
                    "candidate_id": cid,
                    "epoch_budget": int(rung_epoch),
                    "mixed_val_loss": float(search_results["best_val_loss"]),
                    "train_time_sec": float(t_search),
                    "total_time_sec": float(t_total),
                }
            )
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
            keep = max(1, (len(asha_states) + phase3_reduction_factor - 1) // phase3_reduction_factor)
            promoted_ids = [
                s["candidate"].get("candidate_id", -1) for s in asha_states[:keep]
            ]
            promote_msg = (
                f"[P3][rung {rung_idx + 1}] promoting {keep}/{len(asha_states)} "
                f"candidates: {promoted_ids}"
            )
            print(promote_msg)
            logger.info(promote_msg)
            phase3_rounds.append(
                {
                    "rung_index": int(rung_idx + 1),
                    "epoch_budget": int(rung_epoch),
                    "num_candidates": len(round_rows),
                    "num_promoted": int(keep),
                    "results": round_rows,
                    "promoted_candidate_ids": promoted_ids,
                }
            )
            asha_states = asha_states[:keep]
        else:
            phase3_rounds.append(
                {
                    "rung_index": int(rung_idx + 1),
                    "epoch_budget": int(rung_epoch),
                    "num_candidates": len(round_rows),
                    "num_promoted": len(round_rows),
                    "results": round_rows,
                    "promoted_candidate_ids": [
                        s["candidate"].get("candidate_id", -1) for s in asha_states
                    ],
                }
            )

    for state in asha_states:
        cand = state["candidate"]
        cid = cand.get("candidate_id", -1)
        if state["search_results"] is None:
            continue

        trained_non_derived.append(
            {
                "model": copy.deepcopy(state["search_results"]["model"]),
                "val_loss": float(state["mixed_val_loss"]),
                "candidate": cand,
                "search_results": state["search_results"],
                "phase3_round_history": list(state["round_history"]),
                "epochs_trained": int(state["epochs_trained"]),
            }
        )

        t_d0 = time.perf_counter()
        derived = trainer.derive_final_architecture(state["search_results"]["model"])
        val_loss = trainer._evaluate_model(derived, val_loader)
        t_derive = time.perf_counter() - t_d0
        p3_derive_times.append(float(t_derive))

        trained_candidates.append(
            {
                "model": derived,
                "val_loss": float(val_loss),
                "candidate": cand,
                "search_results": state["search_results"],
                "phase3_round_history": list(state["round_history"]),
                "epochs_trained": int(state["epochs_trained"]),
                "mixed_val_loss": float(state["mixed_val_loss"]),
            }
        )
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
    final_discrete_arch: Dict = {}
    if hasattr(final_model, "derive_discrete_architecture"):
        try:
            final_discrete_arch = final_model.derive_discrete_architecture(
                threshold=discrete_arch_threshold
            )
        except Exception as exc:
            logger.warning(f"[P5] discrete arch derivation failed: {exc}")

    def _norm_choice(model_obj) -> str:
        chosen = getattr(model_obj, "selected_norm", None)
        if chosen:
            return str(chosen)
        alpha = getattr(model_obj, "norm_alpha", None)
        if isinstance(alpha, torch.Tensor) and alpha.numel() >= 3:
            names = ["revin", "instance_norm", "identity"]
            idx = int(torch.argmax(alpha.detach()).item())
            return names[idx] if 0 <= idx < len(names) else f"norm_{idx}"
        return "unknown"

    def _decomp_choice(module_obj) -> str:
        if module_obj is None:
            return "not used"
        decomp = getattr(module_obj, "searchable_decomp", None)
        if decomp is None:
            return "disabled"
        logits = getattr(decomp, "alpha_logits", None)
        if not isinstance(logits, torch.Tensor) or logits.numel() == 0:
            return "enabled (weights unavailable)"
        modes = ["none", "moving_avg_trend", "seasonal_residual", "learnable_filter"]
        probs = torch.softmax(logits.detach(), dim=0)
        top_idx = int(torch.argmax(probs).item())
        top_w = float(probs[top_idx].item())
        mode = modes[top_idx] if 0 <= top_idx < len(modes) else f"mode_{top_idx}"
        return f"{mode} (weight={top_w:.3f})"

    def _cell_ops_summary(model_obj) -> List[str]:
        out: List[str] = []
        for ci, cell in enumerate(getattr(model_obj, "cells", [])):
            edge_ops: List[str] = []
            for edge in getattr(cell, "edges", []):
                op_name = None
                fixed_op = getattr(edge, "op", None)
                if fixed_op is not None:
                    op_name = type(fixed_op).__name__.replace("Op", "")
                if op_name is None:
                    edge_ops.append("mixed")
                else:
                    edge_ops.append(op_name)
            if edge_ops:
                counts: Dict[str, int] = {}
                for n in edge_ops:
                    counts[n] = counts.get(n, 0) + 1
                counts_txt = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
                out.append(f"cell_{ci}: {counts_txt}")
        return out

    def _enc_dec_choice(model_obj, role: str) -> str:
        item = final_discrete_arch.get(role)
        if isinstance(item, dict) and "type" in item:
            name = str(item["type"])
            if name.startswith("op_"):
                module_obj = getattr(model_obj, f"forecast_{role}", None)
                rnn_type = (
                    getattr(module_obj, "rnn_type", None)
                    if module_obj is not None
                    else None
                )
                if rnn_type:
                    return str(rnn_type)
            return name
        module_obj = getattr(model_obj, f"forecast_{role}", None)
        if module_obj is None:
            return "not used"
        rnn_type = getattr(module_obj, "rnn_type", None)
        if rnn_type:
            return str(rnn_type)
        rnn_obj = getattr(module_obj, "rnn", None)
        if rnn_obj is not None:
            return type(rnn_obj).__name__
        return type(module_obj).__name__

    def _self_attention_choice(model_obj, role: str) -> str:
        module_obj = getattr(model_obj, f"forecast_{role}", None)
        if module_obj is None:
            return "not used"

        chosen = _enc_dec_choice(model_obj, role).lower()
        if role == "encoder":
            if "transformer" not in chosen and "patch" not in chosen:
                return "not applicable"
        else:
            if "transformer" not in chosen:
                return "not applicable"

        submodule = getattr(module_obj, "rnn", None)
        if submodule is None:
            submodule = getattr(module_obj, "transformer", None)
        if submodule is not None:
            value = getattr(submodule, "self_attention_type", None)
            if isinstance(value, str) and value and value != "auto":
                return value
            layers = getattr(submodule, "layers", None)
            if layers and len(layers) > 0:
                first = layers[0]
                self_attn = None
                if isinstance(first, dict):
                    self_attn = first.get("self_attn")
                elif hasattr(first, "get"):
                    self_attn = first.get("self_attn")
                elif hasattr(first, "__contains__") and "self_attn" in first:
                    self_attn = first["self_attn"]
                if self_attn is not None:
                    attn_type = getattr(self_attn, "attention_type", None)
                    if isinstance(attn_type, str) and attn_type and attn_type != "auto":
                        return attn_type
                    attn_alphas = getattr(self_attn, "attn_alphas", None)
                    modes = getattr(
                        self_attn,
                        "MODES",
                        ("sdp", "linear", "probsparse", "cosine", "local"),
                    )
                    if isinstance(attn_alphas, torch.Tensor) and attn_alphas.numel() == len(
                        modes
                    ):
                        probs = torch.softmax(attn_alphas.detach(), dim=0)
                        top_idx = int(torch.argmax(probs).item())
                        if 0 <= top_idx < len(modes):
                            return str(modes[top_idx])

        direct = getattr(module_obj, "self_attention_type", None)
        if isinstance(direct, str) and direct:
            return direct

        return str(getattr(model_obj, "transformer_self_attention_type", "unknown"))

    def _encoder_patch_choice(model_obj) -> str:
        enc = getattr(model_obj, "forecast_encoder", None)
        if enc is None:
            return "not used"

        submodule = getattr(enc, "rnn", None)
        if submodule is None:
            submodule = getattr(enc, "transformer", None)
        if submodule is None:
            return "unknown"

        direct = getattr(submodule, "patching_mode", None)
        if isinstance(direct, str) and direct:
            if direct != "auto":
                return direct

        logits = getattr(submodule, "patch_alpha_logits", None)
        mode_names = getattr(submodule, "patch_mode_names", ("direct", "patch"))
        if isinstance(logits, torch.Tensor) and logits.numel() == len(mode_names):
            probs = torch.softmax(logits.detach(), dim=0)
            top_idx = int(torch.argmax(probs).item())
            if 0 <= top_idx < len(mode_names):
                return str(mode_names[top_idx])

        resolver = getattr(submodule, "resolve_patch_mode", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    def _self_attention_position_choice(model_obj, role: str) -> str:
        component = getattr(model_obj, f"forecast_{role}", None)
        if component is None:
            return "not used"
        submodule = getattr(component, "rnn", None)
        if submodule is None:
            submodule = getattr(component, "transformer", None)
        if submodule is None:
            return "unknown"
        layers = getattr(submodule, "layers", None)
        if not layers:
            return "unknown"
        first = layers[0]
        attn = None
        if isinstance(first, dict):
            attn = first.get("self_attn")
        elif hasattr(first, "get"):
            attn = first.get("self_attn")
        elif hasattr(first, "__contains__") and "self_attn" in first:
            attn = first["self_attn"]
        if attn is None:
            return "unknown"
        direct = getattr(attn, "position_mode", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(attn, "position_alphas", None)
        modes = getattr(attn, "POSITION_MODES", ())
        if isinstance(logits, torch.Tensor) and logits.numel() == len(modes) and len(modes) > 0:
            probs = torch.softmax(logits.detach(), dim=0)
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

    def _decoder_style_choice(model_obj) -> str:
        dec = getattr(model_obj, "forecast_decoder", None)
        if dec is None:
            return "not used"

        direct = getattr(dec, "decode_style", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct

        logits = getattr(dec, "decode_style_alphas", None)
        style_names = getattr(dec, "decode_style_names", ("autoregressive", "informer"))
        if isinstance(logits, torch.Tensor) and logits.numel() == len(style_names):
            probs = torch.softmax(logits.detach(), dim=0)
            top_idx = int(torch.argmax(probs).item())
            if 0 <= top_idx < len(style_names):
                return str(style_names[top_idx])

        resolver = getattr(dec, "resolve_decode_style", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    def _decoder_query_choice(model_obj) -> str:
        direct = getattr(model_obj, "decoder_query_mode", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(model_obj, "decoder_query_alphas", None)
        names = getattr(model_obj, "decoder_query_mode_names", ())
        if isinstance(logits, torch.Tensor) and logits.numel() == len(names) and len(names) > 0:
            probs = torch.softmax(logits.detach(), dim=0)
            return str(names[int(torch.argmax(probs).item())])
        resolver = getattr(model_obj, "resolve_decoder_query_mode", None)
        if callable(resolver):
            try:
                return str(resolver())
            except Exception:
                pass
        return "unknown"

    def _attention_choice(model_obj) -> str:
        return "not used"

    def _decoder_cross_attention_choice(model_obj) -> str:
        dec = getattr(model_obj, "forecast_decoder", None)
        if dec is None:
            return "not used"

        dec_choice = _enc_dec_choice(model_obj, "decoder").lower()
        internal = "not applicable"
        if "transformer" in dec_choice:
            submodule = getattr(dec, "rnn", None)
            if submodule is None:
                submodule = getattr(dec, "transformer", None)
            internal = "unknown"
            if submodule is not None:
                layers = getattr(submodule, "layers", None)
                if layers:
                    first_layer = layers[0]
                    cross_attn = None
                    if isinstance(first_layer, dict):
                        cross_attn = first_layer.get("cross_attn")
                    elif hasattr(first_layer, "get"):
                        cross_attn = first_layer.get("cross_attn")
                    elif hasattr(first_layer, "__contains__") and "cross_attn" in first_layer:
                        cross_attn = first_layer["cross_attn"]
                    if cross_attn is not None:
                        value = getattr(cross_attn, "attention_type", None)
                        if isinstance(value, str) and value:
                            if value != "auto":
                                internal = value
                            else:
                                logits = getattr(cross_attn, "attn_alphas", None)
                                modes = getattr(cross_attn, "MODES", ())
                                if (
                                    isinstance(logits, torch.Tensor)
                                    and logits.numel() == len(modes)
                                    and len(modes) > 0
                                ):
                                    probs = torch.softmax(logits.detach(), dim=0)
                                    top_idx = int(torch.argmax(probs).item())
                                    if 0 <= top_idx < len(modes):
                                        internal = str(modes[top_idx])

        return internal

    def _decoder_cross_position_choice(model_obj) -> str:
        dec = getattr(model_obj, "forecast_decoder", None)
        if dec is None:
            return "not used"
        submodule = getattr(dec, "rnn", None)
        if submodule is None:
            submodule = getattr(dec, "transformer", None)
        if submodule is None:
            return "unknown"
        layers = getattr(submodule, "layers", None)
        if not layers:
            return "unknown"
        first = layers[0]
        cross_attn = None
        if isinstance(first, dict):
            cross_attn = first.get("cross_attn")
        elif hasattr(first, "get"):
            cross_attn = first.get("cross_attn")
        elif hasattr(first, "__contains__") and "cross_attn" in first:
            cross_attn = first["cross_attn"]
        if cross_attn is None:
            return "unknown"
        direct = getattr(cross_attn, "position_mode", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(cross_attn, "position_alphas", None)
        modes = getattr(cross_attn, "POSITION_MODES", ())
        if isinstance(logits, torch.Tensor) and logits.numel() == len(modes) and len(modes) > 0:
            probs = torch.softmax(logits.detach(), dim=0)
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

    def _ffn_choice(model_obj, role: str) -> str:
        component = getattr(model_obj, f"forecast_{role}", None)
        if component is None:
            return "not used"
        submodule = getattr(component, "rnn", None)
        if submodule is None:
            submodule = getattr(component, "transformer", None)
        if submodule is None:
            return "unknown"
        layers = getattr(submodule, "layers", None)
        if not layers:
            return "unknown"
        first = layers[0]
        ffn = None
        if isinstance(first, dict):
            ffn = first.get("ffn")
        elif hasattr(first, "get"):
            ffn = first.get("ffn")
        elif hasattr(first, "__contains__") and "ffn" in first:
            ffn = first["ffn"]
        if ffn is None:
            return "unknown"
        direct = getattr(ffn, "ffn_mode", None)
        if isinstance(direct, str) and direct and direct != "auto":
            return direct
        logits = getattr(ffn, "ffn_alphas", None)
        modes = getattr(ffn, "MODE_NAMES", ())
        if isinstance(logits, torch.Tensor) and logits.numel() == len(modes) and len(modes) > 0:
            probs = torch.softmax(logits.detach(), dim=0)
            return str(modes[int(torch.argmax(probs).item())])
        return "unknown"

    def _transformer_summary(model_obj, sel_cfg: Dict[str, Any]) -> str:
        enc_choice = _enc_dec_choice(model_obj, "encoder").lower()
        dec_choice = _enc_dec_choice(model_obj, "decoder").lower()
        uses_transformer = ("transformer" in enc_choice) or ("transformer" in dec_choice)
        if not uses_transformer:
            return "not active"
        attn_type = _self_attention_choice(model_obj, "encoder")
        if attn_type in {"not used", "not applicable", "unknown"}:
            attn_type = _self_attention_choice(model_obj, "decoder")
        ffn_variant = _ffn_choice(model_obj, "encoder")
        if ffn_variant in {"not used", "unknown"}:
            ffn_variant = _ffn_choice(model_obj, "decoder")
        tokenizer_mode = _encoder_patch_choice(model_obj)
        return f"attn:{attn_type} ffn:{ffn_variant} enc_tok:{tokenizer_mode}"

    # Print selected architecture before entering final training.
    sel = best_candidate.get("candidate", {})
    arch_mode = sel.get("arch_mode", getattr(final_model, "arch_mode", "N/A"))
    p5_lines: List[str] = [
        "[P5] Selected architecture before final training:",
        (
            "[P5]   candidate_id="
            f"{sel.get('candidate_id', 'N/A')} | arch_mode={arch_mode} | "
            f"hidden_dim={sel.get('hidden_dim', getattr(final_model, 'hidden_dim', 'N/A'))} | "
            f"cells={sel.get('num_cells', getattr(final_model, 'num_cells', 'N/A'))} | "
            f"nodes={sel.get('num_nodes', getattr(final_model, 'num_nodes', 'N/A'))}"
        ),
        f"[P5]   selected_ops={sel.get('selected_ops', 'N/A')}",
        f"[P5]   selected_families={sel.get('selected_families', 'N/A')}",
        f"[P5]   transformer={_transformer_summary(final_model, sel)}",
        f"[P5]   normalization={_norm_choice(final_model)}",
        f"[P5]   encoder_decomposition={_decomp_choice(getattr(final_model, 'forecast_encoder', None))}",
        f"[P5]   decoder_decomposition={_decomp_choice(getattr(final_model, 'forecast_decoder', None))}",
        f"[P5]   encoder={_enc_dec_choice(final_model, 'encoder')}",
        f"[P5]   decoder={_enc_dec_choice(final_model, 'decoder')}",
        f"[P5]   attention={_attention_choice(final_model)}",
        f"[P5]   decoder_cross_attention={_decoder_cross_attention_choice(final_model)}",
        f"[P5]   decoder_cross_position={_decoder_cross_position_choice(final_model)}",
        f"[P5]   decoder_style={_decoder_style_choice(final_model)}",
        f"[P5]   decoder_query_generator={_decoder_query_choice(final_model)}",
    ]
    enc_sa = _self_attention_choice(final_model, "encoder")
    dec_sa = _self_attention_choice(final_model, "decoder")
    if enc_sa not in {"not used", "not applicable"}:
        p5_lines.append(f"[P5]   encoder_self_attention={enc_sa}")
        p5_lines.append(f"[P5]   encoder_attention_position={_self_attention_position_choice(final_model, 'encoder')}")
        p5_lines.append(f"[P5]   encoder_tokenizer={_encoder_patch_choice(final_model)}")
        p5_lines.append(f"[P5]   encoder_ffn={_ffn_choice(final_model, 'encoder')}")
    if dec_sa not in {"not used", "not applicable"}:
        p5_lines.append(f"[P5]   decoder_self_attention={dec_sa}")
        p5_lines.append(f"[P5]   decoder_attention_position={_self_attention_position_choice(final_model, 'decoder')}")
        p5_lines.append(f"[P5]   decoder_ffn={_ffn_choice(final_model, 'decoder')}")
    for cell_line in _cell_ops_summary(final_model):
        p5_lines.append(f"[P5]   {cell_line}")
    if final_discrete_arch:
        for k in sorted(final_discrete_arch.keys()):
            v = final_discrete_arch[k]
            if k in {"encoder", "decoder"} and isinstance(v, dict) and "type" in v:
                v_pretty = dict(v)
                if str(v_pretty.get("type", "")).startswith("op_"):
                    resolved = _enc_dec_choice(final_model, k)
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
            "phase1_rescore_mode": phase1_rescore_mode,
            "phase3_reduction_factor": int(phase3_reduction_factor),
            "phase3_min_epoch_budget": int(phase3_min_epoch_budget),
            "phase3_rung_epochs": [int(x) for x in rung_epochs],
            "phase3_verbose": bool(phase3_verbose),
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


def _resolve_phase3_rung_epochs(
    *,
    search_epochs: int,
    min_epoch_budget: int,
    reduction_factor: int,
    explicit,
) -> List[int]:
    """Build monotonically increasing ASHA rung budgets ending at search_epochs."""
    max_epochs = max(1, int(search_epochs))
    if explicit:
        rung_epochs = sorted(
            {max(1, min(max_epochs, int(v))) for v in explicit if int(v) > 0}
        )
        if not rung_epochs:
            return [max_epochs]
        if rung_epochs[-1] != max_epochs:
            rung_epochs.append(max_epochs)
        return rung_epochs

    budgets: List[int] = []
    cur = max(1, min(int(min_epoch_budget), max_epochs))
    while cur < max_epochs:
        budgets.append(int(cur))
        nxt = max(cur + 1, int(cur * reduction_factor))
        if nxt >= max_epochs:
            break
        cur = nxt
    budgets.append(max_epochs)
    return sorted(set(int(x) for x in budgets))


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
