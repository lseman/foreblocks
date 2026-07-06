import concurrent.futures
import random
import threading
import time
from typing import Any

from ..utils.training import create_progress_bar
from .candidate_scoring import candidate_diversity_bonus
from .metrics import _ZC_GPU_LOCK


def make_default_search_candidate_config(trainer, rng=None) -> dict[str, Any]:
    rng = rng or random
    return trainer._make_candidate_config(
        rng,
        trainer.all_ops,
        trainer.hidden_dims,
        (1, 2),
        (3, 4),
        min_ops=2,
        max_ops=len(trainer.all_ops),
        require_identity=True,
    )


def evaluate_search_candidate(
    trainer,
    *,
    candidate_id: int,
    val_loader,
    max_samples: int,
    num_batches: int = 1,
    include_timing: bool = False,
    rng=None,
) -> dict[str, Any]:
    t0 = time.perf_counter() if include_timing else None
    cfg = make_default_search_candidate_config(trainer, rng=rng)
    # Build under the shared GPU lock: candidate construction runs a FLOPs
    # profiling forward on-device, so doing it concurrently across worker
    # threads oversubscribes the GPU just like the metric eval does. Holding
    # the same lock keeps the whole candidate's GPU work serialized.
    with _ZC_GPU_LOCK:
        model = trainer._build_candidate_model(cfg)
    print(
        f"Evaluating candidate {candidate_id} with config: {cfg['selected_ops']}, "
        f"hidden_dim={cfg['hidden_dim']}, num_cells={cfg['num_cells']}, "
        f"num_nodes={cfg['num_nodes']}, arch_mode={cfg.get('arch_mode', 'encoder_decoder')}, "
        f"families={cfg.get('selected_families', [])}, "
        f"attn={cfg.get('transformer_self_attention_type', 'auto')}, "
        f"ffn={cfg.get('transformer_ffn_variant', 'swiglu')}"
    )
    metrics = trainer.evaluate_zero_cost_metrics(
        model, val_loader, max_samples=max_samples, num_batches=num_batches
    )
    print(f"   Metrics: {metrics}")
    base_score = float(metrics["aggregate_score"])
    diversity_bonus = candidate_diversity_bonus(cfg["selected_ops"], trainer.all_ops)
    adjusted_score = base_score + diversity_bonus
    out = {
        "candidate_id": candidate_id,
        "model": model,
        "metrics": metrics,
        "base_score": base_score,
        "diversity_bonus": diversity_bonus,
        "score": adjusted_score,
        "success": True,
        **cfg,
    }
    if include_timing and t0 is not None:
        out["phase1_dt"] = float(time.perf_counter() - t0)
    return out


def select_top_candidates(candidates: list[dict[str, Any]], top_k: int):
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[: min(int(top_k), len(candidates))]


def run_parallel_candidate_collection(
    *,
    num_candidates: int,
    candidate_fn,
    max_workers: int | None = None,
    on_result=None,
    error_log_fn=None,
    progress: bool = False,
    progress_desc: str = "Phase 1 candidates",
    candidate_timeout: float = 120.0,
) -> list[dict[str, Any]]:
    if max_workers is None:
        device = str(getattr(getattr(candidate_fn, "__self__", None), "device", ""))
        # ``candidate_fn`` is normally a local closure, so fall back to the
        # conservative CUDA default used by the multi-fidelity caller below.
        max_workers = 1 if device.startswith("cuda") else None
    collected = []
    lock = threading.Lock()
    completed = 0

    def _done_cb(fut):
        nonlocal completed
        try:
            result = fut.result()
        except Exception:
            return
        with lock:
            completed += 1
            if on_result is not None:
                on_result(result, completed)

    # Total wall-clock budget: generous per-candidate allowance × candidate count.
    # Per-metric timeouts inside _compute_safely handle the common hang cases; this
    # outer budget is a belt-and-suspenders guard against hangs outside _compute_safely
    # (e.g. the shared forward pass in compute_all).
    total_timeout: float | None = (
        candidate_timeout * num_candidates if candidate_timeout > 0 else None
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(candidate_fn, i) for i in range(num_candidates)]
        for f in futs:
            f.add_done_callback(_done_cb)
        try:
            completed_iter = concurrent.futures.as_completed(
                futs, timeout=total_timeout
            )
            if progress:
                completed_iter = create_progress_bar(
                    completed_iter,
                    progress_desc,
                    leave=False,
                    total=len(futs),
                    unit="candidate",
                )
            for f in completed_iter:
                try:
                    result = f.result()
                    if result.get("success", False):
                        collected.append(result)
                except Exception as e:
                    if error_log_fn is not None:
                        error_log_fn(e)
        except concurrent.futures.TimeoutError:
            timed_out = sum(1 for f in futs if not f.done())
            if error_log_fn is not None:
                error_log_fn(
                    TimeoutError(
                        f"{timed_out} candidate(s) did not complete within "
                        f"{total_timeout:.0f}s total budget; collected "
                        f"{len(collected)}/{num_candidates}."
                    )
                )
            # Collect any futures that finished before the timeout.
            for f in futs:
                if f.done() and not f.cancelled():
                    try:
                        result = f.result(timeout=1)
                        if result.get("success", False) and result not in collected:
                            collected.append(result)
                    except Exception:
                        pass
    return collected
