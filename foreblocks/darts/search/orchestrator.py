import concurrent.futures
import random
import threading
import time
from typing import Any, Dict, List, Optional

from .candidate_scoring import candidate_diversity_bonus


def make_default_search_candidate_config(trainer, rng=None) -> Dict[str, Any]:
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
) -> Dict[str, Any]:
    t0 = time.perf_counter() if include_timing else None
    cfg = make_default_search_candidate_config(trainer, rng=rng)
    model = trainer._build_candidate_model(cfg)
    print(
        f"Evaluating candidate {candidate_id} with config: {cfg['selected_ops']}, hidden_dim={cfg['hidden_dim']}, num_cells={cfg['num_cells']}, num_nodes={cfg['num_nodes']}"
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


def select_top_candidates(candidates: List[Dict[str, Any]], top_k: int):
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[: min(int(top_k), len(candidates))]


def run_parallel_candidate_collection(
    *,
    num_candidates: int,
    candidate_fn,
    max_workers: Optional[int] = None,
    on_result=None,
    error_log_fn=None,
) -> List[Dict[str, Any]]:
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(candidate_fn, i) for i in range(num_candidates)]
        for f in futs:
            f.add_done_callback(_done_cb)
        for f in concurrent.futures.as_completed(futs):
            try:
                result = f.result()
                if result.get("success", False):
                    collected.append(result)
            except Exception as e:
                if error_log_fn is not None:
                    error_log_fn(e)
    return collected
