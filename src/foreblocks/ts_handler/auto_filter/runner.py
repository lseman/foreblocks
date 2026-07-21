"""foreblocks.ts_handler.auto_filter.runner.

Filter runner and auto-filter main entry point.

"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Literal

import pandas as pd

from foreblocks.ts_handler.auto_filter.metrics import (
    ScoringWeights,
    _candidate_band_penalties,
    _compute_scores,
    filter_metrics,
)
from foreblocks.ts_handler.auto_filter.registry import (
    _FILTER_REGISTRY,
    _SLOW_FILTERS,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None


def _resolve_n_jobs(n_jobs: int | None) -> int:
    if n_jobs is None or n_jobs == -1:
        return max(os.cpu_count() or 1, 1)
    return max(int(n_jobs), 1)


def _run_filter_candidate(
    name: str,
    fn: Any,
    ts: pd.Series,
) -> tuple[str, pd.Series]:
    return name, fn(ts)


def _metric_row(
    name: str,
    series: pd.Series,
    ts: pd.Series,
    filter_fn: Any | None,
    use_mc_gcv: bool,
) -> tuple[str, dict[str, float]]:
    return name, filter_metrics(
        series,
        ts,
        filter_fn=filter_fn,
        use_mc_gcv=use_mc_gcv,
    )


def _executor_cls(parallel_backend: str):
    if parallel_backend == "thread":
        return ThreadPoolExecutor
    if parallel_backend == "process":
        return ProcessPoolExecutor
    raise ValueError(
        "parallel_backend must be one of {'thread', 'process'}, "
        f"got {parallel_backend!r}."
    )


def _progress_iter(iterable, *, total: int, desc: str, enabled: bool):
    if not enabled or tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


def auto_filter(
    ts: pd.Series,
    fast: bool = False,
    weights: ScoringWeights | None = None,
    target_band: dict[str, float] | None = None,
    n_jobs: int | None = 1,
    parallel_backend: Literal["thread", "process"] = "thread",
    use_mc_gcv: bool = True,
    progress: bool = False,
) -> tuple[str, pd.Series, pd.DataFrame]:
    if weights is None:
        weights = ScoringWeights()
    if target_band is None:
        target_band = _DEFAULT_TARGET_BAND

    active = {
        name: fn
        for name, fn in _FILTER_REGISTRY.items()
        if not (fast and name in _SLOW_FILTERS)
    }

    resolved_n_jobs = _resolve_n_jobs(n_jobs)
    candidates_unordered: dict[str, pd.Series] = {}
    if resolved_n_jobs == 1 or len(active) <= 1:
        for name, fn in _progress_iter(
            active.items(),
            total=len(active),
            desc="auto_filter candidates",
            enabled=progress,
        ):
            try:
                candidates_unordered[name] = fn(ts)
            except Exception as exc:
                warnings.warn(
                    f"Filter '{name}' raised an exception and was skipped: {exc}",
                    stacklevel=2,
                )
    else:
        max_workers = min(resolved_n_jobs, len(active))
        executor_cls = _executor_cls(parallel_backend)
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_filter_candidate, name, fn, ts): name
                for name, fn in active.items()
            }
            for future in _progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="auto_filter candidates",
                enabled=progress,
            ):
                name = futures[future]
                try:
                    candidate_name, candidate = future.result()
                    candidates_unordered[candidate_name] = candidate
                except Exception as exc:
                    warnings.warn(
                        f"Filter '{name}' raised an exception and was skipped: {exc}",
                        stacklevel=2,
                    )

    candidates = {
        name: candidates_unordered[name]
        for name in active
        if name in candidates_unordered
    }

    if not candidates:
        raise RuntimeError("All filters failed. Cannot rank.")

    if resolved_n_jobs == 1 or len(candidates) <= 1:
        metrics_rows = {
            name: filter_metrics(
                series,
                ts,
                filter_fn=active.get(name),
                use_mc_gcv=use_mc_gcv,
            )
            for name, series in _progress_iter(
                candidates.items(),
                total=len(candidates),
                desc="auto_filter metrics",
                enabled=progress,
            )
        }
    else:
        metrics_unordered: dict[str, dict[str, float]] = {}
        max_workers = min(resolved_n_jobs, len(candidates))
        executor_cls = _executor_cls(parallel_backend)
        with executor_cls(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    _metric_row,
                    name,
                    series,
                    ts,
                    active.get(name),
                    use_mc_gcv,
                ): name
                for name, series in candidates.items()
            }
            for future in _progress_iter(
                as_completed(futures),
                total=len(futures),
                desc="auto_filter metrics",
                enabled=progress,
            ):
                name = futures[future]
                metrics_name, metrics = future.result()
                metrics_unordered[metrics_name] = metrics
        metrics_rows = {
            name: metrics_unordered[name]
            for name in candidates
            if name in metrics_unordered
        }
    mdf = pd.DataFrame(metrics_rows).T
    band_pen = (
        _candidate_band_penalties(ts, candidates, target_band) if target_band else None
    )
    mdf["score"] = _compute_scores(mdf, weights, band_penalty=band_pen)
    if band_pen is not None:
        mdf["band_penalty"] = band_pen.reindex(mdf.index).fillna(0.0)
    mdf = mdf.sort_values("score")

    best_name = mdf.index[0]
    return best_name, candidates[best_name], mdf


# ---------------------------------------------------------------------------
# Default target band
# ---------------------------------------------------------------------------

_DEFAULT_TARGET_BAND: dict[str, float] = {
    "rel_mae_min": 0.02,
    "rel_mae_max": 0.12,
    "roughness_ratio_min": 0.35,
    "roughness_ratio_max": 0.92,
    "derivative_corr_min": 0.90,
}
