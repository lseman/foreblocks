import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..candidates.scoring import (
    normalize_metric_value as _normalize_metric_value_shared,
    score_from_metrics as _score_from_metrics_shared,
)
from .computer import _ZC_GPU_LOCK, MetricsComputer
from .config import Config, Result


class ZeroCostNAS:
    """Main zero-cost NAS evaluation class"""

    def __init__(self, config: Config | None = None):
        self.config = config or Config()
        self.computer = MetricsComputer(self.config)

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 3,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a single model"""
        # Serialise GPU access so concurrent candidate threads don't
        # oversubscribe the device (see _ZC_GPU_LOCK).
        with _ZC_GPU_LOCK:
            model = model.to(device)
            model.eval()

            batches = []
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                inputs, targets = self._extract_inputs_targets(batch, model, device)
                inputs = inputs[: self.config.max_samples]
                targets = targets[: self.config.max_samples]
                batches.append((inputs, targets))

            if not batches:
                return {
                    "metrics": {},
                    "success_rates": {},
                    "error_messages": {
                        "_global": "No valid batches after extraction/slicing."
                    },
                    "aggregate_score": float("-inf"),
                    "config": self.config,
                }

            all_results = []
            model_only_results = self.computer.compute_model_only_metrics(model)
            heavy_batches = max(
                1, int(getattr(self.config, "heavy_metrics_batches", 1))
            )
            for i, (inputs, targets) in enumerate(batches):
                batch_results = self.computer.compute_all(
                    model,
                    inputs,
                    targets,
                    include_heavy_metrics=(i < heavy_batches),
                    model_only_results=model_only_results,
                )
                all_results.append(batch_results)
            final_results: dict[str, Result] = self._aggregate_results(all_results)
            score = self._compute_score(final_results)

        return {
            "metrics": {k: r.value for k, r in final_results.items()},
            "success_rates": {k: r.success for k, r in final_results.items()},
            "error_messages": {
                k: r.error for k, r in final_results.items() if not r.success
            },
            "aggregate_score": score,
            "config": self.config,
        }

    def _extract_inputs_targets(self, batch, model, device):
        """Handles various batch formats and generates dummy targets if needed"""
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            inputs, targets = batch[0].to(device), batch[1].to(device)
        else:
            inputs = (
                batch[0].to(device)
                if isinstance(batch, (list, tuple))
                else batch.to(device)
            )
            with torch.no_grad():
                output = model(inputs[:1])
                if isinstance(output, tuple):
                    output = output[0]

                if output.dim() > 1 and output.size(1) > 1:
                    targets = torch.randint(
                        0, output.size(1), (inputs.size(0),), device=device
                    )
                else:
                    targets = (
                        torch.randint(0, 2, (inputs.size(0),), device=device)
                        if output.dim() > 1
                        else torch.randn(inputs.size(0), device=device)
                    )

        return inputs, targets

    def _aggregate_results(
        self, all_results: list[dict[str, Result]]
    ) -> dict[str, Result]:
        """Aggregate metric results across batches using sigma-clipped mean."""
        metrics = all_results[0].keys()
        aggregated = {}

        for metric in metrics:
            # print(f"Aggregating results for metric: {metric}")
            vals = [r[metric].value for r in all_results if r[metric].success]
            # print(f"Values for {metric}: {vals}")
            times = [r[metric].time for r in all_results]
            success = any(r[metric].success for r in all_results)
            avg_time = sum(times) / len(times) if times else 0.0

            agg_value = float("nan")
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                arr_kept = arr
                # Iterative sigma-clipping is more stable for tiny batch counts.
                for _ in range(5):
                    if arr_kept.size < 3:
                        break
                    mu = float(np.mean(arr_kept))
                    sd = float(np.std(arr_kept))
                    if not np.isfinite(sd) or sd <= 0:
                        break
                    keep = np.abs(arr_kept - mu) <= (2.5 * sd)
                    if np.all(keep):
                        break
                    new_arr = arr_kept[keep]
                    if new_arr.size == 0 or new_arr.size == arr_kept.size:
                        break
                    arr_kept = new_arr
                agg_value = float(np.mean(arr_kept)) if arr_kept.size else float("nan")

            is_nan = math.isnan(agg_value)

            aggregated[metric] = Result(
                value=0.0 if is_nan else agg_value,
                success=success and not is_nan,
                error=(
                    ""
                    if success and not is_nan
                    else f"{metric} resulted in NaN"
                    if is_nan
                    else "All batches failed"
                ),
                time=avg_time,
            )
        # print(f"Aggregated results: {aggregated}")

        return aggregated

    def _compute_score(self, results: dict[str, Result]) -> float:
        """Compute weighted aggregate score"""
        total_score = 0.0
        total_weight = 0.0

        def _weight_for_metric(metric: str) -> float | None:
            if metric in self.config.weights:
                return float(self.config.weights[metric])
            if metric == "activation_diversity" and "zennas" in self.config.weights:
                return float(self.config.weights["zennas"])
            if metric == "zennas" and "activation_diversity" in self.config.weights:
                return float(self.config.weights["activation_diversity"])
            return None

        for metric, result in results.items():
            if not result.success:
                continue

            weight = _weight_for_metric(metric)
            if weight is None:
                continue
            normalized = _normalize_metric_value_shared(metric, result.value)

            total_score += normalized * weight
            total_weight += abs(weight)

        return total_score / max(total_weight, 1.0)

    def evaluate_model_raw_metrics(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        device: torch.device,
        num_batches: int = 3,
    ) -> dict[str, Any]:
        """
        Compute raw metric values only (no weighting).
        Robust to individual metric failures, returning:
        - raw_metrics: aggregated raw values for metrics that succeeded at least once
        - success_rates: fraction of batches where each metric succeeded
        - errors: last error string seen for each metric (if any)
        """
        print("Evaluating raw metrics with robust error handling...")
        if isinstance(device, str):
            device = torch.device(device)

        per_metric_values: dict[str, list[float]] = {}
        per_metric_success: dict[str, int] = {}
        per_metric_total: dict[str, int] = {}
        per_metric_errors: dict[str, str] = {}

        # Serialise GPU access across concurrent candidate threads (see
        # _ZC_GPU_LOCK). Held across batch extraction + metric computation; the
        # pure-CPU aggregation below runs outside the lock.
        with _ZC_GPU_LOCK:
            model = model.to(device)
            model.eval()

            # ---- collect (inputs, targets) batches (same as evaluate_model)
            batches: list[tuple[torch.Tensor, torch.Tensor]] = []
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break

                inputs, targets = self._extract_inputs_targets(batch, model, device)
                inputs = inputs[: self.config.max_samples]
                targets = targets[: self.config.max_samples]

                if inputs is None or targets is None:
                    continue
                if inputs.numel() == 0 or targets.numel() == 0:
                    continue

                batches.append((inputs, targets))

            if not batches:
                return {
                    "raw_metrics": {},
                    "success_rates": {},
                    "errors": {"_global": "No valid batches after extraction/slicing."},
                }

            model_only_results = self.computer.compute_model_only_metrics(model)
            heavy_batches = max(
                1, int(getattr(self.config, "heavy_metrics_batches", 1))
            )

            # IMPORTANT: DO NOT use torch.no_grad() here (grad-based metrics
            # need autograd)
            for batch_idx, (inputs, targets) in enumerate(batches):
                try:
                    results = self.computer.compute_all(
                        model,
                        inputs,
                        targets,
                        include_heavy_metrics=(batch_idx < heavy_batches),
                        model_only_results=model_only_results,
                    )
                except Exception as e:
                    per_metric_errors["_batch_compute_all"] = str(e)
                    continue

                self._accumulate_raw_batch(
                    results,
                    per_metric_values,
                    per_metric_success,
                    per_metric_total,
                    per_metric_errors,
                )

        return self._finalize_raw_metrics(
            per_metric_values, per_metric_success, per_metric_total, per_metric_errors
        )

    @staticmethod
    def _accumulate_raw_batch(
        results,
        per_metric_values,
        per_metric_success,
        per_metric_total,
        per_metric_errors,
    ) -> None:
        for name, res in results.items():
            per_metric_total[name] = per_metric_total.get(name, 0) + 1

            try:
                if isinstance(res, Result) and not res.success:
                    per_metric_errors[name] = res.error or "Metric failed"
                    continue
                if isinstance(res, Result):
                    val = float(res.value)
                else:
                    val = float(res)

                # Filter NaN/inf
                if not torch.isfinite(torch.tensor(val)):
                    raise ValueError(f"Non-finite value: {val}")

                per_metric_values.setdefault(name, []).append(val)
                per_metric_success[name] = per_metric_success.get(name, 0) + 1

            except Exception as e:
                per_metric_errors[name] = str(e)

    @staticmethod
    def _finalize_raw_metrics(
        per_metric_values, per_metric_success, per_metric_total, per_metric_errors
    ) -> dict[str, Any]:
        raw_metrics: dict[str, float] = {
            name: float(sum(vals) / len(vals))
            for name, vals in per_metric_values.items()
            if len(vals) > 0
        }

        success_rates: dict[str, float] = {
            name: float(per_metric_success.get(name, 0) / max(tot, 1))
            for name, tot in per_metric_total.items()
        }

        # Debug helper: if empty, print why (optional)
        if not raw_metrics:
            print("❌ evaluate_model_raw_metrics: all metrics failed.")
            for k, v in per_metric_errors.items():
                print(f"  {k}: {v}")

        return {
            "raw_metrics": raw_metrics,
            "success_rates": success_rates,
            "errors": per_metric_errors,
        }

    def score_from_metrics(
        self, metrics: dict[str, float], weights: dict[str, float]
    ) -> float:
        """Compute weighted score from precomputed raw metrics."""
        return _score_from_metrics_shared(metrics, weights)

    @staticmethod
    def _normalize_metric_value(metric: str, value: float) -> float:
        """Normalize metric values on comparable scales while preserving signed signals."""
        return _normalize_metric_value_shared(metric, value)
