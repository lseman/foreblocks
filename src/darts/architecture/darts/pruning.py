"""Operation pruning policies and bookkeeping for searchable cells."""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class OperationPruner:
    cells: nn.ModuleList
    pruning_history: list[dict[str, Any]]
    pruned_operations: set[str]
    operation_performance: dict[str, Any]

    def prune_weak_operations(
        self, threshold: float = 0.1, strategy: str = "probability"
    ) -> dict[str, Any]:
        """
        Prune weak operations based on their weights/importance

        Args:
            threshold: Minimum weight to keep an operation
            strategy: "probability" | "gradient" | "entropy" | "performance"

        Returns:
            Dict with pruning statistics
        """
        pruning_stats = {
            "operations_pruned": 0,
            "operations_kept": 0,
            "pruned_details": {},
            "threshold_used": threshold,
            "strategy": strategy,
        }

        if strategy == "probability":
            pruning_stats.update(self._prune_by_probability(threshold))
        elif strategy == "gradient":
            pruning_stats.update(self._prune_by_gradient_magnitude(threshold))
        elif strategy == "entropy":
            pruning_stats.update(self._prune_by_entropy(threshold))
        elif strategy == "performance":
            pruning_stats.update(self._prune_by_performance(threshold))
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

        # Store pruning history
        self.pruning_history.append(pruning_stats)
        print(
            f"Pruning completed with strategy '{strategy}' and threshold {threshold}."
        )
        return pruning_stats


    def _prune_by_probability(self, threshold: float) -> dict[str, Any]:
        """Prune operations based on their probability weights"""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        # Prune cell operations
        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if hasattr(edge, "get_alphas") and hasattr(edge, "available_ops"):
                        alphas = edge.get_alphas()
                        probs = F.softmax(alphas, dim=0)

                        # Find operations below threshold
                        weak_ops = []
                        for k, (op_name, prob) in enumerate(
                            zip(edge.available_ops, probs)
                        ):
                            if prob.item() < threshold and op_name != "Identity":
                                weak_ops.append((k, op_name, prob.item()))

                        # Prune weak operations by setting very low alpha values
                        if weak_ops:
                            for k, op_name, prob in weak_ops:
                                if self._set_edge_op_logit(edge, k, value=-10.0):
                                    # Track pruned operation
                                    op_id = f"cell_{i}_edge_{j}_{op_name}"
                                    self.pruned_operations.add(op_id)
                                    stats["pruned_details"][op_id] = prob
                                    stats["operations_pruned"] += 1

                        # Count kept operations
                        kept_ops = len(edge.available_ops) - len(weak_ops)
                        stats["operations_kept"] += kept_ops

        return stats


    def _set_edge_op_logit(self, edge: nn.Module, op_idx: int, value: float) -> bool:
        """Set a single operation logit for both flat and hierarchical MixedOp edges."""
        try:
            with torch.no_grad():
                if (
                    getattr(edge, "use_hierarchical", False)
                    and hasattr(edge, "op_to_group")
                    and hasattr(edge, "group_names")
                    and hasattr(edge, "op_alphas")
                    and op_idx in edge.op_to_group
                ):
                    group_idx, local_idx = edge.op_to_group[op_idx]
                    group_name = edge.group_names[group_idx]
                    edge.op_alphas[group_name][local_idx] = value
                    return True

                if hasattr(edge, "_alphas") and edge._alphas is not None:
                    edge._alphas[op_idx] = value
                    return True
        except Exception:
            return False

        return False


    def freeze_pruned_operations(
        self,
        pruning_stats: dict[str, Any] | None = None,
        logit_value: float = -20.0,
    ) -> int:
        """
        Hard-freeze previously pruned operations by forcing their logits to a low value.
        Returns number of logits successfully updated.
        """
        if pruning_stats and "pruned_details" in pruning_stats:
            pruned_ids = list(pruning_stats.get("pruned_details", {}).keys())
        else:
            pruned_ids = list(self.pruned_operations)

        frozen = 0
        for op_id in pruned_ids:
            match = re.match(r"^cell_(\d+)_edge_(\d+)_(.+)$", str(op_id))
            if match is None:
                continue

            cell_idx = int(match.group(1))
            edge_idx = int(match.group(2))
            op_name = match.group(3)

            if cell_idx < 0 or cell_idx >= len(self.cells):
                continue
            cell = self.cells[cell_idx]
            if (
                not hasattr(cell, "edges")
                or edge_idx < 0
                or edge_idx >= len(cell.edges)
            ):
                continue

            edge = cell.edges[edge_idx]
            if not hasattr(edge, "available_ops") or op_name not in edge.available_ops:
                continue

            op_idx = edge.available_ops.index(op_name)
            if self._set_edge_op_logit(edge, op_idx, value=float(logit_value)):
                frozen += 1

        return frozen


    def _prune_by_gradient_magnitude(self, threshold: float) -> dict[str, Any]:
        """Prune operations with consistently low architecture-gradient signal."""
        stats = {
            "operations_pruned": 0,
            "operations_kept": 0,
            "pruned_details": {},
            "gradient_fallback_edges": 0,
        }
        fallback_edges = []

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) == 0
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    grad_scores = torch.zeros_like(probs)

                    if getattr(edge, "use_hierarchical", False) and hasattr(
                        edge, "op_to_group"
                    ):
                        for op_idx in range(len(edge.available_ops)):
                            group_idx, local_idx = edge.op_to_group[op_idx]
                            group_name = edge.group_names[group_idx]
                            op_alpha = edge.op_alphas[group_name]
                            if op_alpha.grad is not None:
                                grad_scores[op_idx] = op_alpha.grad[local_idx].abs()
                    elif hasattr(edge, "_alphas") and edge._alphas.grad is not None:
                        grad_scores = edge._alphas.grad.detach().abs().float()

                    max_grad = (
                        grad_scores.max().item() if grad_scores.numel() > 0 else 0.0
                    )
                    if max_grad > 1e-12:
                        grad_scores = grad_scores / max_grad
                    else:
                        # Explicit fallback when gradients are unavailable.
                        grad_scores = probs.clone()
                        stats["gradient_fallback_edges"] += 1
                        fallback_edges.append(f"cell_{i}_edge_{j}")

                    weak_ops = []
                    top_idx = (
                        int(torch.argmax(probs).item()) if probs.numel() > 0 else -1
                    )
                    for k, op_name in enumerate(edge.available_ops):
                        score = float(grad_scores[k].item())
                        if score < threshold and op_name != "Identity" and k != top_idx:
                            weak_ops.append((k, op_name, score))

                    for k, op_name, score in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = score
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        if fallback_edges:
            preview = ", ".join(fallback_edges[:5])
            if len(fallback_edges) > 5:
                preview += ", ..."
            warnings.warn(
                "Gradient-based pruning fell back to probability scores for "
                f"{len(fallback_edges)} edges (no gradients available). "
                f"Examples: {preview}",
                RuntimeWarning,
                stacklevel=2,
            )

        return stats


    def _prune_by_entropy(self, threshold: float) -> dict[str, Any]:
        """Prune low-probability ops more aggressively when an edge distribution is confident."""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) <= 1
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    entropy = float((-(probs * torch.log(probs + 1e-8))).sum().item())
                    max_entropy = float(
                        torch.log(torch.tensor(float(len(probs)))).item()
                    )
                    confidence = 1.0 - (entropy / max(max_entropy, 1e-8))

                    # Increase pruning pressure only when the edge is already confident.
                    adaptive_threshold = threshold * (0.5 + confidence)
                    top_idx = int(torch.argmax(probs).item())

                    weak_ops = []
                    for k, (op_name, prob) in enumerate(zip(edge.available_ops, probs)):
                        p = float(prob.item())
                        if (
                            p < adaptive_threshold
                            and op_name != "Identity"
                            and k != top_idx
                        ):
                            weak_ops.append((k, op_name, p))

                    for k, op_name, p in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = p
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        return stats


    def _prune_by_performance(self, threshold: float) -> dict[str, Any]:
        """Prune operations using combined probability and tracked runtime performance."""
        stats = {"operations_pruned": 0, "operations_kept": 0, "pruned_details": {}}

        for i, cell in enumerate(self.cells):
            if hasattr(cell, "edges"):
                for j, edge in enumerate(cell.edges):
                    if (
                        not hasattr(edge, "available_ops")
                        or len(edge.available_ops) == 0
                    ):
                        continue

                    probs = edge.get_alphas().detach().float()
                    probs = probs / probs.sum().clamp_min(1e-8)

                    if hasattr(edge, "performance_tracker"):
                        perf = edge.performance_tracker.detach().float()
                        if perf.numel() == probs.numel():
                            p_min, p_max = perf.min(), perf.max()
                            if (p_max - p_min).abs().item() > 1e-8:
                                perf_norm = (perf - p_min) / (p_max - p_min)
                            else:
                                perf_norm = torch.sigmoid(perf)
                        else:
                            perf_norm = probs.clone()
                    else:
                        perf_norm = probs.clone()

                    usage = (
                        edge.usage_counter.detach().float()
                        if hasattr(edge, "usage_counter")
                        else torch.ones_like(probs)
                    )

                    combined = 0.7 * probs + 0.3 * perf_norm
                    top_idx = int(torch.argmax(combined).item())

                    weak_ops = []
                    for k, op_name in enumerate(edge.available_ops):
                        score = float(combined[k].item())
                        op_id = f"cell_{i}_edge_{j}_{op_name}"
                        self.operation_performance[op_id] = {
                            "probability": float(probs[k].item()),
                            "performance": float(perf_norm[k].item()),
                            "usage": float(usage[k].item()),
                            "combined": score,
                        }

                        # Avoid pruning very under-observed operations too early.
                        mature_enough = usage[k].item() >= 3
                        if (
                            score < threshold
                            and mature_enough
                            and op_name != "Identity"
                            and k != top_idx
                        ):
                            weak_ops.append((k, op_name, score))

                    for k, op_name, score in weak_ops:
                        if self._set_edge_op_logit(edge, k, value=-10.0):
                            op_id = f"cell_{i}_edge_{j}_{op_name}"
                            self.pruned_operations.add(op_id)
                            stats["pruned_details"][op_id] = score
                            stats["operations_pruned"] += 1

                    stats["operations_kept"] += len(edge.available_ops) - len(weak_ops)

        return stats

