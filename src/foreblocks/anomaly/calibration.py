"""foreblocks.anomaly.calibration.

Score calibration, confidence estimation, and ensemble combination for anomaly detectors.

Raw anomaly scores from different detectors often lack meaningful probability semantics.
This module provides temperature scaling, Platt scaling, and isotonic regression to
convert raw scores into calibrated probabilities. It also supports combining multiple
detector scores via learned weights and computing per-sample confidence scores.
Use when you need reliable thresholds, comparable scores across models, or ensemble fusion.

Core API:
- TemperatureScaler: learnable temperature for calibrating raw scores
- PlattScaler: logistic regression scaling with bias term
- isotonic_calibrate: non-parametric monotonic calibration
- EnsembleScoreCombiner: combine multi-detector scores with learned weights
- compute_confidence: per-sample confidence and uncertainty estimates
- fit_score_distribution: fit MAD/Gaussian/percentile baseline distributions

"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

# ── Temperature scaling ──


class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0, learnable: bool = True) -> None:
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(
                torch.tensor(init_temp, dtype=torch.float32)
            )
        else:
            self.register_buffer(
                "temperature", torch.tensor(init_temp, dtype=torch.float32)
            )

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        T = self.temperature.clamp_min(0.01)
        return scores / T

    def fit(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        *,
        lr: float = 0.01,
        n_steps: int = 200,
    ) -> float:
        s = torch.tensor(raw_scores, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=n_steps)

        def closure() -> float:
            optimizer.zero_grad()
            scaled = self(s)
            probs = torch.sigmoid(scaled)
            nll = -(
                y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)
            ).mean()
            nll.backward()
            return nll.item()

        optimizer.step(closure)
        return self.temperature.item()


# ── Platt scaling (logistic regression on scores) ──


class PlattScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.a * scores + self.b)

    def fit(
        self,
        raw_scores: np.ndarray,
        labels: np.ndarray,
        *,
        lr: float = 0.01,
        n_steps: int = 500,
    ) -> tuple[float, float]:
        s = torch.tensor(raw_scores, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32)

        optimizer = torch.optim.Adam([self.a, self.b], lr=lr)

        for _ in range(n_steps):
            optimizer.zero_grad()
            probs = self(s)
            nll = -(
                y * torch.log(probs + 1e-8) + (1 - y) * torch.log(1 - probs + 1e-8)
            ).mean()
            nll.backward()
            optimizer.step()

        return self.a.item(), self.b.item()


# ── Isotonic regression ──


def isotonic_calibrate(
    raw_scores: np.ndarray,
    labels: np.ndarray,
    *,
    increasing: bool = True,
) -> np.ndarray:
    scores = np.asarray(raw_scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    n = len(scores)
    if n == 0:
        return np.empty(0)

    idx = np.argsort(scores)
    sorted_labels = labels[idx]

    values = sorted_labels.copy()
    weights = np.ones(n)

    i = 1
    while i < n:
        violation = (
            values[i] < values[i - 1] if increasing else values[i] > values[i - 1]
        )
        if violation:
            w_sum = weights[i - 1] + weights[i]
            v_avg = (weights[i - 1] * values[i - 1] + weights[i] * values[i]) / w_sum
            values[i - 1] = v_avg
            values[i] = v_avg
            weights[i - 1] = w_sum
            weights[i] = 0
            i = max(1, i - 1)
        else:
            i += 1

    calibrated = np.empty(n)
    calibrated[idx] = values
    return np.clip(calibrated, 0.0, 1.0)


# ── Confidence scoring ──


@dataclass
class ConfidenceResult:
    probabilities: np.ndarray  # P(anomaly) after calibration
    confidence: np.ndarray  # 1 - P(anomaly) for normal, P(anomaly) for anomaly
    uncertainty: np.ndarray  # 1 - |P(anomaly) - 0.5| * 2 (0=uncertain, 1=confident)
    labels: np.ndarray  # binary decision


def compute_confidence(
    raw_scores: np.ndarray,
    threshold: float,
    *,
    calibrated_scores: np.ndarray | None = None,
) -> ConfidenceResult:
    scores = np.asarray(raw_scores, dtype=np.float64)

    if calibrated_scores is not None:
        probs = np.asarray(calibrated_scores, dtype=np.float64)
    else:
        probs = 1.0 / (1.0 + np.exp(-0.5 * (scores - threshold)))

    probs = np.clip(probs, 0.0, 1.0)
    labels = (scores > threshold).astype(np.int8)

    confidence = 2.0 * np.abs(probs - 0.5)
    uncertainty = 1.0 - confidence

    return ConfidenceResult(
        probabilities=probs,
        confidence=confidence,
        uncertainty=uncertainty,
        labels=labels,
    )


# ── Ensemble score combiner ──


class EnsembleScoreCombiner:
    def __init__(self, n_detectors: int, strategy: str = "equal") -> None:
        self.n_detectors = n_detectors
        self.strategy = strategy
        if strategy == "equal":
            self.weights = np.ones(n_detectors) / n_detectors
        elif strategy == "learned":
            self.weights = np.ones(n_detectors) / n_detectors
            self.learned = False
        elif strategy == "adaptive":
            self.weights = np.ones(n_detectors) / n_detectors
            self.per_detector_scores: list[list[float]] = [
                [] for _ in range(n_detectors)
            ]
        else:
            raise ValueError(f"unknown strategy: {strategy}")

    def normalize_scores(self, all_scores: np.ndarray) -> np.ndarray:
        normalized = np.zeros_like(all_scores)
        for i in range(all_scores.shape[1]):
            col = all_scores[:, i]
            finite = np.isfinite(col)
            mean = np.nanmean(col[finite]) if finite.any() else 0.0
            std = np.nanstd(col[finite]) if finite.any() else 1.0
            std = max(std, 1e-8)
            normalized[:, i] = (col - mean) / std
        return normalized

    def combine(self, all_scores: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        scores = self.normalize_scores(all_scores) if normalize else all_scores
        return scores @ self.weights

    def fit_weights(
        self,
        all_scores: np.ndarray,
        labels: np.ndarray,
        *,
        method: str = "roc",
    ) -> np.ndarray:
        scores_arr = all_scores.copy()
        best_weight = np.ones(self.n_detectors) / self.n_detectors
        best_metric = -1.0

        metric_funcs: list[tuple[str, Callable[[np.ndarray, np.ndarray], float]]] = [
            ("roc_j", self._roc_j),
            ("accuracy", self._classification_accuracy),
            ("logloss", self._neg_logloss),
        ]
        for name, metric_fn in metric_funcs:
            w = self._optimize_weights(scores_arr, labels, metric_fn)
            combined = scores_arr @ w
            m = metric_fn(combined, labels)
            is_logloss = name == "logloss"
            if (is_logloss and m < best_metric) or (not is_logloss and m > best_metric):
                best_metric = m
                best_weight = w.copy()

        self.weights = best_weight
        self.strategy = "learned"
        return self.weights

    def _optimize_weights(
        self,
        all_scores: np.ndarray,
        labels: np.ndarray,
        metric_fn: Callable[[np.ndarray, np.ndarray], float],
    ) -> np.ndarray:
        w = np.ones(self.n_detectors) / self.n_detectors
        step = 0.05

        for _ in range(50):
            improved = False
            for i in range(self.n_detectors):
                for delta in [-step, step]:
                    w_test = w.copy()
                    w_test[i] += delta
                    w_test = np.clip(w_test, 0, 1)
                    w_test /= w_test.sum()
                    combined = all_scores @ w_test
                    current = metric_fn(all_scores @ w, labels)
                    candidate = metric_fn(combined, labels)
                    if candidate > current:
                        w = w_test
                        improved = True
            if not improved:
                break
        return w

    def _roc_j(self, scores: np.ndarray, labels: np.ndarray) -> float:
        best_j = -1.0
        mask_pos = labels == 1
        mask_neg = labels == 0
        for thresh in np.percentile(scores, np.arange(5, 96, 5)):
            tp = int(((scores > thresh) & mask_pos).sum())
            fp = int(((scores > thresh) & mask_neg).sum())
            fn = int(((scores <= thresh) & mask_pos).sum())
            tn = int(((scores <= thresh) & mask_neg).sum())
            fpr = fp / (fp + tn + 1e-8)
            tpr = tp / (tp + fn + 1e-8)
            j = tpr - fpr
            if j > best_j:
                best_j = j
        return best_j

    def _classification_accuracy(self, scores: np.ndarray, labels: np.ndarray) -> float:
        thresh = np.percentile(scores, 90)
        preds = (scores > thresh).astype(int)
        return float((preds == labels).mean())

    def _neg_logloss(self, scores: np.ndarray, labels: np.ndarray) -> float:
        probs = 1.0 / (1.0 + np.exp(-scores))
        return float(
            -(
                labels * np.log(probs + 1e-8) + (1 - labels) * np.log(1 - probs + 1e-8)
            ).mean()
        )

    def update_adaptive(
        self, all_scores: np.ndarray, labels: np.ndarray, window_size: int = 100
    ) -> None:
        n = all_scores.shape[0]
        start = 0 if n < window_size else n - window_size

        recent_scores = all_scores[start:]
        recent_labels = labels[start:]

        errors = np.zeros(self.n_detectors)
        for i in range(self.n_detectors):
            thresh = np.percentile(recent_scores[:, i], 90)
            preds = (recent_scores[:, i] > thresh).astype(int)
            errors[i] = 1.0 - float((preds == recent_labels).mean())

        errors = np.clip(errors, 0.01, None)
        self.weights = 1.0 / errors
        self.weights /= self.weights.sum()


# ── Score distribution fitting ──


def fit_score_distribution(
    scores: np.ndarray, *, method: str = "mad"
) -> dict[str, Any]:
    finite = np.asarray(scores, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {"threshold": float("inf")}

    if method == "mad":
        median = float(np.median(finite))
        mad = float(np.median(np.abs(finite - median)))
        if mad < 1e-8:
            mad = float(np.std(finite)) + 1e-8
        return {
            "method": method,
            "median": median,
            "mad": mad,
            "threshold": median + 3.5 * 1.4826 * mad,
        }

    if method == "gaussian":
        mean = float(np.mean(finite))
        std = float(np.std(finite)) + 1e-8
        return {
            "method": method,
            "mean": mean,
            "std": std,
            "threshold": mean + 3.0 * std,
        }

    if method == "percentile":
        p99 = float(np.percentile(finite, 99))
        p95 = float(np.percentile(finite, 95))
        return {"method": method, "p99": p99, "p95": p95, "threshold": (p99 + p95) / 2}

    raise ValueError(f"unknown method: {method}")
