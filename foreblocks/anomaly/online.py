"""foreblocks.anomaly.online.

Test-time adaptation and streaming anomaly detection for deployed models.

Provides online adaptation techniques (TENT-style entropy minimization, batch norm
adaptation) to adjust pre-trained detectors to distribution shifts at test time.
Includes a streaming detector with adaptive thresholds via EMA statistics, and a BN-
adaptive wrapper for existing Foreblocks detectors. Use when your deployment
environment experiences concept drift or when you need real-time anomaly scoring
with self-adapting thresholds.

References:
- ITAD: Iterative Test-time Adaptation for Anomaly Detection (2024)
- TENT: Fully Test-time Adaptation by Entropy (Wang et al., 2021)

Core API:
- EMAStatistics: exponential moving average for online threshold/score tracking
- BatchNormAdapter: adapts BN layers at test time
- TENTAdapter: entropy-minimization test-time adaptation
- StreamingAnomalyDetector: online scoring with adaptive thresholds
- BNAdaptiveWrapper: wraps detectors with BN adaptation

"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch
import torch.nn as nn

# ── Streaming EMA statistics ──


class EMAStatistics:
    """Exponential moving average for online threshold/score adaptation.

    Maintains running mean and std of scores with configurable decay.
    Supports percentile computation for adaptive thresholding.
    """

    def __init__(self, decay: float = 0.99, max_len: int = 10000) -> None:
        self.decay = decay
        self.max_len = max_len
        self._values: deque[float] = deque(maxlen=max_len)
        self._mean = 0.0
        self._std = 1.0
        self._n = 0

    def update(self, score: float) -> None:
        """Update statistics with new score."""
        if not np.isfinite(score):
            return
        self._values.append(float(score))
        self._n += 1
        if self._n == 1:
            self._mean = float(score)
            self._std = 1.0
        else:
            self._mean = self.decay * self._mean + (1 - self.decay) * float(score)
            diff = float(score) - self._mean
            self._std = self.decay * self._std + (1 - self.decay) * diff * diff
            self._std = np.sqrt(self._std) + 1e-8

    @property
    def mean(self) -> float:
        return self._mean

    @property
    def std(self) -> float:
        return self._std

    def adaptive_threshold(self, z: float = 3.0) -> float:
        """Adaptive threshold based on EMA stats."""
        return self._mean + z * self._std

    def percentile(self, p: float) -> float:
        """Approximate percentile from stored values."""
        if len(self._values) < 10:
            return float("inf")  # type: ignore[return-value]
        return float(np.percentile(list(self._values), p))


# ── Batch norm adaptation ──


class BatchNormAdapter:
    """Adapts batch normalization layers at test time.

    Updates running mean/var of BN layers with test batch statistics.
    Supports partial adaptation (mix with pre-trained stats).
    """

    def __init__(self, model: nn.Module, adapt_rate: float = 0.01) -> None:
        self.model = model
        self.adapt_rate = adapt_rate
        self._bn_modules = [
            m
            for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))
        ]

    def adapt(self, batch: torch.Tensor) -> None:
        """Update BN statistics with current batch."""
        rate = self.adapt_rate
        omr = 1.0 - rate
        for bn in self._bn_modules:
            if bn.running_mean is None or bn.running_var is None:
                continue
            if batch.dim() == 3:
                batch_mean = batch.mean(dim=(0, 1))
                batch_var = batch.var(dim=(0, 1), unbiased=False)
            else:
                batch_mean = batch.mean(dim=0)
                batch_var = batch.var(dim=0, unbiased=False)
            bn.running_mean = omr * bn.running_mean + rate * batch_mean.detach()
            bn.running_var = omr * bn.running_var + rate * batch_var.detach()


# ── Entropy-based test-time adaptation ──


class TENTAdapter:
    """Test-time adaptation via entropy minimization.

    Fine-tunes the last layer (or all layers) of a pre-trained model
    by minimizing prediction entropy on test batches.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        *,
        freeze_until: str | None = None,
        n_steps: int = 1,
    ) -> None:
        self.model = model
        self.n_steps = n_steps

        # Find parameters to update (last layer by default)
        if freeze_until:
            all_params = list(self.model.named_parameters())
            cutoff = (
                next(
                    (
                        i
                        for i, (name, _) in enumerate(all_params)
                        if name == freeze_until
                    ),
                    len(all_params),
                )
                + 1
            )
            self._update_params = [p for _, p in all_params[cutoff:]]
        else:
            # Last layer only
            last_layer_params = []
            for name, param in model.named_parameters():
                if name.endswith("weight") or name.endswith("bias"):
                    last_layer_params.append((name, param))
            self._update_params = [p for _, p in last_layer_params]

        self.optimizer = torch.optim.Adam(self._update_params, lr=lr)

    def adapt(self, batch: torch.Tensor) -> float:
        """Run one adaptation step. Returns entropy."""
        self.model.train()
        total_entropy = 0.0

        for _ in range(self.n_steps):
            self.optimizer.zero_grad()
            output = self.model(batch)

            # Compute entropy
            if isinstance(output, tuple):
                # Assume (reconstruction, ...) — use reconstruction error as proxy
                recon, *_ = output
                # Convert to probability-like scores
                errors = recon.mean(dim=(1, 2))
                probs = torch.softmax(errors / 0.1, dim=-1)
            else:
                # Assume 2D output [B, features] or [B, 1]
                errors = output.mean(dim=-1) if output.dim() > 1 else output
                probs = torch.softmax(errors / 0.1, dim=-1)

            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            entropy.backward()
            self.optimizer.step()
            total_entropy += entropy.item()

        return total_entropy / max(self.n_steps, 1)


# ── Anomaly score streaming adapter ──


@dataclass
class StreamingResult:
    """Result from streaming anomaly detection."""

    score: float
    is_anomaly: bool
    threshold: float
    confidence: float  # 0-1, how certain about the decision
    ema_score: float
    ema_threshold: float


class StreamingAnomalyDetector:
    """Online anomaly detector with adaptive threshold and EMA scoring.

    Scores individual samples (or mini-batches) and adapts the threshold
    based on recent score distribution. Designed for real-time monitoring.
    """

    def __init__(
        self,
        detector: Any,  # ForeblocksAnomalyDetector or similar
        initial_threshold: float | None = None,
        ema_decay: float = 0.99,
        z_threshold: float = 3.0,
        adaptation_window: int = 1000,
        min_samples_before_adapt: int = 100,
    ) -> None:
        self.detector = detector
        self.ema = EMAStatistics(decay=ema_decay)
        self.z_threshold = z_threshold
        self.adaptation_window = adaptation_window
        self.min_samples_before_adapt = min_samples_before_adapt
        self._sample_count = 0
        self._batch_buffer: deque[np.ndarray] = deque(maxlen=512)
        self._baseline_threshold = initial_threshold

    def score(self, sample: np.ndarray) -> StreamingResult:
        """Score a single sample (or batch [T, D]) and adapt threshold."""
        self._sample_count += 1

        # Get score from underlying detector
        if self.detector is not None:
            try:
                if hasattr(self.detector, "decision_scores"):
                    windowed = self.detector._windows_from_series(
                        sample, fit_scaler=False
                    )
                    raw_scores = self.detector.score_windows(windowed)
                    # Take last score as current sample score
                    current_score = (
                        float(np.nanmax(raw_scores[-1])) if raw_scores.size > 0 else 0.0
                    )
                else:
                    current_score = self._default_score(sample)
            except Exception:
                current_score = self._default_score(sample)
        else:
            current_score = self._default_score(sample)

        if not np.isfinite(current_score):
            current_score = 0.0

        # Update EMA
        self.ema.update(current_score)

        # Compute adaptive threshold
        if self._sample_count < self.min_samples_before_adapt:
            threshold = (
                self._baseline_threshold
                if self._baseline_threshold is not None
                else self.ema.adaptive_threshold(self.z_threshold)
            )
        else:
            threshold = self.ema.adaptive_threshold(self.z_threshold)

        # Update baseline threshold if we have enough samples
        if (
            self._sample_count > self.adaptation_window
            and self._baseline_threshold is not None
        ):
            recent_scores = list(self.ema._values)[-self.adaptation_window :]
            if recent_scores:
                p99 = np.percentile(recent_scores, 99)
                p1 = np.percentile(recent_scores, 1)
                range_ = p99 - p1
                if range_ > 1e-8:
                    self._baseline_threshold = p1 + 3.0 * range_ * 0.1

        is_anomaly = current_score > threshold

        # Confidence: distance from threshold relative to std
        distance = abs(current_score - threshold)
        std_val = self.ema.std if self.ema.std else 1.0
        confidence = float(1.0) - np.exp(-distance / (std_val + 1e-8))

        return StreamingResult(
            score=current_score,
            is_anomaly=is_anomaly,
            threshold=threshold,
            confidence=confidence,
            ema_score=self.ema.mean,
            ema_threshold=self.ema.adaptive_threshold(self.z_threshold),
        )

    def _default_score(self, sample: np.ndarray) -> float:
        """Default scoring: sum of z-scores per feature."""
        x = np.asarray(sample, dtype=np.float64)
        if x.ndim == 1:
            x = x[:, None]
        if self._sample_count < 2:
            return 0.0
        # Use running mean/std
        mean = self.ema.mean if self._sample_count > 1 else 0.0
        std = self.ema.std if self._sample_count > 1 else 1.0
        z = np.abs(x).mean()
        return z / (std + 1e-8)


# ── Online BN-aware detector wrapper ──


class BNAdaptiveWrapper:
    """Wraps any ForeblocksAnomalyDetector with batch norm adaptation.

    At each predict() call, adapts BN layers before scoring.
    """

    def __init__(
        self,
        detector: Any,
        adapt_rate: float = 0.01,
        n_adapt_steps: int = 1,
    ) -> None:
        self.detector = detector
        if detector.model is not None:
            self.adapter = BatchNormAdapter(detector.model, adapt_rate=adapt_rate)
        else:
            self.adapter = None
        self.n_adapt_steps = n_adapt_steps

    def predict(self, series: np.ndarray) -> Any:
        """Adapt BN stats on first few samples, then predict."""
        if self.detector.model is None:
            raise RuntimeError("Detector is not fitted.")

        windows = self.detector._windows_from_series(series, fit_scaler=False)
        if len(windows) == 0:
            raise ValueError("No windows generated.")

        # Adapt BN on first few windows
        self.detector.model.eval()
        tensor = torch.from_numpy(windows[: min(10, len(windows))].astype(np.float32))
        for i in range(0, tensor.shape[0], max(1, tensor.shape[0] // 4)):
            batch = tensor[i : i + 1].to(self.detector.device)
            with torch.no_grad():
                self.detector.model(batch)
            if self.adapter is not None:
                self.adapter.adapt(batch)

        return self.detector.predict(series)
