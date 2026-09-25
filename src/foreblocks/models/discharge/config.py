"""Configuration for the discharge classifier."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal


@dataclass
class DischargeClassifierConfig:
    fs: int = 256_000
    context_ms: float = 100.0
    frame_ms: float = 10.0

    waveform_feature_dim: int = 32
    waveform_kernels: tuple[int, ...] = (7, 15, 31, 63)
    waveform_blocks: int = 2

    envelope_hz: int = 2_000
    envelope_hidden_dim: int = 32
    envelope_levels: int = 4
    envelope_kernel_size: int = 5

    frame_pooling: Literal["mean", "attention"] = "mean"
    use_context_features: bool = True
    label_smoothing: float = 0.0
    gradient_clip: float = 5.0

    embedding_dim: int = 64
    dropout: float = 0.1

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    patience: int = 5
    device: str | None = None
    seed: int | None = 42
    # bf16 autocast for CUDA training steps (~1.5x faster); inference,
    # embeddings and novelty references stay in float32.
    mixed_precision: bool = False

    novelty_method: Literal["mahalanobis", "ecod", "copod", "cosine"] = "mahalanobis"
    novelty_quantile: float = 0.99

    class_names: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self):
        for name in (
            "fs",
            "envelope_hz",
            "waveform_feature_dim",
            "waveform_blocks",
            "envelope_hidden_dim",
            "envelope_levels",
            "envelope_kernel_size",
            "embedding_dim",
            "epochs",
            "batch_size",
            "patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ("frame_ms", "context_ms", "learning_rate", "gradient_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be nonnegative and finite.")
        for name in ("dropout", "label_smoothing"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1).")
        if not self.waveform_kernels or any(
            isinstance(k, bool) or not isinstance(k, Integral) or k < 1 or k % 2 == 0
            for k in self.waveform_kernels
        ):
            raise ValueError("waveform_kernels must contain positive odd integers.")
        if self.envelope_kernel_size % 2 == 0:
            raise ValueError(
                "envelope_kernel_size must be odd to preserve residual lengths."
            )
        if self.frame_pooling not in {"mean", "attention"}:
            raise ValueError("frame_pooling must be 'mean' or 'attention'.")
        if self.novelty_method not in {"mahalanobis", "ecod", "copod", "cosine"}:
            raise ValueError("Unknown novelty_method.")
        if (
            not math.isfinite(self.novelty_quantile)
            or not 0 < self.novelty_quantile <= 1
        ):
            raise ValueError("novelty_quantile must be in (0, 1].")
        if self.frame_size < 2 or self.context_size < self.frame_size:
            raise ValueError(
                "Frames need at least two samples and must fit in a context."
            )
        _ = self.n_frames

    @property
    def frame_size(self) -> int:
        return round(self.fs * self.frame_ms / 1000.0)

    @property
    def context_size(self) -> int:
        return round(self.fs * self.context_ms / 1000.0)

    @property
    def n_frames(self) -> int:
        n = self.context_size / self.frame_size
        if not n.is_integer():
            raise ValueError(
                f"context_ms={self.context_ms} is not a whole multiple of "
                f"frame_ms={self.frame_ms}"
            )
        return int(n)

    @property
    def envelope_decimation(self) -> int:
        if self.fs % self.envelope_hz != 0:
            raise ValueError(
                f"fs={self.fs} must be an integer multiple of envelope_hz="
                f"{self.envelope_hz}"
            )
        return self.fs // self.envelope_hz
