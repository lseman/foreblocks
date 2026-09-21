"""Typed configuration for composable forecasting heads."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class StageKind(StrEnum):
    SERIAL = "serial"
    PARALLEL = "parallel"


class ParallelFusion(StrEnum):
    CONCAT = "concat"
    SUM = "sum"
    MEAN = "mean"
    WEIGHTED_SUM = "weighted_sum"
    HYPERNETWORK = "hypernetwork_mix"
    ATTENTION = "attention_fusion"
    GATED = "gated_fusion"
    LORA = "lora_mix"
    MOE = "moe_routing"
    MULTISCALE_ATTENTION = "multi_scale_attn"


class AlignmentMode(StrEnum):
    STRICT = "strict"
    PROJECT = "project"


class SerialMerge(StrEnum):
    REPLACE = "replace"
    ADD = "add"
    CONCAT = "concat"
    LORA_RESIDUAL = "lora_residual"


class StructuredOutputPolicy(StrEnum):
    ERROR = "error"
    MAIN = "main"
    MAIN_ADD_SECOND = "main_add_second"


class NASMode(StrEnum):
    NONE = "none"
    GATE = "gate"
    SOFT = "soft"
    GUMBEL = "gumbel"


@dataclass(frozen=True, slots=True)
class HeadNASConfig:
    enabled: bool = False
    temperature: float = 1.0
    gumbel_temperature: float = 0.5
    anneal: bool = True
    entropy_weight: float = 0.0
    expected_cost_weight: float = 0.0
    discretize_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.temperature <= 0 or self.gumbel_temperature <= 0:
            raise ValueError("NAS temperatures must be positive")
        if not 0 <= self.discretize_threshold <= 1:
            raise ValueError("discretize_threshold must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ParallelStageConfig:
    fusion: ParallelFusion = ParallelFusion.CONCAT
    alignment: AlignmentMode = AlignmentMode.STRICT
    project_output: bool = False
    output_dim: int | None = None
    structured_outputs: StructuredOutputPolicy = StructuredOutputPolicy.ERROR
    attention_heads: int = 0
    hidden_dim: int = 64
    dropout: float = 0.0
    moe_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class SerialStageConfig:
    merge: SerialMerge = SerialMerge.REPLACE
    project_output: bool = False
    output_dim: int | None = None


@dataclass(frozen=True, slots=True)
class HeadComposerConfig:
    output_dim: int | None = None
    stop_gradient_on_carry: bool = False
    spectral_norm_invert: bool = True
    nas: HeadNASConfig = field(default_factory=HeadNASConfig)


__all__ = [
    "AlignmentMode",
    "HeadComposerConfig",
    "HeadNASConfig",
    "NASMode",
    "ParallelFusion",
    "ParallelStageConfig",
    "SerialMerge",
    "SerialStageConfig",
    "StageKind",
    "StructuredOutputPolicy",
]
