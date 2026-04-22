from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.nn as nn

BaseHeadLike = nn.Module

CombineMode = Literal["invert", "add", "none", "lora_residual"]
AlphaMode = Literal["off", "gate", "soft", "gumbel"]
AlphaMixStyle = Literal["blend", "residual", "lora"]
AlphaWeights = tuple[torch.Tensor | None, torch.Tensor | None]

ComposerMode = Literal["serial", "parallel", "hybrid"]
ParallelCombine = Literal[
    "concat",
    "sum",
    "mean",
    "weighted_sum",
    "hypernetwork_mix",
    "attention_fusion",
    "gated_fusion",
    "lora_mix",
    "moe_routing",
    "multi_scale_attn",
]
ParallelAlignMode = Literal["strict", "project"]
SerialNoneMerge = Literal["replace", "add", "concat", "lora_residual"]
ParallelStructuredOutputs = Literal["error", "main", "main_add_second"]


@dataclass
class BaseRunState:
    kind: str
    name: str
    stage: Literal["parallel", "serial"]
    base_dim: int | None = None


@dataclass
class ParallelNoneState(BaseRunState):
    kind: Literal["parallel_none"] = field(default="parallel_none", init=False)
    mix_w_head: float | None = None


@dataclass
class SerialInvertState(BaseRunState):
    kind: Literal["serial_invert"] = field(default="serial_invert", init=False)
    gate_on: bool = True
    gate_value: float | None = None
    state: Any = None
    head_ref: nn.Module | None = None


@dataclass
class SerialAddState(BaseRunState):
    kind: Literal["serial_add"] = field(default="serial_add", init=False)
    carry: torch.Tensor | None = None
    carry_shape: tuple[int, ...] = ()
    mix_w_head: float | None = None
    add_project: bool = True


@dataclass
class SerialNoneState(BaseRunState):
    kind: Literal["serial_none"] = field(default="serial_none", init=False)
    mix_w_head: float | None = None
    serial_none_merge: SerialNoneMerge | None = None
    serial_none_project_dim: int | None = None
    serial_none_time_aligned: bool = False


RunStateEntry = (
    ParallelNoneState
    | SerialInvertState
    | SerialAddState
    | SerialNoneState
    | BaseRunState
)
RunStateList = list[RunStateEntry]


@dataclass
class HeadSpec:
    head: BaseHeadLike
    name: str
    combine: CombineMode = "none"
    add_project: bool = True
    custom_add_proj: nn.Module | None = None
    alpha_mode: AlphaMode = "off"
    alpha_init: float = 0.0
    alpha_trainable: bool = True
    weight_carry: bool = True
    alpha_mix_style: AlphaMixStyle = "blend"

    lora_rank: int | None = None
    moe_k: int = 2
    spectral_norm: bool = False
    fusion_scale_init: float = 1.0

    def __post_init__(self):
        valid_combine = ("invert", "add", "none", "lora_residual")
        valid_alpha = ("off", "gate", "soft", "gumbel")
        valid_alpha_mix = ("blend", "residual", "lora")

        if self.combine not in valid_combine:
            raise ValueError(
                f"HeadSpec '{self.name}': combine must be one of {valid_combine}, got '{self.combine}'"
            )
        if self.alpha_mode not in valid_alpha:
            raise ValueError(
                f"HeadSpec '{self.name}': alpha_mode must be one of {valid_alpha}, got '{self.alpha_mode}'"
            )
        if self.alpha_mix_style not in valid_alpha_mix:
            raise ValueError(
                f"HeadSpec '{self.name}': alpha_mix_style must be one of {valid_alpha_mix}, "
                f"got '{self.alpha_mix_style}'"
            )
        if not self.name:
            raise ValueError("HeadSpec name cannot be empty")
        if self.custom_add_proj is not None and self.combine != "add":
            import warnings

            warnings.warn(
                f"HeadSpec '{self.name}': custom_add_proj is set but combine='{self.combine}' "
                "(only used when combine is 'add')"
            )
        if self.spectral_norm and self.combine != "invert":
            import warnings

            warnings.warn(
                f"HeadSpec '{self.name}': spectral_norm only applies to combine='invert' heads"
            )
        if self.lora_rank is not None and self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive if set")

    def __repr__(self) -> str:
        return (
            f"HeadSpec(name={self.name!r}, combine={self.combine!r}, "
            f"alpha_mode={self.alpha_mode!r}, alpha_mix_style={self.alpha_mix_style!r}, "
            f"add_project={self.add_project})"
        )


@dataclass
class ActiveHead:
    spec: HeadSpec
    index: int
    is_parallel: bool
    enabled: bool = True
    hardened: bool | None = None
    alpha_param: nn.Parameter | None = None
    mix_logit: nn.Parameter | None = None
    fusion_scale_param: nn.Parameter | None = None
    lora_down: nn.Module | None = None
    lora_up: nn.Module | None = None
    moe_gate: nn.Module | None = None

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def stage(self) -> Literal["parallel", "serial"]:
        return "parallel" if self.is_parallel else "serial"
