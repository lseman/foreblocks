"""Explicit, arbitrarily ordered serial/parallel head graphs."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from foreblocks.modules.heads.config import (
    HeadComposerConfig,
    ParallelStageConfig,
    SerialStageConfig,
    StageKind,
)
from foreblocks.modules.heads.contracts import HeadShape
from foreblocks.modules.heads.execution import (
    build_stage_composer,
    execute_stage,
    infer_stage_shape,
)
from foreblocks.modules.heads.head_helper import HeadComposer
from foreblocks.modules.heads.head_types import HeadSpec, RunStateList


@dataclass(frozen=True)
class HeadStage:
    name: str
    kind: StageKind
    heads: tuple[HeadSpec, ...]
    parallel: ParallelStageConfig = field(default_factory=ParallelStageConfig)
    serial: SerialStageConfig = field(default_factory=SerialStageConfig)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("head stage name cannot be empty")
        if not self.heads:
            raise ValueError(f"head stage {self.name!r} must contain at least one head")


@dataclass(slots=True)
class HeadGraphState:
    stages: list[tuple[str, RunStateList]] = field(default_factory=list)


class HeadGraph(nn.Module):
    """Execute any ordered graph of serial and parallel head stages."""

    def __init__(
        self,
        stages: tuple[HeadStage, ...] | list[HeadStage],
        *,
        config: HeadComposerConfig | None = None,
        input_shape: HeadShape | None = None,
    ) -> None:
        super().__init__()
        if not stages:
            raise ValueError("HeadGraph requires at least one stage")
        self.stage_specs = tuple(stages)
        if len({stage.name for stage in stages}) != len(stages):
            raise ValueError("head stage names must be unique")
        self.config = config or HeadComposerConfig()
        self.stages = nn.ModuleList(
            [
                build_stage_composer(
                    stage.kind,
                    stage.heads,
                    composer=self.config,
                    parallel=stage.parallel,
                    serial=stage.serial,
                )
                for stage in stages
            ]
        )
        self.output_shape = self.infer_output_shape(input_shape) if input_shape else None

    def infer_output_shape(self, input_shape: HeadShape) -> HeadShape:
        shape = input_shape
        for stage in self.stage_specs:
            shape = infer_stage_shape(
                stage.kind, stage.heads, shape, stage.parallel
            )
        return shape

    def forward(
        self, value: torch.Tensor, encoder: nn.Module | None = None
    ) -> tuple[torch.Tensor, HeadGraphState]:
        state = HeadGraphState()
        current = value
        for spec, module in zip(self.stage_specs, self.stages, strict=True):
            if not isinstance(module, HeadComposer):
                raise TypeError("invalid head graph stage module")
            current, stage_state = execute_stage(module, current)
            state.stages.append((spec.name, stage_state))
        if encoder is not None:
            current = encoder(current)
        return current, state

    def inverse(self, value: torch.Tensor, state: HeadGraphState) -> torch.Tensor:
        if len(state.stages) != len(self.stages):
            raise ValueError("head graph state does not match the configured graph")
        current = value
        for module, (_, stage_state) in reversed(
            list(zip(self.stages, state.stages, strict=True))
        ):
            if isinstance(module, HeadComposer):
                current = module.inverse_post(current, stage_state)
        return current

    def architecture_regularization(self) -> torch.Tensor:
        losses = [
            stage.architecture_regularization(
                entropy_weight=self.config.nas.entropy_weight,
                expected_cost_weight=self.config.nas.expected_cost_weight,
            )
            for stage in self.stages
            if isinstance(stage, HeadComposer)
        ]
        if losses:
            return torch.stack(losses).sum()
        return torch.zeros(())

    def arch_parameters(self) -> Iterator[nn.Parameter]:
        for stage in self.stages:
            if isinstance(stage, HeadComposer):
                yield from stage.arch_parameters()

    def export_architecture(self) -> dict[str, tuple[str, ...]]:
        threshold = self.config.nas.discretize_threshold
        return {
            spec.name: module.export_architecture(threshold)
            for spec, module in zip(self.stage_specs, self.stages, strict=True)
            if isinstance(module, HeadComposer)
        }

    @torch.no_grad()
    def materialize(self, example: torch.Tensor) -> HeadShape:
        """Build lazy projections before optimizer or distributed wrapping."""
        current = example
        for stage in self.stages:
            if not isinstance(stage, HeadComposer):
                continue
            stage.warmup(current)
            current, _ = stage.forward_pre(current)
        self.output_shape = HeadShape(int(current.shape[-1]), int(current.shape[1]))
        return self.output_shape


__all__ = ["HeadGraph", "HeadGraphState", "HeadStage"]
