"""Construction and execution of individual head-graph stages."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from foreblocks.modules.heads.config import (
    HeadComposerConfig,
    ParallelStageConfig,
    SerialStageConfig,
    StageKind,
)
from foreblocks.modules.heads.head_helper import HeadComposer
from foreblocks.modules.heads.head_types import HeadSpec, RunStateList


def build_stage_composer(
    kind: StageKind,
    specs: Sequence[HeadSpec],
    *,
    composer: HeadComposerConfig,
    parallel: ParallelStageConfig,
    serial: SerialStageConfig,
) -> HeadComposer:
    common = dict(
        output_dim=composer.output_dim,
        stop_gradient_on_carry=composer.stop_gradient_on_carry,
        enable_nas=composer.nas.enabled,
        alpha_temperature=composer.nas.temperature,
        gumbel_temperature=composer.nas.gumbel_temperature,
        anneal_alpha=composer.nas.anneal,
        use_spectral_norm_invert=composer.spectral_norm_invert,
    )
    if kind is StageKind.SERIAL:
        return HeadComposer(
            specs=list(specs),
            composer_mode="serial",
            serial_none_merge=serial.merge.value,
            serial_none_project=serial.project_output,
            serial_none_project_dim=serial.output_dim,
            **common,
        )
    return HeadComposer(
        specs=list(specs),
        composer_mode="parallel",
        parallel_combine=parallel.fusion.value,
        parallel_align_mode=parallel.alignment.value,
        parallel_project=parallel.project_output,
        parallel_project_dim=parallel.output_dim,
        parallel_structured_outputs=parallel.structured_outputs.value,
        parallel_hyper_hidden_dim=parallel.hidden_dim,
        parallel_attention_heads=parallel.attention_heads,
        parallel_fusion_dropout=parallel.dropout,
        moe_temperature=parallel.moe_temperature,
        **common,
    )


def execute_stage(
    composer: HeadComposer, value: torch.Tensor
) -> tuple[torch.Tensor, RunStateList]:
    return composer.forward_pre(value)


__all__ = ["build_stage_composer", "execute_stage"]
