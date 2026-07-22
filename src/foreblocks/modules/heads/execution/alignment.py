"""Shape inference for serial and parallel head stages."""

from __future__ import annotations

from collections.abc import Sequence

from foreblocks.modules.heads.config import (
    ParallelFusion,
    ParallelStageConfig,
    StageKind,
)
from foreblocks.modules.heads.contracts import HeadShape, infer_head_shape
from foreblocks.modules.heads.head_types import HeadSpec


def infer_stage_shape(
    kind: StageKind,
    specs: Sequence[HeadSpec],
    input_shape: HeadShape,
    parallel: ParallelStageConfig,
) -> HeadShape:
    if kind is StageKind.SERIAL:
        shape = input_shape
        for spec in specs:
            shape = (
                HeadShape(spec.output_dim, shape.sequence_length)
                if spec.output_dim is not None
                else infer_head_shape(spec.head, shape)
            )
        return shape

    branch_shapes = [
        HeadShape(spec.output_dim, input_shape.sequence_length)
        if spec.output_dim is not None
        else infer_head_shape(spec.head, input_shape)
        for spec in specs
    ]
    if parallel.output_dim is not None:
        return HeadShape(parallel.output_dim, input_shape.sequence_length)
    if parallel.fusion is ParallelFusion.CONCAT:
        return HeadShape(
            sum(shape.features for shape in branch_shapes), input_shape.sequence_length
        )
    dimensions = {shape.features for shape in branch_shapes}
    if len(dimensions) > 1 and parallel.alignment.value == "strict":
        raise ValueError(f"parallel branches have incompatible shapes: {sorted(dimensions)}")
    return HeadShape(next(iter(dimensions), input_shape.features), input_shape.sequence_length)


__all__ = ["infer_stage_shape"]
