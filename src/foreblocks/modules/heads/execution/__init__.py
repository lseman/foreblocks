"""Reusable serial and parallel head-stage execution."""

from foreblocks.modules.heads.execution.alignment import infer_stage_shape
from foreblocks.modules.heads.execution.outputs import HeadOutput, normalize_head_output
from foreblocks.modules.heads.execution.stages import (
    build_stage_composer,
    execute_stage,
)

__all__ = [
    "HeadOutput",
    "build_stage_composer",
    "execute_stage",
    "infer_stage_shape",
    "normalize_head_output",
]
