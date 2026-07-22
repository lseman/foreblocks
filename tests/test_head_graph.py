import torch
from torch import nn

from foreblocks.modules.heads import (
    AlignmentMode,
    HeadComposerConfig,
    HeadGraph,
    HeadNASConfig,
    HeadOutput,
    HeadShape,
    HeadSpec,
    HeadStage,
    ParallelFusion,
    ParallelStageConfig,
    StageKind,
)
from foreblocks.modules.heads.nas import (
    CosineTemperatureSchedule,
    HeadNASController,
)


class ModernResidualHead(nn.Module):
    def forward(self, value: torch.Tensor) -> HeadOutput:
        return HeadOutput(value=value * 0.5, residual=value * 0.5)


def test_head_graph_supports_arbitrary_serial_parallel_order() -> None:
    graph = HeadGraph(
        [
            HeadStage(
                "pre",
                StageKind.SERIAL,
                (HeadSpec(nn.Identity(), "pre_identity", output_dim=4),),
            ),
            HeadStage(
                "features",
                StageKind.PARALLEL,
                (
                    HeadSpec(nn.Identity(), "branch_a", output_dim=4),
                    HeadSpec(nn.Identity(), "branch_b", output_dim=4),
                ),
                parallel=ParallelStageConfig(
                    fusion=ParallelFusion.CONCAT,
                    alignment=AlignmentMode.STRICT,
                ),
            ),
            HeadStage(
                "post",
                StageKind.SERIAL,
                (HeadSpec(nn.Identity(), "post_identity", output_dim=8),),
            ),
        ],
        input_shape=HeadShape(4, 6),
    )

    output, state = graph(torch.randn(2, 6, 4))

    assert output.shape == (2, 6, 8)
    assert graph.output_shape == HeadShape(8, 6)
    assert [name for name, _ in state.stages] == ["pre", "features", "post"]


def test_head_graph_nas_regularization_schedule_and_export() -> None:
    graph = HeadGraph(
        [
            HeadStage(
                "search",
                StageKind.SERIAL,
                (
                    HeadSpec(
                        nn.Identity(),
                        "candidate",
                        alpha_mode="gumbel",
                        parameter_cost=2.0,
                        latency_cost=3.0,
                    ),
                ),
            )
        ],
        config=HeadComposerConfig(
            nas=HeadNASConfig(
                enabled=True,
                entropy_weight=0.1,
                expected_cost_weight=0.2,
            )
        ),
    )
    controller = HeadNASController(
        graph, CosineTemperatureSchedule(initial=1.0, final=0.1, steps=10)
    )

    assert controller.step(10) == 0.1
    loss = controller.loss()
    assert loss.requires_grad
    assert loss.item() > 0
    assert controller.export() == {"search": ("candidate",)}
    assert len(list(graph.arch_parameters())) == 1


def test_head_graph_materializes_and_reports_runtime_shape() -> None:
    graph = HeadGraph(
        [
            HeadStage(
                "parallel",
                StageKind.PARALLEL,
                (HeadSpec(nn.Identity(), "identity"),),
                parallel=ParallelStageConfig(
                    fusion=ParallelFusion.SUM,
                    alignment=AlignmentMode.PROJECT,
                    project_output=True,
                    output_dim=5,
                ),
            )
        ]
    )

    assert graph.materialize(torch.randn(2, 7, 3)) == HeadShape(5, 7)


def test_head_output_contract_works_with_existing_composer() -> None:
    graph = HeadGraph(
        [
            HeadStage(
                "residual",
                StageKind.SERIAL,
                (HeadSpec(ModernResidualHead(), "modern", combine="add"),),
            )
        ]
    )
    value = torch.randn(2, 5, 3)
    output, state = graph(value)
    assert torch.allclose(graph.inverse(output, state), value)
