import torch
import torch.nn as nn

from foreblocks.core.heads.head_helper import HeadComposer, HeadSpec


class IdentityInvert(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x + 0.0, None

    def invert(self, y: torch.Tensor, state: None) -> torch.Tensor:
        return y


class SplitAdd(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x * 0.5, x * 0.5


def test_headcomposer_invert_allows_none_state():
    x = torch.randn(2, 4, 3)
    spec = HeadSpec(
        head=IdentityInvert(),
        name="invert_none",
        combine="invert",
        alpha_mode="gate",
    )
    composer = HeadComposer(specs=[spec], enable_nas=True, composer_mode="serial")

    out, run_state = composer(x)
    recovered = composer.inverse_post(out, run_state)

    assert torch.allclose(recovered, x)
    assert len(run_state) == 1
    assert run_state[0].name == "invert_none"


def test_headcomposer_parallel_and_serial_with_nas():
    x = torch.randn(2, 5, 4)

    serial_spec = HeadSpec(
        head=SplitAdd(),
        name="serial_add",
        combine="add",
        alpha_mode="soft",
    )
    parallel_spec = HeadSpec(
        head=nn.Identity(),
        name="parallel_id",
        combine="none",
        alpha_mode="gate",
    )

    composer = HeadComposer(
        parallel_specs=[parallel_spec],
        serial_specs=[serial_spec],
        enable_nas=True,
        composer_mode="hybrid",
        parallel_combine="concat",
        parallel_align_mode="project",
        serial_none_merge="replace",
    )

    out, run_state = composer(x)
    assert out.shape[0] == x.shape[0]
    assert out.shape[1] == x.shape[1]
    assert len(run_state) == 2
    assert len(list(composer.arch_parameters())) > 0
    report = composer.alpha_report()
    assert "parallel_id" in report
    assert "serial_add" in report


def test_headcomposer_gumbel_alpha_mode():
    x = torch.randn(2, 3, 4)
    spec = HeadSpec(
        head=nn.Identity(),
        name="gumbel_head",
        combine="none",
        alpha_mode="gumbel",
        alpha_mix_style="blend",
    )
    composer = HeadComposer(specs=[spec], enable_nas=True, composer_mode="serial")

    out, _ = composer(x)
    assert out.shape == x.shape
    report = composer.alpha_report()
    assert "gumbel_head" in report
    assert "p_on" in report["gumbel_head"]


def test_headcomposer_gumbel_reporting_and_discretize_are_deterministic():
    spec = HeadSpec(
        head=nn.Identity(),
        name="gumbel_head",
        combine="none",
        alpha_mode="gumbel",
    )
    composer = HeadComposer(specs=[spec], enable_nas=True, composer_mode="serial")

    with torch.no_grad():
        composer.state_manager.alphas["gumbel_head"].copy_(torch.tensor([6.0, -6.0]))

    report_a = composer.alpha_report()
    report_b = composer.alpha_report()

    assert report_a["gumbel_head"]["w_head"] == report_b["gumbel_head"]["w_head"]
    assert report_a["gumbel_head"]["w_skip"] == report_b["gumbel_head"]["w_skip"]

    decision_a = composer.discretize_()
    composer.clear_discretization_()
    decision_b = composer.discretize_()

    assert decision_a == decision_b == {"gumbel_head": True}


def test_headcomposer_parallel_lora_mix():
    x = torch.randn(2, 6, 5)
    spec = HeadSpec(
        head=nn.Identity(),
        name="lora_mix_head",
        combine="none",
        alpha_mode="off",
        lora_rank=2,
    )

    composer = HeadComposer(
        parallel_specs=[spec],
        enable_nas=False,
        composer_mode="parallel",
        parallel_combine="lora_mix",
        parallel_align_mode="project",
    )
    out, _ = composer(x)
    assert out.shape == x.shape
