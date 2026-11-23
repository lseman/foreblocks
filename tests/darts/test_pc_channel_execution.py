import pytest
import torch

from darts.architecture.darts.mixed_op import MixedOp


@pytest.mark.parametrize("candidate", ["TimeConv", "ResidualMLP"])
def test_narrow_candidate_executes_and_transfers_full_width_parameters(candidate):
    edge = MixedOp(
        8, 8, 6, [candidate, "Identity"], use_hierarchical=False,
        adaptive_sampling=False, op_gdas=False, use_drnas=False, use_gumbel=False,
        drop_prob=0.0, pc_ratio=0.25,
    )
    edge.pc_darts_enabled = True
    edge.narrow_candidate_execution = True
    x = torch.randn(3, 6, 8, requires_grad=True)
    observed = []

    def capture(module, args, kwargs):
        observed.append((args[0].shape[-1], kwargs["channels"]))

    hook = edge.ops[0].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = edge(x)
    finally:
        hook.remove()
    width, channels = observed[0]
    assert width == 2
    bypass = torch.ones(8, dtype=torch.bool)
    bypass[channels] = False
    torch.testing.assert_close(result[..., bypass], x[..., bypass])
    result.square().mean().backward()
    assert edge._alphas.grad.abs().sum() > 0
    scale_grad = edge.ops[0].norm.scale.grad
    assert torch.count_nonzero(scale_grad[bypass]) == 0
    assert torch.count_nonzero(scale_grad[channels]) == width
    assert edge.ops[0].eval()(x.detach()).shape == x.shape
