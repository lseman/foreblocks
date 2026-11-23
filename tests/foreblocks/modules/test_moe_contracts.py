"""Regression checks for MoE padding, routing, and caller precision contracts."""

import copy

import pytest
import torch

from foreblocks.nn.moe import layer as moe


@pytest.fixture
def make_moe(monkeypatch):
    monkeypatch.setattr(moe, "optimized_topk_routing", moe.eager_topk_routing)

    def make(**kwargs):
        options = dict(
            d_model=8,
            d_ff=16,
            num_experts=4,
            top_k=2,
            compile_router=False,
            use_grouped_kernel=False,
            use_fused_router_topk=False,
            jitter=0.0,
        )
        options.update(kwargs)
        return moe.MoEFeedForwardDMoE(**options)

    return make


@pytest.mark.parametrize("all_padding", [False, True])
def test_nan_padding_matches_zero_padding_gradients(make_moe, all_padding):
    model = make_moe().eval()
    reference = copy.deepcopy(model)
    mask = torch.tensor([[False, True, False], [True, False, True]])
    if all_padding:
        mask.fill_(True)
    x = torch.randn(2, 3, 8)
    dirty = x.masked_fill(mask[..., None], float("nan")).requires_grad_()
    clean = x.masked_fill(mask[..., None], 0).requires_grad_()
    y, aux = model(dirty, padding_mask=mask)
    expected, expected_aux = reference(clean, padding_mask=mask)
    torch.testing.assert_close(y, expected)
    assert torch.count_nonzero(y[mask]) == 0
    (y.square().sum() + aux).backward()
    (expected.square().sum() + expected_aux).backward()
    torch.testing.assert_close(dirty.grad, clean.grad)
    for parameter, other in zip(model.parameters(), reference.parameters()):
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, other.grad)


def test_soft_dense_matches_all_expert_reference(make_moe):
    model = make_moe(router_type="soft_dense", router_temperature=0.7).eval()
    x = torch.randn(2, 3, 8, requires_grad=True)
    y, _, state = model(x, return_router_state=True)
    flat = model.input_norm(x).reshape(-1, 8)
    probabilities = model.router(flat).top_p
    expected = sum(
        model.experts.forward_expert(i, flat) * probabilities[:, i : i + 1]
        for i in range(model.num_routed_experts)
    ).reshape_as(x)
    torch.testing.assert_close(y, expected)
    assert state.indices.shape == (6, 4)
    assert model._last_num_assignments == 24
    actual_grads = torch.autograd.grad(
        y.square().sum(), (x, *model.parameters()), allow_unused=True
    )
    reference_grads = torch.autograd.grad(
        expected.square().sum(), (x, *model.parameters()), allow_unused=True
    )
    for actual, reference in zip(actual_grads, reference_grads):
        if actual is not None:
            torch.testing.assert_close(actual, reference)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("autocast_dtype", [None, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("configuration", ["plain", "latent", "concat"])
@pytest.mark.parametrize("grouped", [False, True])
def test_cuda_padding_respects_caller_precision(
    make_moe, autocast_dtype, configuration, grouped
):
    options = {"use_grouped_kernel": grouped}
    if configuration == "latent":
        options.update(moe_use_latent=True, moe_latent_dim=4)
    elif configuration == "concat":
        options.update(num_shared=1, shared_combine="concat")
    model = make_moe(**options).cuda().eval()
    x = torch.randn(2, 3, 8, device="cuda", requires_grad=True)
    mask = torch.tensor([[False, True, False], [True, False, True]], device="cuda")
    router_dtypes = []
    hook = model.router.register_forward_hook(
        lambda module, args, out: router_dtypes.append(out.logits.dtype)
    )
    with torch.autocast(
        "cuda",
        dtype=autocast_dtype or torch.float16,
        enabled=autocast_dtype is not None,
    ):
        y, aux = model(x, padding_mask=mask)
        compact, _ = model(x[~mask].unsqueeze(0))
    hook.remove()
    assert router_dtypes == [autocast_dtype or torch.float32] * 2
    assert y.dtype == compact.dtype
    torch.testing.assert_close(y[~mask], compact.squeeze(0))
    assert torch.count_nonzero(y[mask]) == 0
    (y.float().square().sum() + aux.float()).backward()
    assert torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_bf16_autocast_helper_reports_active_policy():
    from foreblocks.nn.moe.utils import autocast_bf16_enabled

    assert not autocast_bf16_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert autocast_bf16_enabled("cpu")
    with torch.autocast("cpu", dtype=torch.float16):
        assert not autocast_bf16_enabled("cpu")


@pytest.mark.parametrize("slots", [1, 4])
def test_token_conditioned_auxiliary_router_is_token_local(make_moe, slots):
    torch.manual_seed(13)
    model = make_moe(router_type="token_conditioned_auxiliary", router_aux_num_aux=slots)
    router = model.router
    x = torch.randn(5, 8)
    altered = x.clone()
    altered[2] = -x[2]
    logits = router(x).logits
    changed = router(altered).logits
    assert not torch.allclose(logits[2], changed[2])
    torch.testing.assert_close(logits[[0, 1, 3, 4]], changed[[0, 1, 3, 4]])
    logits.square().sum().backward()
    assert router.aux_tokens.grad.abs().sum() > 0
