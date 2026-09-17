import pytest
import torch

from foreblocks.modules.moe.experts import moe
from foreblocks.modules.moe.experts.dispatchers import (
    ConfidenceCapacityDispatcher,
    DroplessPackedDispatcher,
)
from foreblocks.modules.moe.experts.moe import MoERoutingState
from foreblocks.modules.moe.experts.routers import RouterOutput
from foreblocks.modules.moe.ff import FeedForwardBlock


def test_latent_moe_preserves_output_shape_and_aux():
    block = FeedForwardBlock(
        d_model=16,
        dim_ff=32,
        use_moe=True,
        num_experts=4,
        num_shared=1,
        top_k=2,
        use_triton=False,
        use_grouped_kernel=False,
        use_fused_router_topk=False,
        compile_router=False,
        compile_experts=False,
        moe_use_latent=True,
        moe_latent_dim=8,
        moe_latent_d_ff=16,
    )
    x = torch.randn(2, 5, 16)

    out, aux = block(x, return_aux_loss=True)

    assert out.shape == x.shape
    assert aux.ndim == 0
    assert block.block.moe_use_latent is True
    assert block.block.routed_d_model == 8
    assert block.block.routed_d_ff == 16


def test_latent_moe_token_choice_without_shared_path():
    block = FeedForwardBlock(
        d_model=12,
        dim_ff=24,
        use_moe=True,
        num_experts=3,
        num_shared=0,
        top_k=1,
        use_triton=False,
        use_grouped_kernel=False,
        use_fused_router_topk=False,
        compile_router=False,
        compile_experts=False,
        moe_use_latent=True,
        moe_latent_dim=6,
    )
    x = torch.randn(3, 4, 12)

    y = block(x)

    assert y.shape == x.shape


def test_moe_falls_back_when_scripted_topk_router_fails(monkeypatch):
    block = FeedForwardBlock(
        d_model=12,
        dim_ff=24,
        use_moe=True,
        num_experts=3,
        num_shared=0,
        top_k=2,
        use_triton=False,
        use_grouped_kernel=False,
        use_fused_router_topk=False,
        compile_router=False,
        compile_experts=False,
    )
    x = torch.randn(2, 4, 12)

    def broken_topk(logits, k):
        raise RuntimeError(
            "nvrtc: error: failed to open libnvrtc-builtins.so.13.0"
        )

    monkeypatch.setattr(moe, "optimized_topk_routing", broken_topk)

    y = block(x)

    assert y.shape == x.shape
    assert block.block._force_eager_router_topk is True


# ─── SOTA feature tests ────────────────────────────────────────────────

def test_moe_aux_loss_computed():
    """MoE returns non-zero aux loss when moe_aux_loss_weight > 0."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=4, num_shared=0,
        top_k=2, use_triton=False,
        moe_aux_loss_weight=0.01,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    _, aux = block(x, return_aux_loss=True)
    assert aux > 0


def test_moe_balancing_loss_has_switch_transformer_scale():
    """Uniform routing has a unit unweighted loss, independent of expert count."""
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        top_k=2, use_triton=False, compile_router=False, compile_experts=False,
        moe_aux_loss_weight=1.0, z_loss_weight=0.0,
        moe_entropy_reg_weight=0.0,
    ).block
    logits = torch.zeros(8, 4)
    # Two assignments per token, distributed uniformly across all experts.
    assignments = torch.tensor([0, 1, 2, 3] * 4)
    loss = block._compute_aux_loss(logits, 8, assignments)
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_moe_aux_loss_uses_float32_router_probabilities():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        top_k=1, use_triton=False, compile_router=False, compile_experts=False,
        moe_aux_loss_weight=1.0, z_loss_weight=0.0,
    ).block
    logits = torch.zeros(4, 4, dtype=torch.bfloat16)
    assignments = torch.arange(4)
    loss = block._compute_aux_loss(logits, 4, assignments)
    assert loss.dtype == torch.float32


def test_dropless_dispatcher_keeps_over_capacity_assignments():
    dispatcher = DroplessPackedDispatcher(num_experts=2, top_k=1, capacity_factor=0.1)
    x = torch.randn(8, 4)
    top_i = torch.zeros(8, 1, dtype=torch.long)
    top_p = torch.ones(8, 1)
    packed_x, _, experts, tokens, _, dropped = dispatcher.pack(x, top_p, top_i)
    assert packed_x.size(0) == 8
    assert experts.numel() == tokens.numel() == 8
    assert dropped == 0


def test_capacity_dispatcher_prunes_weakest_routes_per_expert():
    dispatcher = ConfidenceCapacityDispatcher(
        num_experts=2, top_k=1, capacity_factor=0.5
    )
    x = torch.arange(8, dtype=torch.float32).unsqueeze(1)
    top_i = torch.tensor([[0], [0], [0], [0], [1], [1], [1], [1]])
    top_p = torch.tensor([[0.1], [0.9], [0.2], [0.8], [0.3], [0.7], [0.4], [0.6]])
    _, kept_weights, experts, tokens, _, dropped = dispatcher.pack(x, top_p, top_i)
    assert dropped == 4
    assert experts.tolist() == [0, 0, 1, 1]
    assert set(tokens.tolist()) == {1, 3, 5, 7}
    assert sorted(kept_weights[:, 0].tolist()) == pytest.approx([0.6, 0.7, 0.8, 0.9])


def test_moe_exposes_structured_router_state_without_breaking_default_api():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        top_k=2, use_triton=False, use_grouped_kernel=False,
        compile_router=False, compile_experts=False, jitter=0.0,
    )
    x = torch.randn(2, 3, 8, requires_grad=True)
    out, aux, state = block(
        x, return_aux_loss=True, return_router_state=True
    )
    assert out.shape == x.shape
    assert isinstance(state, MoERoutingState)
    assert state.logits.shape == (6, 4)
    assert state.indices.shape == state.weights.shape == (6, 2)
    assert state.aux_loss is aux
    assert state.logits.requires_grad
    assert block.block.last_routing_state is not None
    assert not block.block.last_routing_state.logits.requires_grad


def test_moe_confidence_capacity_pruning_is_opt_in():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
        top_k=1, use_triton=False, use_grouped_kernel=False,
        compile_router=False, compile_experts=False,
        moe_capacity_pruning=True, moe_capacity_factor=0.5,
    )
    _, state = block(torch.randn(1, 8, 8), return_router_state=True)
    assert isinstance(block.block.dispatcher, ConfidenceCapacityDispatcher)
    assert state.tokens_dropped > 0


def test_moe_padding_mask_excludes_tokens_and_zeroes_their_delta():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=1,
        top_k=2, use_triton=False, use_grouped_kernel=False,
        compile_router=False, compile_experts=False, jitter=0.0,
    )
    block.eval()
    x = torch.randn(2, 4, 8)
    padding_mask = torch.tensor(
        [[False, False, True, True], [False, True, False, True]]
    )
    out = block(x, padding_mask=padding_mask)
    assert torch.count_nonzero(out[padding_mask]) == 0
    assert block.block._last_num_assignments == int((~padding_mask).sum()) * 2


def test_moe_padding_mask_shape_is_validated():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
        use_triton=False, compile_router=False, compile_experts=False,
    )
    with pytest.raises(ValueError, match="padding_mask shape"):
        block(torch.randn(2, 3, 8), padding_mask=torch.zeros(2, 2, dtype=torch.bool))


def test_balancing_loss_uses_pre_dispatch_decisions():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        top_k=1, use_triton=False, compile_router=False, compile_experts=False,
        moe_aux_loss_weight=1.0, z_loss_weight=0.0,
    ).block
    logits = torch.zeros(4, 4)
    post_dispatch = torch.zeros(4, dtype=torch.long)
    pre_dispatch = torch.arange(4).unsqueeze(1)
    loss = block._compute_aux_loss(
        logits, 4, post_dispatch, routing_indices=pre_dispatch
    )
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_entropy_regularization_penalizes_router_collapse():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        top_k=1, use_triton=False, compile_router=False, compile_experts=False,
        moe_aux_loss_weight=0.0, z_loss_weight=0.0,
        moe_entropy_reg_weight=1.0,
    ).block
    assignments = torch.zeros(4, dtype=torch.long)
    uniform = block._compute_aux_loss(torch.zeros(4, 4), 4, assignments)
    collapsed_logits = torch.tensor([[20.0, -20.0, -20.0, -20.0]]).expand(4, -1)
    collapsed = block._compute_aux_loss(collapsed_logits, 4, assignments)
    assert collapsed > uniform
    torch.testing.assert_close(uniform, torch.tensor(0.0))


def test_loss_weight_alias_is_canonical_and_conflicts_are_rejected():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
        use_triton=False, compile_router=False, compile_experts=False,
        load_balance_weight=0.25,
    ).block
    assert block.load_balance_weight == block.moe_aux_loss_weight == 0.25
    with pytest.raises(ValueError, match="are aliases"):
        FeedForwardBlock(
            d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
            use_triton=False, compile_router=False, compile_experts=False,
            load_balance_weight=0.1, moe_aux_loss_weight=0.2,
        )


def test_expert_weights_are_available_as_differentiable_packed_tensors():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=4, num_shared=0,
        use_triton=False, compile_router=False, compile_experts=False,
    ).block
    packed_up, packed_down = block.get_packed_expert_weights()
    assert packed_up.shape == (4, 8, 32)
    assert packed_down.shape == (4, 16, 8)
    (packed_up.sum() + packed_down.sum()).backward()
    assert block.experts.w12.grad is not None
    assert block.experts.w3.grad is not None
    assert packed_up.data_ptr() == block.experts.w12.data_ptr()


def test_legacy_per_expert_checkpoint_loads_into_packed_bank():
    block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
        use_triton=False, compile_router=False, compile_experts=False,
    ).block
    state = block.state_dict()
    packed_up = state.pop("experts.w12")
    packed_down = state.pop("experts.w3")
    for idx in range(2):
        state[f"experts.{idx}.w12.weight"] = packed_up[idx].t().contiguous()
        state[f"experts.{idx}.w3.weight"] = packed_down[idx].t().contiguous()
    restored = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2, num_shared=0,
        use_triton=False, compile_router=False, compile_experts=False,
    ).block
    restored.load_state_dict(state)
    assert torch.equal(restored.experts.w12, packed_up)
    assert torch.equal(restored.experts.w3, packed_down)


def test_shared_expert_is_optional_and_gated_when_enabled():
    default_block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=2,
        use_triton=False, compile_router=False,
    ).block
    assert default_block.num_shared == 0
    gated_block = FeedForwardBlock(
        d_model=8, dim_ff=16, use_moe=True, num_experts=3, num_shared=1,
        shared_gate=True, use_triton=False, compile_router=False,
    ).block
    assert gated_block.shared_gate is not None


def test_moe_z_loss_computed():
    """MoE includes Z-loss when z_loss_weight > 0."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=4, num_shared=0,
        top_k=2, use_triton=False,
        z_loss_weight=0.001,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    _, aux = block(x, return_aux_loss=True)
    assert aux > 0


def test_moe_aux_loss_disabled():
    """MoE returns zero aux loss when all aux weights are 0."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=4, num_shared=0,
        top_k=2, use_triton=False,
        moe_aux_loss_weight=0.0, z_loss_weight=0.0,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    _, aux = block(x, return_aux_loss=True)
    assert aux == 0.0


def test_moe_group_wise_load_balancing():
    """MoE computes group-wise aux loss with moe_num_groups > 1."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=8, num_shared=0,
        top_k=2, use_triton=False,
        moe_aux_loss_weight=0.01, moe_num_groups=4,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    _, aux = block(x, return_aux_loss=True)
    stats = block.block.get_expert_stats()
    assert "group_usage" in stats
    assert "group_uniform_mse" in stats
    assert aux > 0


def test_moe_soft_capacity():
    """MoE with soft capacity uses elastic capacity scaling."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=4, num_shared=1,
        top_k=2, use_triton=False, moe_soft_capacity=True,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    # Warm up usage stats
    for _ in range(3):
        _ = block(x, return_aux_loss=False)
    out, aux = block(x, return_aux_loss=True)
    assert out.shape == x.shape


def test_moe_entropy_reg():
    """MoE includes entropy regularization when moe_entropy_reg_weight > 0."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=8, num_shared=0,
        top_k=2, use_triton=False,
        moe_aux_loss_weight=0.01, moe_entropy_reg_weight=0.001,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    _, aux = block(x, return_aux_loss=True)
    assert aux > 0


def test_moe_full_sota_combo():
    """MoE with all SOTA features: aux, z_loss, groups, soft capacity, entropy."""
    block = FeedForwardBlock(
        d_model=16, dim_ff=32, use_moe=True, num_experts=8, num_shared=0,
        top_k=2, use_triton=False,
        moe_aux_loss_weight=0.01, z_loss_weight=0.001,
        moe_num_groups=4, moe_soft_capacity=True,
        moe_entropy_reg_weight=0.001,
    )
    block.train()
    x = torch.randn(2, 8, 16)
    out, aux = block(x, return_aux_loss=True)
    stats = block.block.get_expert_stats()
    assert out.shape == x.shape
    assert aux > 0
    assert "group_usage" in stats
    assert "group_uniform_mse" in stats


def test_moe_router_output_is_dataclass():
    """Router forward returns RouterOutput dataclass, not tuple."""
    from foreblocks.modules.moe.experts.routers import LinearRouter
    router = LinearRouter(d_model=16, num_experts=4)
    x = torch.randn(2, 8, 16)
    out = router(x)
    assert isinstance(out, RouterOutput)
    assert hasattr(out, "logits")
    assert hasattr(out, "per_token_k")
    assert hasattr(out, "k_logits")
    assert hasattr(out, "k_probs")
    assert hasattr(out, "top_p")
    assert hasattr(out, "top_i")
    assert hasattr(out, "router_entropy")


def test_router_entropy_stays_tensor_in_training_forward():
    """Router entropy does not force a Python scalar inside compiled forwards."""
    from foreblocks.modules.moe.experts.routers import AdaptiveNoisyTopKRouter

    router = AdaptiveNoisyTopKRouter(d_model=16, num_experts=4, max_k=2)
    router.train()
    x = torch.randn(8, 16)

    out = router(x, return_raw_logits=True)

    assert isinstance(out.router_entropy, torch.Tensor)
    assert out.router_entropy.ndim == 0
    assert out.router_entropy.requires_grad is False
