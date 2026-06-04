import torch

import pytest

from foreblocks.modules.moe.ff import FeedForwardBlock
from foreblocks.modules.moe.experts import moe
from foreblocks.modules.moe.experts.routers import RouterOutput


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
