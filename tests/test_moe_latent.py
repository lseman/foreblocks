import torch

from foreblocks.tf.ff import FeedForwardBlock
from foreblocks.tf.experts import moe


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
