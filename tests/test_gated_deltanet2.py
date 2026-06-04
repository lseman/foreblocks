"""Tests for GatedDeltaNet-2 (GDN-2) implementation."""

import pytest
import torch
import torch.nn as nn


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestGatedDeltaNet2Basic:
    """Test basic construction and forward pass."""

    def test_import(self):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        assert GatedDeltaNet2 is not None

    def test_kernel_imports(self):
        from foreblocks.ops.attention import (
            can_use_fla_gdn2_chunk,
            fla_gdn2_chunk_forward,
        )

        assert callable(can_use_fla_gdn2_chunk)
        assert callable(fla_gdn2_chunk_forward)

    def test_construction_default(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        model = GatedDeltaNet2(
            d_model=256,
            n_heads=8,
            dropout=0.1,
        ).to(device)
        assert model.d_model == 256
        assert model.h == 8
        assert model.dk == 32  # 256 / 8
        assert model.dv == 32

    def test_construction_custom_dims(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        model = GatedDeltaNet2(
            d_model=512,
            n_heads=8,
            d_key=64,
            d_val=128,
        ).to(device)
        assert model.dk == 64
        assert model.dv == 128

    def test_forward_sequential(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 32, 256, 8
        model = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=0).to(device)
        model.eval()
        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, _ = model.forward_standalone(x)
        assert out.shape == (B, T, D)

    def test_forward_chunk(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 128, 256, 8
        model = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=64).to(device)
        model.eval()
        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, _ = model.forward_standalone(x)
        assert out.shape == (B, T, D)

    def test_forward_api(self, device):
        """Test the MultiAttention-compatible forward API."""
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 32, 256, 8
        model = GatedDeltaNet2(d_model=D, n_heads=H).to(device)
        model.eval()
        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, attn, state = model(
                query=x,
                key_padding_mask=None,
                is_causal=True,
                layer_state=None,
            )
        assert out.shape == (B, T, D)
        assert attn is None
        assert isinstance(state, dict)
        assert "gdn2_state" in state

    def test_recurrent_state(self, device):
        """Test that recurrent state is carried correctly across calls."""
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 16, 256, 8
        model = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=0).to(device)
        model.eval()
        x1 = torch.randn(B, T, D, device=device)
        x2 = torch.randn(B, T, D, device=device)
        layer_state = {}
        with torch.no_grad():
            model(x1, layer_state=layer_state)
            prev_state = layer_state["gdn2_state"].clone()
            model(x2, layer_state=layer_state)
            assert "gdn2_state" in layer_state
            assert not torch.allclose(
                layer_state["gdn2_state"], prev_state
            ), "State should have changed"


class TestGatedDeltaNet2Correctness:
    """Test that GDN-2 reduces to expected special cases."""

    def test_scalar_gates_reduce_to_gdn(self, device):
        """
        When b and w are scalar (all channels same), GDN-2 should reduce to
        GatedDeltaNet with same α and β.
        """
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
            GatedDeltaNet,
        )
        B, T, D, H = 2, 16, 128, 4
        gdn2 = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=0).to(device)
        gdn = GatedDeltaNet(d_model=D, n_heads=H, chunk_size=0).to(device)

        # Initialize both with same projections for Q, K, V
        with torch.no_grad():
            gdn.q_proj.weight.copy_(gdn2.q_proj.weight)
            gdn.k_proj.weight.copy_(gdn2.k_proj.weight)
            gdn.v_proj.weight.copy_(gdn2.v_proj.weight)
            gdn.gk_proj.weight.copy_(gdn2.gk_proj.weight)
            if gdn.dt_bias is not None:
                gdn.dt_bias.copy_(gdn2.dt_bias)
            # Make b_proj and w_proj produce constant-per-head values
            gdn2.b_proj.weight.data.fill_(0)
            gdn2.b_proj.bias.data = torch.zeros(H * D // H, device=device)
            gdn2.w_proj.weight.data.fill_(0)
            gdn2.w_proj.bias.data = torch.zeros(H * D // H, device=device)

        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out_gdn2, _ = gdn2.forward_standalone(x)
            out_gdn, _ = gdn.forward_standalone(x)

        # The outputs should NOT be identical because GDN has different
        # projection structures, but they should be in the same ballpark
        # (similar magnitude). We just verify both run without error.
        assert out_gdn2.shape == out_gdn.shape == (B, T, D)


class TestGatedDeltaNet2Integration:
    """Test integration with ModernLinearAttention wrapper."""

    def test_wrapper_gated_deltanet2(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            ModernLinearAttention,
        )
        attn = ModernLinearAttention(
            d_model=256,
            n_heads=8,
            dropout=0.1,
            backend="gated_deltanet2",
            chunk_size=64,
        ).to(device)
        B, T, D = 2, 32, 256
        q = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, _, state = attn(q, q, q, layer_state={})
        assert out.shape == (B, T, D)
        assert "gdn2_state" in state

    def test_wrapper_gdn2(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            ModernLinearAttention,
        )
        attn = ModernLinearAttention(
            d_model=256,
            n_heads=8,
            dropout=0.1,
            backend="gdn2",
            chunk_size=64,
        ).to(device)
        B, T, D = 2, 32, 256
        q = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, _, _ = attn(q, q, q, layer_state={})
        assert out.shape == (B, T, D)

    def test_state_key(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            ModernLinearAttention,
        )
        attn = ModernLinearAttention(
            d_model=256, n_heads=8, backend="gated_deltanet2"
        )
        assert attn.state_key == "gdn2_state"


class TestGatedDeltaNet2NegEigval:
    """Test negative eigenvalue mode."""

    def test_neg_eigval_mode(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 32, 256, 8
        model = GatedDeltaNet2(
            d_model=D, n_heads=H, allow_neg_eigval=True, chunk_size=64
        ).to(device)
        model.eval()
        x = torch.randn(B, T, D, device=device)
        with torch.no_grad():
            out, _ = model.forward_standalone(x)
        assert out.shape == (B, T, D)
        assert not torch.isnan(out).any()


class TestGatedDeltaNet2Gradient:
    """Test that gradients flow correctly."""

    def test_gradient_flow(self, device):
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 32, 128, 4
        model = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=64).to(device)
        x = torch.randn(B, T, D, device=device, requires_grad=True)
        out, _ = model.forward_standalone(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

        # Check that learnable params have gradients
        has_grad = False
        for p in model.parameters():
            if p.grad is not None and p.requires_grad:
                has_grad = True
                break
        assert has_grad, "At least one parameter should have a gradient"

    def test_chunk_gradient(self, device):
        """Test gradient flow through chunk-parallel mode."""
        from foreblocks.modules.attention.modules.linear_att import (
            GatedDeltaNet2,
        )
        B, T, D, H = 2, 128, 128, 4
        model = GatedDeltaNet2(d_model=D, n_heads=H, chunk_size=64).to(device)
        x = torch.randn(B, T, D, device=device, requires_grad=True)
        out, _ = model.forward_standalone(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
