import math

import torch
import torch.nn as nn

from foreblocks.hybrid_mamba import (
    GROUPED_SSD_TRITON_AVAILABLE,
    grouped_ssd_scan,
    grouped_ssd_scan_reference,
)
from foreblocks.hybrid_mamba.layers import (
    HybridMamba2Block,
    HybridMambaBlock,
    RotaryEmbedding,
    SlidingWindowAttention,
    StructuredStateSpaceDualityBranch,
    TinyHybridMamba2LM,
    TinyHybridMambaLM,
)


def test_hybrid_mamba_block_cpu_forward_uses_low_rank_dt() -> None:
    block = HybridMambaBlock(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=None,
        use_cuda_scan=False,
    )
    x = torch.randn(2, 11, 32)
    y = block(x)

    assert block.dt_rank == max(4, math.ceil(32 / 16))
    assert block.dt_proj.weight.shape == (64, block.dt_rank)
    assert isinstance(block.residual_proj, nn.Linear)
    assert y.shape == x.shape


def test_tiny_hybrid_mamba_lm_ties_embeddings_by_default() -> None:
    model = TinyHybridMambaLM(
        vocab_size=101,
        d_model=32,
        n_layers=2,
        d_state=8,
        d_conv=4,
        tie_embeddings=True,
    )
    input_ids = torch.randint(0, 101, (2, 9))
    logits = model(input_ids)

    assert model.lm_head.weight is model.embed.weight
    assert logits.shape == (2, 9, 101)


def test_sliding_window_attention_is_causal() -> None:
    attn = SlidingWindowAttention(
        d_model=16,
        num_heads=4,
        window_size=3,
        dropout=0.0,
    )
    attn.eval()

    x = torch.randn(2, 6, 16)
    x_perturbed = x.clone()
    x_perturbed[:, -1, :] = torch.randn_like(x_perturbed[:, -1, :]) * 10.0

    y = attn(x)
    y_perturbed = attn(x_perturbed)

    assert torch.allclose(y[:, :-1, :], y_perturbed[:, :-1, :], atol=1e-6, rtol=1e-6)


def test_hybrid_mamba2_block_cpu_forward() -> None:
    block = HybridMamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        window_size=8,
        use_cuda_scan=False,
    )
    x = torch.randn(2, 10, 32)
    y = block(x)

    assert y.shape == x.shape
    assert isinstance(block.ssm, StructuredStateSpaceDualityBranch)


def test_tiny_hybrid_mamba2_lm_forward_and_tied_weights() -> None:
    model = TinyHybridMamba2LM(
        vocab_size=127,
        d_model=32,
        n_layers=3,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        window_size=8,
        attn_every_n=2,
        tie_embeddings=True,
        use_cuda_scan=False,
    )
    input_ids = torch.randint(0, 127, (2, 7))
    logits = model(input_ids)

    assert model.lm_head.weight is model.embed.weight
    assert logits.shape == (2, 7, 127)


def test_structured_ssd_branch_supports_gated_delta() -> None:
    branch = StructuredStateSpaceDualityBranch(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        use_gated_delta=True,
    )
    x = torch.randn(2, 12, 32)
    y = branch(x)

    assert branch.dt_proj.weight.shape == (4, 4)
    assert y.shape == x.shape


def test_tiny_hybrid_mamba2_lm_supports_gated_delta() -> None:
    model = TinyHybridMamba2LM(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        window_size=8,
        attn_every_n=2,
        tie_embeddings=True,
        use_gated_delta=True,
        use_cuda_scan=False,
    )
    input_ids = torch.randint(0, 64, (2, 5))
    logits = model(input_ids)

    assert logits.shape == (2, 5, 64)


def test_grouped_ssd_scan_matches_reference_on_cpu() -> None:
    u = torch.randn(2, 6, 4, 8)
    dt = torch.rand(2, 6, 4) * 0.1 + 1e-3
    A = -torch.exp(torch.randn(4, 8))
    Bpar = torch.randn(2, 6, 4, 8) * 0.1
    Cpar = torch.randn(2, 6, 4, 8) * 0.1
    Dskip = torch.randn(4, 8)
    gate = torch.randn(2, 6, 4)

    y_ref = grouped_ssd_scan_reference(u, dt, A, Bpar, Cpar, Dskip, delta_gate=gate)
    y = grouped_ssd_scan(u, dt, A, Bpar, Cpar, Dskip, delta_gate=gate, use_triton=True)

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-6)


def test_rotary_embedding_shape_and_causality() -> None:
    rope = RotaryEmbedding(head_dim=16, base=10_000, max_seq_len=64)
    B, H, T = 2, 4, 10
    q = torch.randn(B, H, T, 16)
    k = torch.randn(B, H, T, 16)
    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    # RoPE must change the values
    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)
    # Two identical tokens at different positions must get different encodings
    q_same = torch.ones(1, 1, 2, 16)
    q_enc, _ = rope(q_same, q_same)
    assert not torch.allclose(q_enc[:, :, 0], q_enc[:, :, 1])


def test_sliding_window_attention_gqa_shapes() -> None:
    # 8 query heads, 2 KV heads (n_rep = 4)
    attn = SlidingWindowAttention(
        d_model=32,
        num_heads=8,
        n_kv_heads=2,
        window_size=4,
        dropout=0.0,
    )
    assert attn.n_rep == 4
    assert attn.k_proj.out_features == 2 * (32 // 8)  # kv_dim = 2 * head_dim
    x = torch.randn(2, 6, 32)
    y = attn(x)
    assert y.shape == x.shape


def test_hybrid_mamba_block_pre_norm_disabled() -> None:
    block = HybridMambaBlock(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        use_cuda_scan=False,
        use_pre_norm=False,
    )
    assert isinstance(block.pre_norm, nn.Identity)
    x = torch.randn(2, 8, 32)
    y = block(x)
    assert y.shape == x.shape


def test_hybrid_mamba2_block_out_norm_present() -> None:
    block = HybridMamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        window_size=8,
        use_cuda_scan=False,
    )
    assert isinstance(block.out_norm, nn.LayerNorm)
    assert block.out_norm.normalized_shape == (32,)
    x = torch.randn(2, 10, 32)
    y = block(x)
    assert y.shape == x.shape


def test_tiny_hybrid_mamba2_lm_with_gqa() -> None:
    model = TinyHybridMamba2LM(
        vocab_size=64,
        d_model=32,
        n_layers=4,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        n_kv_heads=2,
        window_size=8,
        attn_every_n=2,
        tie_embeddings=True,
        use_cuda_scan=False,
    )
    input_ids = torch.randint(0, 64, (2, 6))
    logits = model(input_ids)
    assert logits.shape == (2, 6, 64)


def test_grouped_ssd_scan_triton_matches_reference_when_available() -> None:
    if not (torch.cuda.is_available() and GROUPED_SSD_TRITON_AVAILABLE):
        return

    device = "cuda"
    dtype = torch.float32
    u = torch.randn(2, 8, 4, 8, device=device, dtype=dtype)
    dt = torch.rand(2, 8, 4, device=device, dtype=dtype) * 0.1 + 1e-3
    A = -torch.exp(torch.randn(4, 8, device=device, dtype=dtype))
    Bpar = torch.randn(2, 8, 4, 8, device=device, dtype=dtype) * 0.1
    Cpar = torch.randn(2, 8, 4, 8, device=device, dtype=dtype) * 0.1
    Dskip = torch.randn(4, 8, device=device, dtype=dtype)
    gate = torch.randn(2, 8, 4, device=device, dtype=dtype)

    y_ref = grouped_ssd_scan_reference(u, dt, A, Bpar, Cpar, Dskip, delta_gate=gate)
    y = grouped_ssd_scan(u, dt, A, Bpar, Cpar, Dskip, delta_gate=gate, use_triton=True)

    assert torch.allclose(y, y_ref, atol=5e-4, rtol=5e-4)
