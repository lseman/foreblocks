import torch
import torch.nn as nn
import pytest

from foreblocks.custom_mamba import (
    CHUNKED_SSD_TRITON_AVAILABLE,
    FeedForward,
    HybridMamba2Block,
    Mamba2Block,
    RMS_NORM_TRITON_AVAILABLE,
    RMSNorm,
    ROTARY_TRITON_AVAILABLE,
    RotaryEmbedding,
    SlidingWindowAttention,
    chunked_ssd_forward,
    chunked_ssd_forward_reference,
    dt_prep_bwd_triton,
    dt_prep_fallback,
    fused_out_bwd_triton,
    fused_out_fallback,
    rms_norm,
    rms_norm_fallback,
    rotary_apply,
    rotary_apply_fallback,
)


def _clone_requires_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_(True)


def _assert_close(lhs: torch.Tensor, rhs: torch.Tensor, atol: float = 1e-5) -> None:
    assert torch.allclose(lhs, rhs, atol=atol, rtol=atol)


def _sequential_diagonal_ssd(
    u: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    Bpar: torch.Tensor,
    Cpar: torch.Tensor,
    Dskip: torch.Tensor,
) -> torch.Tensor:
    batch, seqlen, heads, head_dim = u.shape
    d_state = Bpar.shape[-1]
    state = torch.zeros(batch, heads, head_dim, d_state, dtype=torch.float32)
    ys = []
    for t in range(seqlen):
        u_t = u[:, t].float()
        dt_t = dt[:, t].float()
        B_t = Bpar[:, t].float()
        C_t = Cpar[:, t].float()
        decay = torch.exp(
            dt_t.unsqueeze(-1).unsqueeze(-1)
            * A.float().unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        )
        state = decay * state + dt_t.unsqueeze(-1).unsqueeze(-1) * B_t.unsqueeze(
            -2
        ) * u_t.unsqueeze(-1)
        y_t = (C_t.unsqueeze(-2) * state).sum(dim=-1) + Dskip.float() * u_t
        ys.append(y_t.to(u.dtype))
    return torch.stack(ys, dim=1)


def test_mamba2_block_cpu_forward_uses_low_rank_dt() -> None:
    block = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=None,
        num_heads=4,
    )
    x = torch.randn(2, 11, 32)
    y = block(x)

    assert block.dt_proj.weight.shape == (4, block.dt_rank)
    assert isinstance(block.residual_proj, nn.Linear)
    assert y.shape == x.shape


def test_mamba2_block_fla_style_knobs_and_attention_mask() -> None:
    block = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        chunk_size=8,
        dt_init_min=0.001,
        dt_init_max=0.1,
        dt_limit=(0.0, 1.0),
        conv_init=0.01,
        use_conv_bias=False,
        use_bias=True,
        norm_eps=1e-5,
    )
    assert block.in_proj.bias is not None
    assert block.out_proj.bias is not None
    assert block.conv.conv.bias is None
    assert block.norm_eps == 1e-5
    assert block.dt_limit == (0.0, 1.0)
    assert block.conv.conv.weight.abs().max() <= 0.010001

    x = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9)
    mask[:, -2:] = 0
    y = block(x, attention_mask=mask)
    assert y.shape == x.shape


def test_mamba2_combined_path_matches_decomposed_path() -> None:
    torch.manual_seed(22)
    fused = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        chunk_size=8,
        use_fused_path=True,
        use_triton_ssd=False,
    )
    decomposed = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        chunk_size=8,
        use_fused_path=False,
        use_triton_ssd=False,
    )
    decomposed.load_state_dict(fused.state_dict())
    x = torch.randn(2, 9, 32)
    mask = torch.ones(2, 9)
    mask[:, -1] = 0

    y_fused = fused(x, attention_mask=mask)
    y_decomposed = decomposed(x, attention_mask=mask)
    _assert_close(y_fused, y_decomposed, atol=1e-6)


def test_mamba2_block_pre_norm_disabled() -> None:
    block = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        use_pre_norm=False,
    )
    assert isinstance(block.pre_norm, nn.Identity)
    x = torch.randn(2, 8, 32)
    assert block(x).shape == x.shape


def test_mamba2_block_step_matches_parallel() -> None:
    torch.manual_seed(0)
    block = Mamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        use_pre_norm=False,
    )
    block.eval()
    x = torch.randn(1, 6, 32)

    with torch.no_grad():
        y_par = block(x)
        state = block.make_state(1)
        y_step = torch.stack([block.step(x[:, t], state) for t in range(x.size(1))], dim=1)

    _assert_close(y_par, y_step, atol=1e-4)


def test_hybrid_mamba2_block_cpu_forward() -> None:
    block = HybridMamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        n_groups=2,
        window_size=8,
    )
    x = torch.randn(2, 10, 32)
    y = block(x)

    assert y.shape == x.shape
    assert isinstance(block.ssm, Mamba2Block)
    assert isinstance(block.out_norm, nn.LayerNorm)


def test_hybrid_mamba2_block_step_matches_parallel_with_attention_sink() -> None:
    torch.manual_seed(2)
    block = HybridMamba2Block(
        d_model=32,
        d_inner=64,
        d_state=8,
        d_conv=4,
        dt_rank=4,
        num_heads=4,
        window_size=4,
        n_sink_tokens=2,
    )
    block.eval()
    x = torch.randn(1, 7, 32)

    with torch.no_grad():
        y_parallel = block(x)
        state = block.make_state(1)
        y_step = torch.stack(
            [block.step(x[:, t], state) for t in range(x.shape[1])],
            dim=1,
        )

    _assert_close(y_parallel, y_step, atol=1e-4)


def test_chunked_ssd_forward_matches_sequential_diagonal_scan() -> None:
    torch.manual_seed(10)
    batch, seqlen, heads, head_dim, d_state = 2, 9, 3, 4, 5
    u = torch.randn(batch, seqlen, heads, head_dim)
    dt = torch.rand(batch, seqlen, heads) * 0.1 + 1e-3
    A = -torch.exp(torch.randn(heads))
    Bpar = torch.randn(batch, seqlen, heads, d_state) * 0.1
    Cpar = torch.randn(batch, seqlen, heads, d_state) * 0.1
    Dskip = torch.randn(heads, head_dim)

    y_ref = _sequential_diagonal_ssd(u, dt, A, Bpar, Cpar, Dskip)
    y_chunk = chunked_ssd_forward(u, dt, A, Bpar, Cpar, Dskip, chunk_size=4)
    y_simple = chunked_ssd_forward_reference(u, dt, A, Bpar, Cpar, Dskip, chunk_size=4)

    _assert_close(y_chunk, y_ref, atol=2e-5)
    _assert_close(y_simple, y_ref, atol=1e-6)


def test_dt_prep_backward_matches_autograd_reference() -> None:
    torch.manual_seed(11)
    batch, seqlen, dim = 2, 5, 7
    dt_raw = torch.randn(batch, seqlen, dim)
    bias = torch.randn(dim)
    grad_out = torch.randn_like(dt_raw)

    d_dt_raw, d_bias = dt_prep_bwd_triton(grad_out, dt_raw, bias)

    dt_raw_ref = _clone_requires_grad(dt_raw)
    bias_ref = _clone_requires_grad(bias)
    out_ref = dt_prep_fallback(dt_raw_ref, bias_ref)
    out_ref.backward(grad_out)

    _assert_close(d_dt_raw, dt_raw_ref.grad)
    assert d_bias is not None
    _assert_close(d_bias, bias_ref.grad)


def test_fused_out_backward_matches_autograd_reference() -> None:
    torch.manual_seed(12)
    batch, seqlen, dim = 2, 4, 9
    y = torch.randn(batch, seqlen, dim)
    z = torch.randn(batch, seqlen, dim)
    residual = torch.randn(batch, seqlen, dim)
    norm_weight = torch.randn(dim)
    grad_out = torch.randn_like(y)

    d_y, d_z, d_residual, d_norm_weight = fused_out_bwd_triton(
        grad_out, y, z, residual, norm_weight
    )

    y_ref = _clone_requires_grad(y)
    z_ref = _clone_requires_grad(z)
    residual_ref = _clone_requires_grad(residual)
    norm_weight_ref = _clone_requires_grad(norm_weight)
    out_ref = fused_out_fallback(y_ref, z_ref, residual_ref, norm_weight_ref)
    out_ref.backward(grad_out)

    _assert_close(d_y, y_ref.grad)
    _assert_close(d_z, z_ref.grad)
    _assert_close(d_residual, residual_ref.grad)
    assert d_norm_weight is not None
    _assert_close(d_norm_weight, norm_weight_ref.grad)


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


def test_sliding_window_attention_gqa_shapes() -> None:
    attn = SlidingWindowAttention(
        d_model=32,
        num_heads=8,
        n_kv_heads=2,
        window_size=4,
        dropout=0.0,
    )
    assert attn.n_rep == 4
    assert attn.k_proj.out_features == 2 * (32 // 8)
    x = torch.randn(2, 6, 32)
    assert attn(x).shape == x.shape


def test_sliding_window_attention_qk_norm_and_softcap() -> None:
    attn = SlidingWindowAttention(
        d_model=32,
        num_heads=8,
        n_kv_heads=2,
        window_size=4,
        dropout=0.0,
        qk_norm=True,
        logit_softcap=20.0,
    )
    assert isinstance(attn.q_norm, RMSNorm)
    assert isinstance(attn.k_norm, RMSNorm)
    x = torch.randn(2, 6, 32)
    assert attn(x).shape == x.shape


def test_rotary_embedding_shape_and_causality() -> None:
    rope = RotaryEmbedding(head_dim=16, base=10_000, max_seq_len=64)
    q = torch.randn(2, 4, 10, 16)
    k = torch.randn(2, 4, 10, 16)
    q_rot, k_rot = rope(q, k)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape
    assert not torch.allclose(q_rot, q)
    assert not torch.allclose(k_rot, k)


def test_rotary_apply_matches_fallback_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(20)
    torch.cuda.empty_cache()
    try:
        q = torch.randn(1, 2, 5, 8, device="cuda", requires_grad=True)
        cos = torch.randn(1, 1, 5, 8, device="cuda")
        sin = torch.randn(1, 1, 5, 8, device="cuda")
        grad = torch.randn_like(q)
    except torch.AcceleratorError as exc:
        pytest.skip(f"CUDA allocation unavailable: {exc}")

    y_fast = rotary_apply(q, cos, sin)
    y_ref = rotary_apply_fallback(q, cos, sin)
    _assert_close(y_fast, y_ref, atol=1e-6)

    y_fast.backward(grad, retain_graph=True)
    dq_fast = q.grad.detach().clone()
    q.grad = None
    y_ref.backward(grad)
    _assert_close(dq_fast, q.grad, atol=1e-6)


def test_rms_norm_matches_fallback_on_cuda() -> None:
    if not torch.cuda.is_available():
        return
    torch.manual_seed(21)
    torch.cuda.empty_cache()
    try:
        x = torch.randn(1, 2, 5, 8, device="cuda", requires_grad=True)
        weight = torch.randn(8, device="cuda", requires_grad=True)
        grad = torch.randn_like(x)
    except torch.AcceleratorError as exc:
        pytest.skip(f"CUDA allocation unavailable: {exc}")

    y_fast = rms_norm(x, weight)
    y_ref = rms_norm_fallback(x, weight)
    _assert_close(y_fast, y_ref, atol=1e-6)

    y_fast.backward(grad, retain_graph=True)
    dx_fast = x.grad.detach().clone()
    dw_fast = weight.grad.detach().clone()
    x.grad = None
    weight.grad = None
    y_ref.backward(grad)
    _assert_close(dx_fast, x.grad, atol=1e-6)
    _assert_close(dw_fast, weight.grad, atol=1e-6)


def test_feedforward_block_forward() -> None:
    ff = FeedForward(d_model=32, expansion=8 / 3, dropout=0.0)
    x = torch.randn(2, 10, 32)
    assert ff(x).shape == x.shape


def test_chunked_ssd_flag_is_boolean() -> None:
    assert isinstance(CHUNKED_SSD_TRITON_AVAILABLE, bool)
    assert isinstance(ROTARY_TRITON_AVAILABLE, bool)
    assert isinstance(RMS_NORM_TRITON_AVAILABLE, bool)
