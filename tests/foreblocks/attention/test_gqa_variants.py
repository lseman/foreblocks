"""Grouped KV heads must agree with explicit head repetition."""

from copy import deepcopy

import pytest
import torch

from foreblocks.nn.attention import (
    AttentionCacheConfig,
    AttentionConfig,
    AttentionPositionConfig,
    AttentionShapeConfig,
    AttentionVariantConfig,
    MultiAttention,
)
from foreblocks.nn.embeddings.symplectic import SyPERotator


def make_attention(name, kv_heads, *, fast=False, chunk_size=64, paged=False):
    return (
        MultiAttention(
            AttentionConfig(
                shape=AttentionShapeConfig(
                    d_model=16, n_heads=4, n_kv_heads=kv_heads, dropout=0
                ),
                cache=AttentionCacheConfig(
                    use_mla=False, use_paged_cache=paged, block_size=4, max_blocks=32
                ),
                position=AttentionPositionConfig(encoding="none"),
                variant=AttentionVariantConfig(
                    name=name,
                    backend="eager",
                    use_swiglu=False,
                    use_flash_sliding=fast,
                    window_size=3,
                    chunk_size=chunk_size,
                ),
            )
        )
        .double()
        .eval()
    )


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_sype_rotates_grouped_keys_without_expanding_storage(kv_heads):
    torch.manual_seed(13)
    rotator = SyPERotator(4).double()
    with torch.no_grad():
        rotator.alpha.copy_(torch.tensor([-0.3, 0.2]))
        rotator.gamma.copy_(torch.tensor([0.1, -0.2]))
    repeated = deepcopy(rotator)
    q = torch.randn(2, 5, 4, 4, dtype=torch.float64).transpose(1, 2).requires_grad_()
    k = (
        torch.randn(2, 5, kv_heads, 4, dtype=torch.float64)
        .transpose(1, 2)
        .requires_grad_()
    )
    tau = torch.rand(2, 5, dtype=torch.float64, requires_grad=True)
    actual_q, actual_k = rotator.rotate_qk(q, k, tau)
    assert actual_k.shape == k.shape
    repeat = 4 // kv_heads
    expected_q, expected_k = repeated.rotate_qk(
        q, k.repeat_interleave(repeat, dim=1), tau
    )
    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k.repeat_interleave(repeat, dim=1), expected_k)
    actual_loss = actual_q.square().sum() + repeat * actual_k.square().sum()
    expected_loss = expected_q.square().sum() + expected_k.square().sum()
    actual_grads = torch.autograd.grad(actual_loss, (q, k, tau, *rotator.parameters()))
    expected_grads = torch.autograd.grad(
        expected_loss, (q, k, tau, *repeated.parameters())
    )
    for actual, expected in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("kv_heads", [1, 2])
def test_sype_full_forward_and_gradients_match_explicit_repeated_projections(kv_heads):
    torch.manual_seed(17)
    grouped = make_attention("sype", kv_heads)
    reference = make_attention("sype", 4)
    state = grouped.state_dict()
    for name in ("k_proj.weight", "v_proj.weight"):
        state[name] = (
            state[name]
            .reshape(kv_heads, 4, 16)
            .repeat_interleave(4 // kv_heads, dim=0)
            .reshape(16, 16)
        )
    reference.load_state_dict(state)
    value = torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True)
    actual = grouped(value, is_causal=True)[0]
    expected = reference(value, is_causal=True)[0]
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(
        actual.square().sum(), (value, *grouped.parameters())
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (value, *reference.parameters())
    )
    for (name, _), actual_grad, expected_grad in zip(
        [("input", value), *grouped.named_parameters()], actual_grads, expected_grads
    ):
        if name in ("k_proj.weight", "v_proj.weight"):
            expected_grad = (
                expected_grad.reshape(kv_heads, 4 // kv_heads, 4, 16)
                .sum(dim=1)
                .reshape_as(actual_grad)
            )
        torch.testing.assert_close(actual_grad, expected_grad)


def reference_window(
    q, k, v, *, causal, window=3, starts=None, blocked=None, padding=None
):
    positions = torch.arange(q.size(2), device=q.device).view(1, -1, 1)
    if starts is not None:
        positions = positions + starts.view(-1, 1, 1)
    keys = torch.arange(k.size(2), device=q.device).view(1, 1, -1)
    if causal:
        window_mask = (keys > positions) | (keys < positions - window + 1)
    else:
        window_mask = (keys < positions - window // 2) | (
            keys > positions + window // 2
        )
    window_mask = window_mask.unsqueeze(1)
    if blocked is not None:
        window_mask = window_mask | blocked
    if padding is not None:
        window_mask = window_mask | padding[:, None, None, :]
    scores = q @ k.transpose(-2, -1) / q.size(-1) ** 0.5
    scores = scores.masked_fill(window_mask, -torch.inf)
    empty = torch.isneginf(scores).all(dim=-1, keepdim=True)
    weights = scores.masked_fill(empty, 0).softmax(dim=-1).masked_fill(empty, 0)
    return weights @ v, weights


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("chunk_size", [2, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("mask_kind", ["none", "padding", "attention"])
def test_sliding_window_outputs_and_gradients_match_dense_reference(
    kv_heads, fast, chunk_size, causal, mask_kind, monkeypatch
):
    torch.manual_seed(21)
    module = make_attention(
        "sliding_window", kv_heads, fast=fast, chunk_size=chunk_size
    )
    if fast:

        def no_fallback(*args, **kwargs):
            raise AssertionError(
                "SDPA should handle this case without a manual fallback"
            )

        monkeypatch.setattr(module.impl, "manual", no_fallback)
    value = torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True)
    blocked = (
        torch.zeros(2, 4, 9, 9, dtype=torch.bool) if mask_kind == "attention" else None
    )
    padding = torch.zeros(2, 9, dtype=torch.bool) if mask_kind == "padding" else None
    if blocked is not None:
        blocked[0, 0, 0, :] = True  # Entirely masked query/head must remain finite.
        blocked[1, 2, :, 3] = True
    if padding is not None:
        padding[0, :2] = True
        padding[1, -2:] = True
    actual, weights, _ = module(
        value,
        is_causal=causal,
        attn_mask=blocked,
        key_padding_mask=padding,
        need_weights=not fast,
    )
    q = module.q_proj(value).reshape(2, 9, 4, 4).transpose(1, 2)
    k = (
        module.k_proj(value)
        .reshape(2, 9, kv_heads, 4)
        .transpose(1, 2)
        .repeat_interleave(4 // kv_heads, dim=1)
    )
    v = (
        module.v_proj(value)
        .reshape(2, 9, kv_heads, 4)
        .transpose(1, 2)
        .repeat_interleave(4 // kv_heads, dim=1)
    )
    expected, expected_weights = reference_window(
        q, k, v, causal=causal, blocked=blocked, padding=padding
    )
    expected = module.out_proj(expected.transpose(1, 2).reshape(2, 9, 16))
    torch.testing.assert_close(actual, expected)
    if not fast:
        torch.testing.assert_close(weights, expected_weights)
    actual_grads = torch.autograd.grad(
        actual.square().sum(), (value, *module.parameters())
    )
    expected_grads = torch.autograd.grad(
        expected.square().sum(), (value, *module.parameters())
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("chunk_size", [2, 64])
def test_sliding_window_respects_distinct_batch_cache_offsets(kv_heads, chunk_size):
    module = make_attention("sliding_window", kv_heads, chunk_size=chunk_size)
    q = torch.randn(2, 4, 3, 4, dtype=torch.float64)
    k = torch.randn(2, kv_heads, 10, 4, dtype=torch.float64)
    v = torch.randn_like(k)
    starts = torch.tensor([2, 6])
    actual, weights = module.impl.manual(
        q, k, v, None, None, True, True, q_start_pos=starts
    )
    expected, expected_weights = reference_window(
        q,
        k.repeat_interleave(4 // kv_heads, dim=1),
        v.repeat_interleave(4 // kv_heads, dim=1),
        causal=True,
        starts=starts,
    )
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(weights, expected_weights)


@pytest.mark.parametrize("name", ["sype", "sliding_window"])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("paged", [False, True])
def test_grouped_attention_chunked_cache_matches_full_forward(
    name, kv_heads, fast, paged
):
    torch.manual_seed(29)
    module = make_attention(name, kv_heads, fast=fast, paged=paged, chunk_size=2)
    value = torch.randn(2, 9, 16, dtype=torch.float64)
    with torch.no_grad():
        full = module(value, is_causal=True)[0]
        state = {}
        parts = [
            module(value[:, a:b], is_causal=True, layer_state=state)[0]
            for a, b in [(0, 3), (3, 5), (5, 6), (6, 9)]
        ]
    torch.testing.assert_close(torch.cat(parts, dim=1), full)
    if paged:
        assert state["paged_cache"].Hkv == kv_heads
    else:
        assert state["k"].size(1) == kv_heads
        assert state["v"].size(1) == kv_heads
