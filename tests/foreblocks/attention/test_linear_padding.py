"""A padded causal step is absent from memory, convolution, and position state."""

from copy import deepcopy

import pytest
import torch

from foreblocks.nn.attention.algorithms.linear import ModernLinearAttention

BACKENDS = ("rda", "gla", "deltanet", "gated_delta", "gated_deltanet2", "kimi")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def call(module, x, state, padding=None):
    y, _, returned = module(
        x, x, x, is_causal=True, key_padding_mask=padding, layer_state=state
    )
    return y, returned if returned is not None else state


def assert_state_row(actual, expected, b, batch, *, exact=False):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        reference = expected[key]
        if isinstance(value, dict):
            assert_state_row(value, reference, b, batch, exact=exact)
        elif isinstance(value, torch.Tensor) and value.ndim:
            torch.testing.assert_close(
                value[b : b + 1],
                torch.as_tensor(reference, device=value.device).reshape(
                    (1, *value.shape[1:])
                ),
                atol=0 if exact else 3e-5,
                rtol=0 if exact else 3e-5,
            )
        else:
            torch.testing.assert_close(
                torch.as_tensor(value), torch.as_tensor(reference), atol=0, rtol=0
            )


def assert_cache_unchanged(actual, expected, batch):
    assert actual.keys() == expected.keys()
    for key, reference in expected.items():
        value = actual[key]
        if isinstance(reference, dict):
            assert_cache_unchanged(value, reference, batch)
        else:
            device = value.device if isinstance(value, torch.Tensor) else None
            value = torch.as_tensor(value, device=device)
            reference = torch.as_tensor(reference, device=device)
            if value.shape != reference.shape:
                reference = reference.expand(batch)
            torch.testing.assert_close(value, reference, atol=0, rtol=0)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("mode", ["chunk", "recurrent"])
@pytest.mark.parametrize("device", DEVICES)
def test_padding_matches_removed_tokens_and_streamed_execution(backend, mode, device):
    torch.manual_seed(83)
    module = (
        ModernLinearAttention(
            32, 4, backend=backend, mode=mode, chunk_size=4, dropout=0
        )
        .to(device)
        .eval()
    )
    x = torch.randn(2, 11, 32, device=device)
    padding = torch.tensor(
        [[1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1], [0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0]],
        device=device,
        dtype=torch.bool,
    )
    dirty = x.masked_fill(padding[..., None], float("nan"))
    with torch.no_grad():
        full, state = call(module, dirty, {}, padding)
        outputs, streamed_state = [], {}
        for start, end in [(0, 3), (3, 4), (4, 8), (8, 11)]:
            output, streamed_state = call(
                module, dirty[:, start:end], streamed_state, padding[:, start:end]
            )
            outputs.append(output)
        torch.testing.assert_close(torch.cat(outputs, 1), full, atol=3e-5, rtol=3e-5)
        assert torch.isfinite(full).all()
        assert torch.count_nonzero(full[padding]) == 0
        for b in range(2):
            compact = x[b : b + 1, ~padding[b]]
            expected, expected_state = call(module, compact, {})
            torch.testing.assert_close(
                full[b, ~padding[b]], expected[0], atol=3e-5, rtol=3e-5
            )
            assert_state_row(state, expected_state, b, 2)
            assert_state_row(streamed_state, expected_state, b, 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_fully_padded_chunks_preserve_warm_state_and_future_outputs(backend, device):
    torch.manual_seed(89)
    module = (
        ModernLinearAttention(32, 4, backend=backend, dropout=0, chunk_size=4)
        .to(device)
        .eval()
    )
    prefill = torch.randn(2, 5, 32, device=device)
    with torch.no_grad():
        _, state = call(module, prefill, {})
        before = deepcopy(state)
        for length in (1, 7):
            padded = torch.full((2, length, 32), float("nan"), device=device)
            output, state = call(
                module,
                padded,
                state,
                torch.ones(2, length, device=device, dtype=torch.bool),
            )
            assert torch.count_nonzero(output) == 0
            assert_cache_unchanged(state, before, 2)
        next_token = torch.randn(2, 1, 32, device=device)
        expected, _ = call(module, next_token, before)
        actual, _ = call(module, next_token, state)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["rda", "gla"])
@pytest.mark.parametrize("device", DEVICES)
def test_rope_padding_uses_per_sequence_valid_positions(backend, device):
    torch.manual_seed(97)
    module = (
        ModernLinearAttention(
            32, 4, backend=backend, dropout=0, pos_encoding_type="rope"
        )
        .to(device)
        .eval()
    )
    x = torch.randn(2, 8, 32, device=device)
    padding = torch.tensor(
        [[1, 0, 0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 0, 0, 0]],
        device=device,
        dtype=torch.bool,
    )
    with torch.no_grad():
        full, state = call(module, x, {}, padding)
        streamed, stream_state = [], {}
        for start, end in [(0, 3), (3, 5), (5, 8)]:
            output, stream_state = call(
                module, x[:, start:end], stream_state, padding[:, start:end]
            )
            streamed.append(output)
        torch.testing.assert_close(torch.cat(streamed, 1), full, atol=3e-5, rtol=3e-5)
        for b in range(2):
            compact = x[b : b + 1, ~padding[b]]
            expected, _ = call(module, compact, {})
            torch.testing.assert_close(
                full[b, ~padding[b]], expected[0], atol=3e-5, rtol=3e-5
            )
        # Resume without a mask after sequences accumulated different offsets.
        next_x = torch.randn(2, 2, 32, device=device)
        actual, _ = call(module, next_x, stream_state)
        expected, _ = call(module, next_x, state)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_has_zero_input_gradient_and_valid_gradients_match_compaction(backend):
    torch.manual_seed(101)
    module = ModernLinearAttention(
        32, 4, backend=backend, dropout=0, chunk_size=4
    ).eval()
    x = torch.randn(2, 7, 32, requires_grad=True)
    padding = torch.tensor(
        [[1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool
    )
    dirty = x.masked_fill(padding[..., None], float("nan"))
    actual, _ = call(module, dirty, {}, padding)
    actual.square().sum().backward()
    parameter_gradients = {
        name: p.grad.clone()
        for name, p in module.named_parameters()
        if p.grad is not None
    }
    assert all(torch.isfinite(g).all() for g in parameter_gradients.values())
    module.zero_grad(set_to_none=True)
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[padding]) == 0
    for b in range(2):
        compact = x[b : b + 1, ~padding[b]].detach().clone().requires_grad_()
        expected, _ = call(module, compact, {})
        expected.square().sum().backward()
        torch.testing.assert_close(
            x.grad[b, ~padding[b]], compact.grad[0], atol=3e-5, rtol=3e-5
        )
    for name, p in module.named_parameters():
        if name in parameter_gradients:
            torch.testing.assert_close(
                p.grad, parameter_gradients[name], atol=1e-4, rtol=1e-4
            )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("device", DEVICES)
def test_cold_fully_padded_sequence_and_valid_sequence_share_batch(backend, device):
    torch.manual_seed(103)
    module = (
        ModernLinearAttention(32, 4, backend=backend, dropout=0, chunk_size=4)
        .to(device)
        .eval()
    )
    x = torch.randn(2, 7, 32, device=device)
    padding = torch.zeros(2, 7, device=device, dtype=torch.bool)
    padding[0] = True
    x[0] = float("nan")
    with torch.no_grad():
        actual, state = call(module, x, {}, padding)
        expected, expected_state = call(module, x[1:], {})
    assert torch.isfinite(actual).all()
    assert torch.count_nonzero(actual[0]) == 0
    torch.testing.assert_close(actual[1:], expected, atol=3e-5, rtol=3e-5)
    assert_state_row(state, expected_state, 1, 2)


@pytest.mark.parametrize("backend", ["gated_delta", "gated_deltanet2", "kimi"])
def test_padding_without_short_convolution(backend):
    torch.manual_seed(107)
    module = ModernLinearAttention(
        32, 4, backend=backend, dropout=0, use_short_conv=False
    ).eval()
    x = torch.randn(1, 7, 32)
    padding = torch.tensor([[1, 0, 1, 0, 0, 1, 0]], dtype=torch.bool)
    with torch.no_grad():
        actual, state = call(module, x, {}, padding)
        expected, ref = call(module, x[:, ~padding[0]], {})
    torch.testing.assert_close(actual[:, ~padding[0]], expected, atol=3e-5, rtol=3e-5)
    assert_state_row(state, ref, 0, 1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "bad_mask", [torch.zeros(2, 7), torch.zeros(1, 7, dtype=torch.bool)]
)
def test_padding_requires_boolean_batch_aligned_mask(backend, bad_mask):
    module = ModernLinearAttention(32, 4, backend=backend, dropout=0)
    x = torch.randn(2, 7, 32)
    with pytest.raises(ValueError, match="key_padding_mask"):
        call(module, x, {}, bad_mask)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_precision_padding_preserves_warm_cache(backend, dtype):
    torch.manual_seed(109)
    module = (
        ModernLinearAttention(64, 4, backend=backend, dropout=0, chunk_size=4)
        .cuda()
        .to(dtype)
        .eval()
    )
    with torch.no_grad():
        _, state = call(module, torch.randn(2, 5, 64, device="cuda", dtype=dtype), {})
        before = deepcopy(state)
        output, state = call(
            module,
            torch.full((2, 7, 64), float("nan"), device="cuda", dtype=dtype),
            state,
            torch.ones(2, 7, device="cuda", dtype=torch.bool),
        )
    assert torch.count_nonzero(output) == 0
    assert_cache_unchanged(state, before, 2)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mixed_precision_padding_has_finite_gradients(backend, dtype):
    torch.manual_seed(111)
    module = (
        ModernLinearAttention(32, 4, backend=backend, dropout=0, chunk_size=4)
        .cuda()
        .to(dtype)
        .eval()
    )
    x = torch.randn(2, 7, 32, device="cuda", dtype=dtype, requires_grad=True)
    padding = torch.tensor(
        [[1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0]], device="cuda", dtype=torch.bool
    )
    actual, _ = call(
        module, x.masked_fill(padding[..., None], float("nan")), {}, padding
    )
    actual.float().square().mean().backward()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[padding]) == 0
    for p in module.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize("backend", ["rda", "gated_delta", "gated_deltanet2"])
def test_cross_attention_padding_only_masks_memory(backend):
    torch.manual_seed(113)
    module = ModernLinearAttention(32, 4, backend=backend, dropout=0).eval()
    query = torch.randn(2, 3, 32)
    key, value = [torch.randn(2, 7, 32, requires_grad=True) for _ in range(2)]
    padding = torch.tensor(
        [[1, 0, 0, 1, 0, 1, 0], [0, 1, 0, 0, 1, 0, 0]], dtype=torch.bool
    )
    dirty_key, dirty_value = (
        t.masked_fill(padding[..., None], float("nan")) for t in (key, value)
    )
    actual = module(query, dirty_key, dirty_value, key_padding_mask=padding)[0]
    actual.square().sum().backward()
    assert torch.isfinite(actual).all()
    for t in (key, value):
        assert torch.isfinite(t.grad).all()
        assert torch.count_nonzero(t.grad[padding]) == 0
    with torch.no_grad():
        for b in range(2):
            expected = module(
                query[b : b + 1],
                key[b : b + 1, ~padding[b]],
                value[b : b + 1, ~padding[b]],
            )[0]
            torch.testing.assert_close(
                actual[b : b + 1], expected, atol=3e-5, rtol=3e-5
            )


@pytest.mark.parametrize("backend", BACKENDS)
def test_padding_between_streamed_chunks_preserves_backpropagation(backend):
    torch.manual_seed(127)
    module = ModernLinearAttention(
        32, 4, backend=backend, mode="recurrent", dropout=0
    ).eval()
    x = torch.randn(1, 7, 32, requires_grad=True)
    ref = x.detach().clone().requires_grad_()
    padding = torch.tensor([[0, 1, 0, 0, 1, 1, 0]], dtype=torch.bool)
    full, _ = call(module, ref, {}, padding)
    outputs, state = [], {}
    for start, end in [(0, 1), (1, 2), (2, 4), (4, 6), (6, 7)]:
        output, state = call(module, x[:, start:end], state, padding[:, start:end])
        outputs.append(output)
    torch.cat(outputs, 1)[:, -1].square().sum().backward()
    full[:, -1].square().sum().backward()
    torch.testing.assert_close(x.grad, ref.grad, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize("backend", ["gated_delta", "gated_deltanet2"])
def test_preprojected_kv_padding_preserves_supplied_state(backend):
    module = ModernLinearAttention(32, 4, backend=backend, dropout=0).impl.eval()
    state_key = "gdn_state" if backend == "gated_delta" else "gdn2_state"
    initial = torch.randn(2, module.h, module.dk, module.dv)
    state = {state_key: initial.clone()}
    padding = torch.ones(2, 3, dtype=torch.bool)
    x = torch.full((2, 3, 32), float("nan"))
    k = torch.full((2, 3, module.h * module.dk), float("nan"))
    v = torch.full((2, 3, module.h * module.dv), float("nan"))
    with torch.no_grad():
        actual = module(
            x,
            k,
            v,
            key_padding_mask=padding,
            layer_state=state,
            skip_cross_attention=True,
        )[0]
    assert torch.count_nonzero(actual) == 0
    torch.testing.assert_close(state[state_key], initial, atol=0, rtol=0)
