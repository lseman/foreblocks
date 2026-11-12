"""Regression tests for the stable attention package surface."""

from __future__ import annotations

from foreblocks.nn import attention
from foreblocks.nn.attention import (
    AttentionConfig,
    AttentionShapeConfig,
    MultiAttention,
)
from foreblocks.nn.attention.algorithms import GatedDeltaNetBackend


def test_public_api_is_explicit_and_resolvable() -> None:
    assert attention.__all__
    assert len(attention.__all__) == len(set(attention.__all__))
    assert all(hasattr(attention, name) for name in attention.__all__)
    assert "torch" not in attention.__all__
    assert "GatedDeltaNetBackend" not in attention.__all__
    assert "PagedKVCache" not in attention.__all__
    assert GatedDeltaNetBackend.__name__ == "GatedDeltaNetBackend"


def test_multi_attention_consumes_grouped_config() -> None:
    config = AttentionConfig(shape=AttentionShapeConfig(d_model=16, n_heads=4))
    configured = MultiAttention(config)

    assert configured.d_model == 16
    assert configured.n_heads == 4
    assert configured.impl.context is configured.context


def test_attention_namespaces_keep_unused_implementations_unloaded() -> None:
    import subprocess
    import sys

    code = """
import sys
from foreblocks.nn import attention
from foreblocks.nn.attention import AttentionConfig, AttentionShapeConfig
from foreblocks.nn.attention import algorithms, cache, variants
assert 'foreblocks.nn.attention.multihead' not in sys.modules
assert not any(name.startswith('foreblocks.nn.attention.algorithms.linear.') for name in sys.modules)
assert 'foreblocks.nn.attention.cache.paged' not in sys.modules
assert 'foreblocks.nn.attention.variants.moba' not in sys.modules
assert set(attention.__all__).issubset(dir(attention))
for namespace in (attention, algorithms, cache, variants):
    try:
        getattr(namespace, 'not_an_export')
    except AttributeError:
        pass
    else:
        raise AssertionError(namespace.__name__)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_standard_attention_does_not_load_other_algorithms() -> None:
    import subprocess
    import sys

    code = """
import sys
from foreblocks.nn.attention import AttentionConfig, AttentionShapeConfig, MultiAttention
module = MultiAttention(AttentionConfig(shape=AttentionShapeConfig(d_model=16, n_heads=4)))
assert not any(name.startswith('foreblocks.nn.attention.algorithms.') for name in sys.modules)
for name in ('moba', 'nsa', 'prob_sparse', 'sliding_window', 'softpick', 'spectral'):
    assert f'foreblocks.nn.attention.variants.{name}' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_linear_wrapper_loads_only_the_selected_backend() -> None:
    import subprocess
    import sys

    code = """
import sys
from foreblocks.nn.attention.algorithms import ModernLinearAttention
module = ModernLinearAttention(d_model=16, n_heads=4, backend='gla')
assert 'foreblocks.nn.attention.algorithms.linear.gla' in sys.modules
for name in ('rda', 'deltanet', 'gated_delta', 'gated_deltanet2', 'kimi'):
    assert f'foreblocks.nn.attention.algorithms.linear.{name}' not in sys.modules
assert 'foreblocks.nn.attention.algorithms.spectral.frequency' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_lazy_exports_resolve_to_the_concrete_classes() -> None:
    from importlib import import_module

    for module_name in ("algorithms", "algorithms.linear", "variants", "cache"):
        namespace = import_module(f"foreblocks.nn.attention.{module_name}")
        for name in namespace.__all__:
            exported = getattr(namespace, name)
            implementation = import_module(exported.__module__)
            assert exported is getattr(implementation, name)


def test_position_extension_preserves_parameter_paths_and_uniform_attention() -> None:
    import torch

    from foreblocks.nn.attention import (
        AttentionCacheConfig,
        AttentionPositionConfig,
        AttentionVariantConfig,
    )

    torch.manual_seed(31)
    module = MultiAttention(
        AttentionConfig(
            shape=AttentionShapeConfig(d_model=16, n_heads=4, dropout=0),
            cache=AttentionCacheConfig(use_mla=False),
            position=AttentionPositionConfig(encoding="none"),
            variant=AttentionVariantConfig(backend="eager", use_swiglu=False),
        )
    ).eval()
    keys = set(module.state_dict())
    assert keys == {
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "out_proj.weight",
        "out_proj.bias",
    }
    module.register_position_transform(
        "uniform", lambda q, k, context: (torch.zeros_like(q), k)
    )
    values = torch.randn(2, 8, 16, requires_grad=True)
    actual, _, _ = module(values)
    expected = module.out_proj(
        module.v_proj(values).mean(dim=1, keepdim=True)
    ).expand_as(actual)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert set(module.state_dict()) == keys
