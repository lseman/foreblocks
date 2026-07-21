"""Regression tests for the stable attention package surface."""

from __future__ import annotations

from foreblocks.modules import attention
from foreblocks.modules.attention import (
    AttentionConfig,
    AttentionShapeConfig,
    MultiAttention,
)
from foreblocks.modules.attention.implementations import GatedDeltaNet


def test_public_api_is_explicit_and_resolvable() -> None:
    assert attention.__all__
    assert len(attention.__all__) == len(set(attention.__all__))
    assert all(hasattr(attention, name) for name in attention.__all__)
    assert "torch" not in attention.__all__
    assert "GatedDeltaNet" not in attention.__all__
    assert "PagedKVCache" not in attention.__all__
    assert GatedDeltaNet.__name__ == "GatedDeltaNet"


def test_multi_attention_consumes_grouped_config() -> None:
    config = AttentionConfig(shape=AttentionShapeConfig(d_model=16, n_heads=4))
    configured = MultiAttention(config)

    assert configured.d_model == 16
    assert configured.n_heads == 4
    assert configured.impl.context is configured.context
