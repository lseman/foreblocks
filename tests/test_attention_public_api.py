"""Regression tests for the stable attention package surface."""

from __future__ import annotations

import inspect

from foreblocks.modules import attention
from foreblocks.modules.attention import (
    AttentionConfig,
    AttentionShapeConfig,
    MultiAttention,
    MultiAttentionConfig,
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


def test_grouped_config_covers_legacy_constructor() -> None:
    constructor = inspect.signature(MultiAttention.__init__)
    expected = {name for name in constructor.parameters if name != "self"}
    config = MultiAttentionConfig(shape=AttentionShapeConfig(d_model=16, n_heads=4))

    assert MultiAttentionConfig is AttentionConfig
    assert set(config.to_legacy_kwargs()) == expected


def test_config_factory_preserves_legacy_constructor() -> None:
    config = MultiAttentionConfig(shape=AttentionShapeConfig(d_model=16, n_heads=4))
    configured = MultiAttention.from_config(
        config,
        use_mla=False,
        use_paged_cache=False,
        use_swiglu=False,
    )
    legacy = MultiAttention(
        16,
        4,
        use_mla=False,
        use_paged_cache=False,
        use_swiglu=False,
    )

    assert configured.d_model == legacy.d_model == 16
    assert configured.n_heads == legacy.n_heads == 4
    assert configured.impl.context is configured.context
