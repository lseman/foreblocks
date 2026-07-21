"""Registry of attention algorithm variants."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foreblocks.modules.attention.variants.base import AttentionImpl

VariantFactory = Callable[[Any], AttentionImpl]


class AttentionVariantRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, VariantFactory] = {}

    def register(self, *names: str):
        def decorator(factory: VariantFactory) -> VariantFactory:
            for name in names:
                key = name.lower()
                if key in self._factories:
                    raise ValueError(f"attention variant {key!r} is already registered")
                self._factories[key] = factory
            return factory

        return decorator

    def create(self, name: str, owner: Any) -> AttentionImpl:
        try:
            factory = self._factories[name.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown attention type: {name}") from exc
        return factory(owner)

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


ATTENTION_VARIANTS = AttentionVariantRegistry()


def _register_builtin_variants() -> None:
    from foreblocks.modules.attention.variants import (
        DilatedSlidingWindowAttentionImpl,
        MoBAAttentionImpl,
        NSAImpl,
        ProbSparseAttentionImpl,
        SlidingWindowAttentionImpl,
        SoftpickAttentionImpl,
        SpectralAttentionImpl,
        StandardAttentionImpl,
    )

    ATTENTION_VARIANTS.register("standard", "sype")(StandardAttentionImpl)
    ATTENTION_VARIANTS.register("prob_sparse")(ProbSparseAttentionImpl)
    ATTENTION_VARIANTS.register("nsa")(NSAImpl)
    ATTENTION_VARIANTS.register("moba")(MoBAAttentionImpl)
    ATTENTION_VARIANTS.register("sliding_window")(SlidingWindowAttentionImpl)
    ATTENTION_VARIANTS.register("dilated_sliding_window", "dilated_window")(
        DilatedSlidingWindowAttentionImpl
    )
    ATTENTION_VARIANTS.register("softpick")(SoftpickAttentionImpl)
    ATTENTION_VARIANTS.register("frequency", "dwt", "autocor")(SpectralAttentionImpl)


_register_builtin_variants()

__all__ = ["ATTENTION_VARIANTS", "AttentionVariantRegistry", "VariantFactory"]
