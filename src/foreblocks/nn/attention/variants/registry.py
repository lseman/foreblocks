"""Registry of attention algorithm variants."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from importlib import import_module

from foreblocks.nn.attention.variants.base import AttentionImpl, AttentionOwner

VariantFactory = Callable[[AttentionOwner], AttentionImpl]


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

    def create(self, name: str, context: AttentionOwner) -> AttentionImpl:
        try:
            factory = self._factories[name.lower()]
        except KeyError as exc:
            raise ValueError(f"unknown attention type: {name}") from exc
        return factory(context)

    def names(self) -> tuple[str, ...]:
        return tuple(self._factories)


ATTENTION_VARIANTS = AttentionVariantRegistry()


def register_attention_variant(*names: str):
    """Register a custom attention variant factory under one or more names."""
    return ATTENTION_VARIANTS.register(*names)


def _build_builtin(
    module: str, class_name: str, context: AttentionOwner
) -> AttentionImpl:
    implementation = getattr(
        import_module(f"foreblocks.nn.attention.variants.{module}"), class_name
    )
    return implementation(context)


def _register_builtin_variants() -> None:
    # Register factories, not imports: unused variants stay unloaded.
    builtins = (
        (("standard", "sype"), "standard", "StandardAttentionImpl"),
        (("prob_sparse",), "prob_sparse", "ProbSparseAttentionImpl"),
        (("nsa",), "nsa", "NSAAttentionImpl"),
        (("moba",), "moba", "MoBAAttentionImpl"),
        (("sliding_window",), "sliding_window", "SlidingWindowAttentionImpl"),
        (
            ("dilated_sliding_window", "dilated_window"),
            "dilated_sliding_window",
            "DilatedSlidingWindowAttentionImpl",
        ),
        (("softpick",), "softpick", "SoftpickAttentionImpl"),
        (("frequency", "dwt", "autocor"), "spectral", "SpectralAttentionImpl"),
    )
    for names, module, class_name in builtins:
        ATTENTION_VARIANTS.register(*names)(partial(_build_builtin, module, class_name))


_register_builtin_variants()

__all__ = [
    "ATTENTION_VARIANTS",
    "AttentionVariantRegistry",
    "VariantFactory",
    "register_attention_variant",
]
