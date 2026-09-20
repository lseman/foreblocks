"""Canonical public architecture components.

Exports are resolved lazily (mirroring ``darts/__init__.py``) so that a
submodule needing only a small, dependency-light piece of this package —
e.g. ``config.py`` deriving its search-space defaults from
``op_registry.py`` — does not have to pay for importing every architecture
submodule (``mixed_op``, ``time_series_darts``, ``darts_cell``, ...) just to
touch one of them.
"""

from importlib import import_module


__all__ = [
    "ArchitectureConverter",
    "DARTSCell",
    "FixedDecoder",
    "FixedEncoder",
    "MixedDecoder",
    "MixedEncoder",
    "MixedOp",
    "TimeSeriesDARTS",
    "derive_final_architecture",
]


def __getattr__(name):
    lazy_exports = {
        "ArchitectureConverter": (".search.converter", "ArchitectureConverter"),
        "DARTSCell": (".search.darts_cell", "DARTSCell"),
        "derive_final_architecture": (".search.finalization", "derive_final_architecture"),
        "FixedDecoder": (".search.fixed_encoder_decoder", "FixedDecoder"),
        "FixedEncoder": (".search.fixed_encoder_decoder", "FixedEncoder"),
        "MixedDecoder": (".search.mixed_encoder_decoder", "MixedDecoder"),
        "MixedEncoder": (".search.mixed_encoder_decoder", "MixedEncoder"),
        "MixedOp": (".search.mixed_op", "MixedOp"),
        "TimeSeriesDARTS": (".search.time_series_darts", "TimeSeriesDARTS"),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
