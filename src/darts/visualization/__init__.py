"""Visualization sub-package: architecture diagram rendering.

Standalone matplotlib-based diagram generation for transformer/DARTS
architectures. Not part of the search or training pipeline — used for
producing illustrative figures from a trained model or an explicit spec.
"""

from importlib import import_module

__all__ = [
    "draw_single_block",
    "draw_encoder_decoder",
    "draw_selected_transformer_architecture",
    "make_encoder_layers",
    "make_decoder_layers",
    "make_hybrid_layers",
    "extract_selected_transformer_spec",
]


def __getattr__(name):
    lazy_exports = {
        "draw_single_block": (".transformer_diagram", "draw_single_block"),
        "draw_encoder_decoder": (".transformer_diagram", "draw_encoder_decoder"),
        "draw_selected_transformer_architecture": (
            ".transformer_diagram",
            "draw_selected_transformer_architecture",
        ),
        "make_encoder_layers": (".transformer_diagram", "make_encoder_layers"),
        "make_decoder_layers": (".transformer_diagram", "make_decoder_layers"),
        "make_hybrid_layers": (".transformer_diagram", "make_hybrid_layers"),
        "extract_selected_transformer_spec": (
            ".transformer_diagram",
            "extract_selected_transformer_spec",
        ),
    }
    if name in lazy_exports:
        module_name, attr_name = lazy_exports[name]
        module = import_module(module_name, __name__)
        return getattr(module, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
