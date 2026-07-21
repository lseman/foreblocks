"""foreblocks.ui.node_spec.

Node discovery decorator for annotating model components with UI metadata.

Provides the `node` decorator used to mark classes for auto-discovery by the
foreblocks Studio. Supports type_id, category, color, inputs/outputs config,
and optional explicit codegen specs. Decorated classes become available in
the Studio node palette for drag-and-drop workflow construction.

Core API:
- node: decorator to register a class as a discoverable foreblocks node

"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def node(
    *,
    type_id: str | None = None,
    name: str | None = None,
    category: str = "Misc",
    color: str = "bg-gradient-to-br from-slate-700 to-slate-800",
    subtypes: list[str] | None = None,
    # Optional hard overrides (skip inference for these fields if provided)
    inputs: list[str] | None = None,
    optional_inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    config: dict[str, Any] | None = None,
    config_sources: list[type] | None = None,
    # Optional explicit codegen spec (if provided, we won't auto-infer it)
    py: dict[str, Any] | None = None,
    # Control whether we run auto-inference (default yes)
    infer: bool = True,
) -> Callable[[type], type]:

    def wrap(cls: type) -> type:
        cls.__is_node__ = True  # <- discovery gate
        setattr(
            cls,
            "__node_options__",
            {
                "type_id": type_id,
                "name": name,
                "category": category,
                "color": color,
                "subtypes": subtypes or [],
                "overrides": {
                    "inputs": inputs,
                    "optional_inputs": optional_inputs,
                    "outputs": outputs,
                    "config": config,
                    "config_sources": config_sources or [],
                },
                "py": py,  # explicit codegen metadata (overrides auto-inference)
                "infer": infer,
            },
        )
        return cls

    return wrap
