# foreblocks/node.py
from __future__ import annotations

from typing import Any
from collections.abc import Callable


def node(
    *,
    type_id: str | None = None,
    name: str | None = None,
    category: str = "Misc",
    color: str = "bg-gradient-to-br from-slate-700 to-slate-800",
    subtypes: list[str] | None = None,
    # Optional hard overrides (skip inference for these fields if provided)
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    config: dict[str, Any] | None = None,
    config_sources: list[type] | None = None,
    # Optional explicit codegen spec (if provided, we won't auto-infer it)
    py: dict[str, Any] | None = None,
    # Control whether we run auto-inference (default yes)
    infer: bool = True,
) -> Callable[[type], type]:
    """
    Decorate a class to participate in node discovery.
    Only decorated classes are considered. Inference is run unless you override.
    """
    def wrap(cls: type) -> type:
        setattr(cls, "__is_node__", True)  # <- discovery gate
        setattr(cls, "__node_options__", {
            "type_id": type_id,
            "name": name,
            "category": category,
            "color": color,
            "subtypes": subtypes or [],
            "overrides": {
                "inputs": inputs,
                "outputs": outputs,
                "config": config,
                "config_sources": config_sources or [],
            },
            "py": py,     # <── NEW: optional explicit codegen metadata
            "infer": infer,
        })
        return cls
    return wrap
