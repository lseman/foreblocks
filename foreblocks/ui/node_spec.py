# foreblocks/node.py
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


def node(
    *,
    type_id: Optional[str] = None,
    name: Optional[str] = None,
    category: str = "Misc",
    color: str = "bg-gradient-to-br from-slate-700 to-slate-800",
    subtypes: Optional[List[str]] = None,
    # Optional hard overrides (skip inference for these fields if provided)
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    # Optional explicit codegen spec (if provided, we won't auto-infer it)
    py: Optional[Dict[str, Any]] = None,
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
            },
            "py": py,     # <── NEW: optional explicit codegen metadata
            "infer": infer,
        })
        return cls
    return wrap
