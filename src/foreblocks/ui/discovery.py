"""foreblocks.ui.discovery.

Node discovery and auto-categorization for the foreblocks Studio UI.

Scans registered foreblocks packages for node-annotated classes, builds a
category index, and produces JSON-serializable discovery payloads. Used by
the Studio server to populate the node palette and support drag-and-drop
workflow construction.

Core API:
- discover_nodes: find all registered foreblocks nodes
- discover_nodes_payload: generate JSON-serializable discovery data
- categories_map: map category IDs to display names

"""

# foreblocks/discovery.py
from __future__ import annotations

import importlib
import inspect
import os
import pkgutil
from collections.abc import Iterable

from foreblocks.ui.auto_spec import build_node_spec

ALLOWED_PACKAGES = [
    "foreblocks.modules.blocks",
    "foreblocks.core",
    "foreblocks.modules.heads",
    "foreblocks.data",
    "foreblocks.models.popular",
    "foreblocks.models.transformer",
    "foreblocks.sequence.forecast_blocks",
    "foreblocks.core.training",
]


def _iter_modules(package_name: str) -> Iterable[str]:
    pkg = importlib.import_module(package_name)
    if hasattr(pkg, "__path__"):
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            yield mod.name
    else:
        yield pkg.__name__


def _all_candidate_classes() -> Iterable[type]:
    seen_modules: set[str] = set()
    seen_classes: set[tuple[str, str]] = set()
    for root in ALLOWED_PACKAGES:
        for modname in _iter_modules(root):
            if modname in seen_modules:
                continue
            seen_modules.add(modname)
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if obj.__module__ == mod.__name__:
                    key = (obj.__module__, obj.__qualname__)
                    if key in seen_classes:
                        continue
                    seen_classes.add(key)
                    yield obj


def _discover_node_specs() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for cls in _all_candidate_classes():
        # Only include classes that have been explicitly decorated (not inherited)
        if "__is_node__" not in cls.__dict__:
            continue

        if os.environ.get("FOREBLOCKS_UI_DISCOVERY_DEBUG"):
            print(f"[Discovery] Found node class: {cls.__name__} in {cls.__module__}")

        opts = getattr(cls, "__node_options__", {}) or {}

        # Build a normalized spec (includes inputs/outputs/config + py)
        try:
            spec = build_node_spec(cls, opts)
        except Exception:
            # If something goes wrong building spec for this class, skip it
            continue

        type_id = spec.get("type_id")
        if not type_id:
            # Fallback: class name lowercased
            type_id = getattr(cls, "__name__", "node").lower()
            spec["type_id"] = type_id

        if type_id in out:
            # Prefer first one to keep stability; skip duplicates silently
            continue

        out[type_id] = spec
    return out


def categories_map(nodes: dict[str, dict]) -> dict[str, list[str]]:
    cats: dict[str, list[str]] = {}
    for t, spec in nodes.items():
        cats.setdefault(spec.get("category", "Misc"), []).append(t)
    for k in cats:
        cats[k].sort()
    return cats


# ── Back-compat function: preserve original signature if other code expects it
def discover_nodes() -> dict[str, dict]:
    specs = _discover_node_specs()
    # Shape similar to before, but now includes 'py'
    simplified = {}
    for type_id, spec in specs.items():
        simplified[type_id] = {
            "name": spec.get("name"),
            "category": spec.get("category"),
            "inputs": spec.get("inputs", []),
            "optional_inputs": spec.get("optional_inputs", []),
            "outputs": spec.get("outputs", []),
            "config": spec.get("config", {}),
            "subtypes": spec.get("subtypes", []),
            "color": spec.get("color"),
            "py": spec.get("py", {}),  # <── NEW for generic code-gen
        }
    return simplified


# ── Preferred endpoint payload for /nodes
def discover_nodes_payload() -> dict[str, dict]:
    specs = _discover_node_specs()
    # Strip to the fields the frontend cares about
    nodes_payload = {
        tid: {
            "name": s.get("name"),
            "category": s.get("category"),
            "color": s.get("color"),
            "subtypes": s.get("subtypes", []),
            "inputs": s.get("inputs", []),
            "optional_inputs": s.get("optional_inputs", []),
            "outputs": s.get("outputs", []),
            "config": s.get("config", {}),
            "py": s.get("py", {}),  # <── include codegen metadata
        }
        for tid, s in specs.items()
    }
    return {
        "nodes": nodes_payload,
        "categories": categories_map(nodes_payload),
    }
