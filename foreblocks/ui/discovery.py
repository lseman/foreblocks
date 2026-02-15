# foreblocks/discovery.py
from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Dict, Iterable, List, Tuple

# NEW: use the unified spec builder (handles inputs/outputs/config + py)
from foreblocks.ui.auto_spec import build_node_spec

ALLOWED_PACKAGES = [
    "foreblocks.blocks",
    "foreblocks.blocks.enc_dec",
    "foreblocks.core",
    "foreblocks.core.heads",
    "foreblocks.data",
    "foreblocks.aux.utils",
    "foreblocks.tf.transformer",
    "foreblocks.training",
]

def _iter_modules(package_name: str) -> Iterable[str]:
    """Yield module names under a package or just the module itself if not a package."""
    pkg = importlib.import_module(package_name)
    if hasattr(pkg, "__path__"):
        for mod in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
            yield mod.name
    else:
        yield pkg.__name__

def _all_candidate_classes() -> Iterable[type]:
    """Yield all classes found under allowed packages/modules."""
    for root in ALLOWED_PACKAGES:
        for modname in _iter_modules(root):
            try:
                mod = importlib.import_module(modname)
            except Exception:
                continue
            for _, obj in inspect.getmembers(mod, inspect.isclass):
                if obj.__module__ == mod.__name__:
                    yield obj

def _discover_node_specs() -> Dict[str, Dict]:
    """
    Discover decorated node classes and build normalized specs using build_node_spec().
    Returns a dict: {type_id: full_spec}
    """
    out: Dict[str, Dict] = {}
    for cls in _all_candidate_classes():
        # Only include classes that have been explicitly decorated (not inherited)
        if "__is_node__" not in cls.__dict__:
            continue

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

def categories_map(nodes: Dict[str, Dict]) -> Dict[str, List[str]]:
    cats: Dict[str, List[str]] = {}
    for t, spec in nodes.items():
        cats.setdefault(spec.get("category", "Misc"), []).append(t)
    for k in cats:
        cats[k].sort()
    return cats

# ── Back-compat function: preserve original signature if other code expects it
def discover_nodes() -> Dict[str, Dict]:
    """
    Back-compat: return a simplified map for callers that don't need categories.
    Each node dict contains name/category/inputs/outputs/config/subtypes/color/py.
    """
    specs = _discover_node_specs()
    # Shape similar to before, but now includes 'py'
    simplified = {}
    for type_id, spec in specs.items():
        simplified[type_id] = {
            "name": spec.get("name"),
            "category": spec.get("category"),
            "inputs": spec.get("inputs", []),
            "outputs": spec.get("outputs", []),
            "config": spec.get("config", {}),
            "subtypes": spec.get("subtypes", []),
            "color": spec.get("color"),
            "py": spec.get("py", {}),  # <── NEW for generic code-gen
        }
    return simplified

# ── Preferred endpoint payload for /nodes
def discover_nodes_payload() -> Dict[str, Dict]:
    """
    Return the full payload expected by the frontend:
      { "nodes": {type_id: {...}}, "categories": {category: [type_ids...] } }
    """
    specs = _discover_node_specs()
    # Strip to the fields the frontend cares about
    nodes_payload = {
        tid: {
            "name": s.get("name"),
            "category": s.get("category"),
            "color": s.get("color"),
            "subtypes": s.get("subtypes", []),
            "inputs": s.get("inputs", []),
            "outputs": s.get("outputs", []),
            "config": s.get("config", {}),
            "py": s.get("py", {}),   # <── include codegen metadata
        }
        for tid, s in specs.items()
    }
    return {
        "nodes": nodes_payload,
        "categories": categories_map(nodes_payload),
    }
