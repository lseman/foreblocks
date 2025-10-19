# foreblocks/auto_spec.py
from __future__ import annotations

"""
Auto-spec utilities for node discovery:
- Infers inputs/outputs/config from class annotations, dataclasses, pydantic, or __init__.
- Builds a generic 'py' codegen spec when not explicitly provided by the node author.
"""

import collections.abc as cabc
import enum
import inspect
from dataclasses import fields, is_dataclass
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

# ──────────────────────────────────────────────────────────────────────────────
# Port markers used in type annotations
# ──────────────────────────────────────────────────────────────────────────────

class PortIn:
    """Marker for input ports in forward(...) annotations via Annotated[..., PortIn]."""
    ...

class PortOutBundle:
    """Marker to declare multiple output port names in the return type via Annotated[..., PortOutBundle('a','b',...)]"""
    def __init__(self, *names: str) -> None:
        self.names = names


# ──────────────────────────────────────────────────────────────────────────────
# Helpers for type inspection
# ──────────────────────────────────────────────────────────────────────────────

def _is_annotated_port(tp, marker) -> bool:
    return get_origin(tp) is Annotated and any(
        isinstance(a, marker) or a is marker for a in get_args(tp)[1:]
    )

def _typed_dict_keys(tp) -> Optional[List[str]]:
    anns = getattr(tp, "__annotations__", None)
    return list(anns.keys()) if isinstance(anns, dict) else None

def _unwrap_annotated(tp):
    origin = get_origin(tp)
    if origin is Annotated:
        args = get_args(tp)
        return args[0] if args else tp
    return tp

def _ann_is_optional(tp) -> bool:
    return get_origin(tp) is Union and any(a is type(None) for a in get_args(tp))

def _ann_is_callable(tp) -> bool:
    base = _unwrap_annotated(tp)
    origin = get_origin(base)
    return base is cabc.Callable or origin is cabc.Callable

def _ann_is_module_like(tp) -> bool:
    """
    True if annotation is nn.Module or a subclass (optionally Optional[...] / Union[..., None]).
    Import torch.nn lazily so this file works without torch installed until needed.
    """
    base = _unwrap_annotated(tp)
    origin = get_origin(base)
    args = get_args(base)

    try:
        import torch.nn as nn  # lazy
    except Exception:
        nn = None

    def _is_module_cls(x):
        return isinstance(x, type) and nn is not None and issubclass(x, nn.Module)

    # Optional/Union
    if origin is Union and args:
        for a in args:
            if a is type(None):
                continue
            if _is_module_cls(_unwrap_annotated(a)):
                return True
        return False

    # Direct nn.Module or subclass
    if _is_module_cls(base):
        return True

    # Exact nn.Module type (not subclass)
    try:
        import torch.nn as nn2
        if base is nn2.Module:
            return True
    except Exception:
        pass

    return False


# ──────────────────────────────────────────────────────────────────────────────
# Safe defaults for required __init__ params
# ──────────────────────────────────────────────────────────────────────────────

def _safe_default_for_annotation(ann):
    """
    Produce a JSON-safe default based on a typing annotation.
    Falls back to None if we cannot infer safely.
    """
    if ann is inspect._empty:
        return None

    ann = _unwrap_annotated(ann)
    origin = get_origin(ann)
    args   = get_args(ann)

    # Optional/Union[..., None]
    if origin is Union and any(a is type(None) for a in args):
        return None

    # Literal[...] -> pick first literal
    if origin is Literal and len(args) > 0:
        return args[0]

    # Collections
    if origin in (list, List, Sequence, tuple, Tuple):
        return []
    if origin in (dict, Dict, Mapping):
        return {}
    if origin in (set, Set):
        return []  # JSON-safe

    # Primitives
    if ann in (int,):
        return 0
    if ann in (float,):
        return 0.0
    if ann in (bool,):
        return False
    if ann in (str,):
        return ""

    # Enums
    try:
        if isinstance(ann, type) and issubclass(ann, enum.Enum):
            members = list(ann)
            return members[0].value if members else None
    except Exception:
        pass

    # Unknown -> None
    return None

def _safe_default_for_param(p: inspect.Parameter) -> Any:
    """Decide a safe default for a required __init__ parameter."""
    return _safe_default_for_annotation(p.annotation)


# ──────────────────────────────────────────────────────────────────────────────
# Config inference
# ──────────────────────────────────────────────────────────────────────────────

def _collect_config_from_dataclass(cfg_cls) -> Dict[str, Any]:
    # requires default values
    inst = cfg_cls()
    return {f.name: getattr(inst, f.name) for f in fields(cfg_cls)}

def _collect_config_from_pydantic(cfg_cls) -> Dict[str, Any]:
    # pydantic v2
    inst = cfg_cls()
    return inst.model_dump()

def _collect_config_from_init(cls) -> Dict[str, Any]:
    """
    Include ALL non-variadic parameters (except self):
      - if default exists: use it
      - else: synthesize a safe default from annotation (may be None)
    """
    cfg: Dict[str, Any] = {}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return cfg
    for name, p in sig.parameters.items():
        if name in ("self", "args", "kwargs", "*args", "**kwargs"):
            continue
        if p.default is not inspect._empty:
            cfg[name] = p.default
        else:
            cfg[name] = _safe_default_for_param(p)
    return cfg

def infer_config(cls) -> Dict[str, Any]:
    """
    Order:
      - If inner class `Config` exists: try dataclass → try pydantic v2
      - Else fall back to __init__ signature (now includes safe defaults)
    """
    Config = getattr(cls, "Config", None)
    if Config is not None:
        try:
            if is_dataclass(Config):
                return _collect_config_from_dataclass(Config)
        except Exception:
            pass
        try:
            from pydantic import BaseModel as _BM  # type: ignore
            if isinstance(Config, type) and issubclass(Config, _BM):
                return _collect_config_from_pydantic(Config)
        except Exception:
            pass
    return _collect_config_from_init(cls)


# ──────────────────────────────────────────────────────────────────────────────
# Inputs / Outputs inference
# ──────────────────────────────────────────────────────────────────────────────

# Names that should be considered "component inputs" if present in __init__,
# regardless of exact annotation (useful when hints are missing/Any)
_WHITELIST_INIT_INPUTS: Set[str] = {
    # core
    "encoder", "decoder", "head",
    # processing / pre/post
    "input_preprocessor", "output_postprocessor",
    "input_normalization", "output_normalization",
    "output_block", "input_skip_connection_module",
    # attention & scheduling
    "attention_module", "scheduled_sampling_fn",
    # time features
    "time_feature_embedding_enc", "time_feature_embedding_dec",
    # head composer
    "head_composer",
}

def _looks_like_init_input(name: str, ann) -> bool:
    """
    Heuristic: treat as an input port if either:
      - Name is whitelisted (common component names), OR
      - Annotation is nn.Module or subclass (optionally Optional[...] / Union[..., None]), OR
      - Annotation is Callable (optionally Optional[...] / Union[..., None]).
    """
    if name in _WHITELIST_INIT_INPUTS:
        return True
    if _ann_is_module_like(ann):
        return True
    # Callable or Optional[Callable]
    if _ann_is_callable(ann):
        return True
    base = _unwrap_annotated(ann)
    if _ann_is_optional(base):
        # Optional[...] where inner is callable?
        inner = [a for a in get_args(base) if a is not type(None)]
        if inner and _ann_is_callable(inner[0]):
            return True
    return False

def infer_inputs(cls) -> List[str]:
    """
    STRICT-STRICT + INIT-COMPONENTS:
      - If class defines __inputs__, use that.
      - Else:
          * Scan forward(...) parameters: ONLY Annotated[..., PortIn]
          * ALSO scan __init__ parameters: treat component-like params as inputs
            (nn.Module-like, Callable-like, or whitelisted names).
      - We DO NOT treat `torch.Tensor` (or any other type) as implicit inputs.
    """
    if hasattr(cls, "__inputs__"):
        return list(getattr(cls, "__inputs__"))

    ins: List[str] = []

    # 1) forward(...) explicit PortIn markers
    fwd = getattr(cls, "forward", None)
    if callable(fwd):
        try:
            hints = get_type_hints(fwd, include_extras=True)
        except Exception:
            hints = {}
        try:
            sig = inspect.signature(fwd)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                ann = hints.get(name)
                if ann is not None and _is_annotated_port(ann, PortIn):
                    ins.append(name)

    # 2) __init__(...) component-like parameters as inputs
    try:
        sig_init = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        sig_init = None

    if sig_init is not None:
        try:
            init_hints = get_type_hints(cls.__init__, include_extras=True)
        except Exception:
            init_hints = {}
        for name, p in sig_init.parameters.items():
            if name in ("self", "args", "kwargs", "*args", "**kwargs"):
                continue
            ann = init_hints.get(name, p.annotation)
            if _looks_like_init_input(name, ann):
                if name not in ins:
                    ins.append(name)

    return ins


def infer_outputs(cls) -> List[str]:
    """
    - If class defines __outputs__, use that.
    - Else inspect forward(...) return annotation:
        * NamedTuple subclass → its _fields
        * TypedDict → keys
        * Annotated[..., PortOutBundle(...)] → names
        * Else try example_io() probe hook (optional)
    """
    if hasattr(cls, "__outputs__"):
        return list(getattr(cls, "__outputs__"))

    fwd = getattr(cls, "forward", None)
    if not callable(fwd):
        return []

    try:
        hints = get_type_hints(fwd, include_extras=True)
    except Exception:
        hints = {}
    ret = hints.get("return")
    if ret is None:
        return []

    # NamedTuple subclass
    if isinstance(ret, type) and issubclass(ret, tuple) and hasattr(ret, "_fields"):
        return list(getattr(ret, "_fields"))

    # TypedDict
    keys = _typed_dict_keys(ret)
    if keys:
        return keys

    # Annotated[..., PortOutBundle(...)]
    if get_origin(ret) is Annotated:
        metas = get_args(ret)[1:]
        for m in metas:
            if isinstance(m, PortOutBundle):
                return list(m.names)

    # Optional runtime probe if author provided example inputs
    probe = getattr(cls, "example_io", None)
    if callable(probe):
        try:
            inst = cls()                 # requires defaults to instantiate
            example = inst.example_io()  # dict of inputs
            out = inst.forward(**example)
            if isinstance(out, dict):
                return list(out.keys())
            if hasattr(out, "_asdict"):
                return list(out._asdict().keys())
        except Exception:
            pass

    return []


# ──────────────────────────────────────────────────────────────────────────────
# Generic code-gen spec inference (NEW)
# ──────────────────────────────────────────────────────────────────────────────

def _infer_ctor_and_bindings(cls: type, inputs: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic for building a default 'py' spec when authors don't provide one.

    - ctor  := cls.__name__
    - imports := ["from {cls.__module__} import {cls.__name__}"]
    - var_prefix := lowercased class name (safe default)
    - bind.kwargs:
        For each __init__ parameter (excluding self/variadics):
          - if param name in inputs → "@input:<name>"
          - elif param name in cfg  → "@config:<name>"
          - else: skip (user can override in decorator or __py__/py_spec)
    - output_map: omitted → frontend will default each output port to "@self"
    """
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        sig = None

    if sig is not None:
        for pname, p in sig.parameters.items():
            if pname in ("self", "*args", "**kwargs", "args", "kwargs"):
                continue
            if pname in (inputs or []):
                kwargs[pname] = f"@input:{pname}"
            elif pname in (cfg or {}):
                kwargs[pname] = f"@config:{pname}"
            # else skip

    py = {
        "imports": [f"from {cls.__module__} import {cls.__name__}"],
        "ctor": cls.__name__,
        "var_prefix": getattr(cls, "__name__", "node").lower(),
        "bind": {"kwargs": kwargs},
        # "output_map": { ... }  # optional; default handled in frontend
    }
    return py


def build_node_spec(cls: type, options: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge:
      - decorator overrides
      - inferred ports & config
      - explicit or inferred 'py' codegen spec
    into a single dict ready to be serialized by /nodes.

    Precedence for 'py':
      decorator.py  >  cls.__py__  >  cls.py_spec()  >  inferred default
    """
    # 1) Gather overrides and perform inference (if allowed)
    ov = options.get("overrides", {}) or {}
    do_infer = options.get("infer", True)

    inferred_cfg = infer_config(cls) if do_infer else {}
    inferred_in  = infer_inputs(cls) if do_infer else []
    inferred_out = infer_outputs(cls) if do_infer else []

    inputs  = ov.get("inputs")  or inferred_in
    outputs = ov.get("outputs") or inferred_out
    config  = ov.get("config")  or inferred_cfg

    # 2) Resolve 'py' spec
    explicit_py    = options.get("py")
    class_py       = getattr(cls, "__py__", None)
    class_py_spec  = getattr(cls, "py_spec", None)

    if explicit_py is not None:
        py = explicit_py
    elif class_py is not None:
        py = class_py
    elif callable(class_py_spec):
        try:
            py = class_py_spec()
        except Exception:
            py = _infer_ctor_and_bindings(cls, inputs, config)
    else:
        py = _infer_ctor_and_bindings(cls, inputs, config)

    # 3) Final normalized spec
    type_id = options.get("type_id") or cls.__name__
    name    = options.get("name") or cls.__name__

    return {
        "type_id": type_id,
        "name": name,
        "category": options.get("category") or "Misc",
        "color": options.get("color") or "bg-gradient-to-br from-slate-700 to-slate-800",
        "subtypes": list(options.get("subtypes") or []),
        "inputs": list(inputs or []),
        "outputs": list(outputs or []),
        "config": dict(config or {}),
        "py": dict(py or {}),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Public exports
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "PortIn",
    "PortOutBundle",
    "infer_config",
    "infer_inputs",
    "infer_outputs",
    "build_node_spec",
]
