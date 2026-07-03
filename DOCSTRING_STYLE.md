# Docstring Style Guide for Foreblocks

## Module Docstrings

**Goal**: Reader understands problem solved + core concept without reading the full module.

**Structure**:

```python
"""foreblocks.package.module.

<One-line problem or capability>

<2-3 sentence explanation of the core concept and when/why to use it>

Core API:
- ClassName: brief description
- function_name: brief description
- ...
"""
```

**Example** (mhc.py):
```python
"""foreblocks.models.transformer.mhc.

Multi-stream information routing via learned, manifold-constrained gating.

MHC (Manifold Hyper-Connection) enables transformer blocks to maintain and
intelligently mix multiple parallel representation streams, routing information
across streams based on learned pre-aggregation, post-write, and residual gates
that satisfy doubly-stochastic constraints. Designed for models that separate
concerns into distinct streams (e.g., temporal trend, seasonality, anomaly).

Core API:
- MHCHyperConnection: learnable stream routing module
- mhc_init_streams, mhc_collapse_streams: stream lifecycle helpers
- sinkhorn_doubly_stochastic: differentiable matrix projection
"""
```

### What to avoid

- ❌ Describing the file path/package structure (reader already knows this)
- ❌ Generic "This module contains X utilities"
- ❌ Listing all functions/classes (only core API)
- ❌ Implementation details (those belong in method docstrings)

### What to include

- ✅ The problem or capability in one sentence
- ✅ Why someone would use this module
- ✅ When to use it (constraints, prerequisites, typical workflows)
- ✅ Top-level classes and functions that users need to know

---

## Class Docstrings

**Goal**: Understand class purpose, tensor shapes, and key methods at a glance.

**Structure**:

```python
class ClassName(nn.Module):
    """
    <One-line problem/capability>
    
    <Optional detailed explanation if not obvious from the name>
    
    Attributes:
        param1: description
        param2: description
    
    Shape conventions:
        input: [B, T, D]
        output: [B, T, D]
    """
```

**Example**:
```python
class MHCHyperConnection(nn.Module):
    """
    Token-wise manifold-constrained Hyper-Connection.
    
    Routes information across N parallel representation streams using learned
    gates that maintain doubly-stochastic flow constraints.
    
    Attributes:
        pre_proj: stream selection gate (learns which stream to read from)
        post_proj: block output routing gate (learns which stream to write to)
        res_proj: inter-stream mixing (doubly-stochastic coupling)
    
    Shape conventions:
        streams: [B, N, T, D] (B batches, N streams, T timesteps, D dimensions)
        maps: {pre: [B, T, N], post: [B, T, N], res: [B, T, N, N]}
    """
```

---

## Function Docstrings

Keep brief unless the function is complex or has surprising behavior.

```python
def function_name(arg1: Type, arg2: Type) -> ReturnType:
    """One-line summary of what it does.
    
    Optional: longer explanation only if the behavior is non-obvious.
    
    Args:
        arg1: what it is and constraints
        arg2: what it is and constraints
    
    Returns:
        what it returns and shape/type info
    
    Raises:
        ValueError: when this happens
    """
```

---

## Comments vs Docstrings

- **Docstrings** (triple-quoted): explain *why* use this, *what* it does, interface/contract
- **Comments** (inline `#`): explain *why* a specific implementation choice (not obvious from reading)

Only add comments when the code wouldn't be clear otherwise. Good names eliminate most comments.

---

## Examples in Docstrings

Include if the module/class is complex or has surprising usage patterns.

```python
"""
...
    
Example:
    >>> x = torch.randn(2, 3, 64, 128)  # [B=2, T=64, D=128]
    >>> streams = mhc_init_streams(x, n_streams=4)  # [B, N=4, T, D]
    >>> hc = MHCHyperConnection(d_model=128, n_streams=4)
    >>> aggregated, maps = hc.pre_aggregate(streams)
    >>> # aggregated: [B, T, D], maps used for combined write
"""
```

