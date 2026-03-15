# Hybrid Mamba

`foreblocks.hybrid_mamba` provides custom SSM (State Space Model) building blocks that combine selective-scan dynamics with sliding-window attention. Two block variants are available:

- **`HybridMambaBlock`** — pure SSM (Mamba v1-style selective scan)
- **`HybridMamba2Block`** — parallel SSM + sliding-window attention branches fused with a learned gate (Mamba-2 / SSD-based)

## Installation requirements

The module is pure PyTorch by default. Optional acceleration layers:

| Backend | What it enables | How to activate |
|---|---|---|
| Triton | Fused causal conv and grouped-SSD scan | `pip install triton` |
| CUDA extension | Selective scan CUDA kernel | `precompile_selective_scan_extension()` |

Check availability at runtime:

```python
from foreblocks.hybrid_mamba import TRITON_AVAILABLE, extension_available

print(TRITON_AVAILABLE)       # True if triton is installed
print(extension_available())  # True if CUDA extension is built
```

## `HybridMambaBlock`

A single Mamba-style block. Expands `d_model` → `d_inner` via a causal depthwise conv, runs selective scan, then projects back.

```python
import torch
from foreblocks.hybrid_mamba import HybridMambaBlock

block = HybridMambaBlock(
    d_model=256,
    d_inner=512,   # defaults to 2 * d_model
    d_state=16,
    d_conv=4,
    dt_rank=None,  # auto: max(4, ceil(d_model / 16))
    use_cuda_scan=True,
)

x = torch.randn(8, 64, 256)  # (batch, seq_len, d_model)
y = block(x)                 # same shape as x
```

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | required | Input / output feature dimension |
| `d_inner` | `2 * d_model` | Inner (expanded) dimension |
| `d_state` | `16` | SSM state dimension per feature |
| `d_conv` | `4` | Causal conv kernel size |
| `dt_rank` | auto | Low-rank projection for Δt; `None` → `max(4, ceil(d_model/16))` |
| `dt_min` / `dt_max` | `1e-4` / `1.0` | Clamp range for the time-step after softplus |
| `use_cuda_scan` | `True` | Use CUDA kernel if extension is loaded; falls back to PyTorch otherwise |

## `HybridMamba2Block`

Combines a multi-head SSD branch (`StructuredStateSpaceDualityBranch`) with a `SlidingWindowAttention` branch. The two outputs are mixed with a sigmoid gate:

```
output = sigmoid(gate(x)) * SSM(x) + (1 - sigmoid(gate(x))) * Attn(x)
```

```python
from foreblocks.hybrid_mamba import HybridMamba2Block

block = HybridMamba2Block(
    d_model=256,
    d_inner=512,
    d_state=16,
    d_conv=4,
    dt_rank=None,
    num_heads=8,
    window_size=128,    # attention only looks back this many tokens
    attn_dropout=0.0,
    use_gated_delta=False,
)

x = torch.randn(4, 128, 256)
y = block(x)  # (4, 128, 256)
```

### Additional parameters over `HybridMambaBlock`

| Parameter | Default | Description |
|---|---|---|
| `num_heads` | `8` | Heads for both SSD and attention |
| `window_size` | `128` | Sliding-window size for local attention |
| `attn_dropout` | `0.0` | Attention dropout during training |
| `use_gated_delta` | `False` | Add per-head gate on Δt in the SSD branch |

## Stacking blocks into a model

`TinyHybridMamba2LM` shows the recommended stacking pattern: every `attn_every_n` layers uses a `HybridMamba2Block`; all others use the cheaper `HybridMambaBlock`.

```python
from foreblocks.hybrid_mamba import TinyHybridMamba2LM

model = TinyHybridMamba2LM(
    vocab_size=50257,
    d_model=512,
    n_layers=12,
    d_state=16,
    d_conv=4,
    num_heads=8,
    window_size=256,
    attn_every_n=4,     # HybridMamba2Block at layers 0, 4, 8; rest are plain Mamba
    tie_embeddings=True,
)
```

For use as a time-series backbone, replace the embedding + LM-head with your own projection layers and feed patch embeddings or raw-feature vectors in place of `input_ids`.

## Diagnostics

```python
from foreblocks.hybrid_mamba import run_default_diagnostics, benchmark_block

run_default_diagnostics()   # quick correctness checks for ops on current device

stats = benchmark_block(d_model=256, seq_len=512, batch=8)
print(stats)  # wall-clock and memory stats
```

## Ops reference

| Symbol | Where | Description |
|---|---|---|
| `causal_depthwise_conv1d` | `ops/` | Causal grouped conv; Triton kernel when available |
| `selective_scan` | `ops/` | S4/Mamba v1 scan; CUDA kernel when loaded |
| `grouped_ssd_scan` | `ops/` | Multi-head SSD scan (Mamba-2); Triton kernel when available |
| `dt_prep` | `ops/` | Δt bias add + softplus + clamp |
| `fused_out` | `ops/` | Fused RMSNorm + gate + residual add |
