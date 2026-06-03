# custom_mamba

A self-contained PyTorch module that implements **Mamba2-style sequence mixing** with chunked SSD scan kernels and an optional sliding-window attention hybrid layer.

Two hybrid flavours are provided:

| Class | Mixing strategy |
|---|---|
| `Mamba2Block` | Mamba2-style diagonal-A chunked SSD (SSM only, no attention) |
| `HybridMamba2Block` | Mamba2 SSD branch + sliding-window attention, gated mix output |

Both blocks are ready to stack inside any encoder or autoregressive language model.

---

## Installation

The Python layers work with any recent PyTorch installation. `custom_mamba`
uses PyTorch reference code plus Triton kernels where available.

```bash
# From the foreblocks root
pip install -e foreblocks/custom_mamba

```

---

## Quick start

### Single block

```python
import torch
from foreblocks.custom_mamba import HybridMamba2Block, Mamba2Block

# Pure Mamba2 SSM block
block = Mamba2Block(
    d_model=256,
    d_state=16,
    d_conv=4,
    chunk_size=256,
)
x = torch.randn(2, 512, 256)   # (batch, seq_len, d_model)
y = x + block(x)               # residual connection

# SSM + sliding-window attention block
block2 = HybridMamba2Block(
    d_model=256,
    d_state=16,
    num_heads=8,
    window_size=128,
)
y2 = x + block2(x)
```

---

## Architecture

```
Input x  (B, T, D)
    │
    ├─ SSM branch (Mamba2Block)
    │      in_proj → causal_conv1d → chunked_ssd_forward → out_proj
    │
    ├─ Attention branch (HybridMamba2Block only)
    │      SlidingWindowAttention with RoPE + optional GQA + sink tokens
    │
    └─ Gated mix → LayerNorm → out_proj → residual add
```

### Key components

**`Mamba2Block`** (`blocks/mamba2.py`)
- Diagonal per-head `A`, grouped `B/C`, per-head `D`, and chunked SSD scan.
- Causal depthwise conv1d gate before the scan.
- FLA-style knobs for conv init/bias, projection bias, dt init range, runtime `dt_limit`, and padding masks.
- Supports token-by-token recurrent inference via `make_state` / `step`.

**`HybridMamba2Block`** (`blocks/hybrid.py`)
- Combines an SSD (Structured State Space Duality) branch with `SlidingWindowAttention`.
- Learned gating merges both streams before a shared output projection.
- Optional layer-scale parameter for improved deep-stack stability.

**`SlidingWindowAttention`** (`blocks/attention.py`)
- Causal local attention; each token attends to at most `window_size` past tokens.
- Rotary position embeddings (RoPE).
- Grouped Query Attention (GQA) via `n_kv_heads`.
- Attention sink tokens (StreamingLLM-style) to prevent boundary starvation.
- Optional per-head QK-norm and logit softcap.

---

## Ops layer

Low-level kernels live in `ops/` with three backends that are selected automatically:

| Op | Pure Python | Triton | CUDA ext |
|---|---|---|---|
| `causal_depthwise_conv1d` | `causal_depthwise_conv1d_reference` | `causal_depthwise_conv1d_triton` | — |
| `chunked_ssd_forward` | `chunked_ssd_forward_reference` | `chunked_ssd_forward_triton` | — |
| `mamba2_split_conv1d_scan_combined` | local composed fallback | conv/SSD/norm Triton subkernels | — |
| `dt_prep` / `fused_out` | `*_fallback` | `*_triton` | — |
| `rotary_apply` | `rotary_apply_fallback` | `rotary_apply` fast path | — |
| `rms_norm` | `rms_norm_fallback` | `rms_norm` fast path | — |

Triton availability is exposed as `TRITON_AVAILABLE`, `CAUSAL_CONV1D_TRITON_AVAILABLE`, `CHUNKED_SSD_TRITON_AVAILABLE`, `ROTARY_TRITON_AVAILABLE`, and `RMS_NORM_TRITON_AVAILABLE`.

---

## Diagnostics

```python
pytest tests/test_hybrid_mamba.py
```

---

## Module layout

```
custom_mamba/
├── __init__.py          Public API re-exports
├── setup.py             Package metadata
├── blocks/
│   ├── hybrid.py        HybridMamba2Block
│   ├── mamba2.py        Mamba2Block
│   ├── attention.py     SlidingWindowAttention
│   ├── conv.py          CausalDepthwiseConv1d
│   ├── feedforward.py   FeedForward (SwiGLU)
│   ├── norms.py         RMSNorm / RMSNormWeightOnly
│   └── rotary.py        RotaryEmbedding
└── ops/
    ├── causal_conv1d.py    Causal depthwise conv (Triton / reference)
    ├── mamba2_combined.py  FLA-style split/conv/scan combined path
    ├── rms_norm.py         RMSNorm for attention Q/K norm
    ├── rotary.py           RoPE apply kernel
    ├── ssd.py              Diagonal-A chunked SSD scan
    └── triton_ops.py       dt_prep, fused_out Triton kernels
```
