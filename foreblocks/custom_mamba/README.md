# custom_mamba

A self-contained PyTorch module that implements **Hybrid Mamba** — a family of sequence-mixing blocks that combine a selective state-space model (SSM) branch with sliding-window attention inside a single residual layer.

Two hybrid flavours are provided:

| Class | Mixing strategy |
|---|---|
| `HybridMambaBlock` | Pure Mamba-style selective scan (SSM only, no attention) |
| `HybridMamba2Block` | SSD branch + sliding-window attention, gated mix output |

Both blocks are ready to stack inside any encoder or autoregressive language model.

---

## Installation

The Python layers work with any recent PyTorch installation. The optional CUDA extension accelerates the selective-scan kernel; it is compiled on first use via `torch.utils.cpp_extension.load`.

```bash
# From the foreblocks root
pip install -e foreblocks/custom_mamba

# Optionally pre-compile the CUDA extension
python -m custom_mamba --precompile
```

Environment variables that control the CUDA build:

| Variable | Default | Purpose |
|---|---|---|
| `CUSTOM_MAMBA_BUILD_DIR` | `<package>/.build` | Where compiled artifacts are cached |
| `CUSTOM_MAMBA_CUDA_ARCH_LIST` | `8.0;8.6;8.9;9.0` | Target GPU architectures |
| `CUSTOM_MAMBA_FORCE_CUDA_VERSION` | `0` | Force a specific CUDA runtime version |

---

## Quick start

### Single block

```python
import torch
from foreblocks.custom_mamba import HybridMambaBlock, HybridMamba2Block

# Pure SSM block
block = HybridMambaBlock(d_model=256, d_state=16, d_conv=4)
x = torch.randn(2, 512, 256)   # (batch, seq_len, d_model)
y = x + block(x)               # residual connection

# SSM + sliding-window attention block
block2 = HybridMamba2Block(
    d_model=256,
    d_state=16,
    num_heads=8,
    window_size=128,
)
y2 = block2(x)
```

### Complete language model

```python
from foreblocks.custom_mamba import TinyHybridMambaLM, TinyHybridMamba2LM

model = TinyHybridMambaLM(
    vocab_size=32_000,
    d_model=256,
    n_layers=6,
    mlp_every_n=2,          # insert a FeedForward after every 2nd SSM layer
)
logits = model(input_ids)   # (batch, seq_len, vocab_size)

# Autoregressive generation (uses fast recurrent step mode)
output = model.generate(input_ids, max_new_tokens=128, temperature=0.8)
```

---

## Architecture

```
Input x  (B, T, D)
    │
    ├─ SSM branch (HybridMambaBlock / SSD branch)
    │      in_proj → causal_conv1d → selective_scan → out_proj
    │
    ├─ Attention branch (HybridMamba2Block only)
    │      SlidingWindowAttention with RoPE + optional GQA + sink tokens
    │
    └─ Gated mix → LayerNorm → out_proj → residual add
```

### Key components

**`HybridMambaBlock`** (`blocks/ssm.py`)
- Selective scan with dense per-feature `A` / `D` matrices.
- Optional CUDA kernel via the compiled extension; falls back to a pure-Python reference implementation.
- Causal depthwise conv1d gate before the scan.
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

**`StructuredStateSpaceDualityBranch`** (`blocks/ssd.py`)
- Multi-head SSD scan with grouped heads and input-dependent `dt` / `B` / `C`.

---

## Ops layer

Low-level kernels live in `ops/` with three backends that are selected automatically:

| Op | Pure Python | Triton | CUDA ext |
|---|---|---|---|
| `selective_scan` | `selective_scan_reference` | — | `selective_scan` |
| `causal_depthwise_conv1d` | `causal_depthwise_conv1d_reference` | `causal_depthwise_conv1d_triton` | — |
| `grouped_ssd_scan` | `grouped_ssd_scan_reference` | `grouped_ssd_scan_triton` | — |
| `dt_prep` / `fused_out` | `*_fallback` | `*_triton` | — |

Triton availability is exposed as `TRITON_AVAILABLE`, `CAUSAL_CONV1D_TRITON_AVAILABLE`, and `GROUPED_SSD_TRITON_AVAILABLE`.

---

## Diagnostics

```python
from foreblocks.custom_mamba import run_default_diagnostics

run_default_diagnostics()   # forward/backward correctness + benchmark
```

Individual checks:

```python
from foreblocks.custom_mamba import (
    check_forward_close,
    check_backward,
    benchmark_block,
    compare_against_official,
)
```

---

## Module layout

```
custom_mamba/
├── __init__.py          Public API re-exports
├── cuda.py              CUDA extension loader / pre-compiler
├── diagnostics.py       Correctness checks and micro-benchmarks
├── layers.py            Convenience layer wrappers
├── build.py / setup.py  Extension build configuration
├── csrc/                C++/CUDA source for selective scan kernel
├── blocks/
│   ├── hybrid.py        HybridMamba2Block
│   ├── ssm.py           HybridMambaBlock
│   ├── attention.py     SlidingWindowAttention
│   ├── ssd.py           StructuredStateSpaceDualityBranch
│   ├── conv.py          CausalDepthwiseConv1d
│   ├── feedforward.py   FeedForward (SwiGLU)
│   ├── norms.py         RMSNorm / RMSNormWeightOnly
│   ├── rotary.py        RotaryEmbedding
│   └── models.py        TinyHybridMambaLM / TinyHybridMamba2LM
└── ops/
    ├── selective_scan.py   Selective scan (CUDA / reference)
    ├── causal_conv1d.py    Causal depthwise conv (Triton / reference)
    ├── ssd.py              Grouped SSD scan (Triton / reference)
    └── triton_ops.py       dt_prep, fused_out Triton kernels
```
