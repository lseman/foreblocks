---
title: Hybrid Mamba
description: Hybrid Mamba and Mamba-2 style SSM blocks for time-series forecasting.
editLink: true
---


[[toc]]
# Hybrid Mamba

`foreblocks.custom_mamba` provides custom SSM (State Space Model) building blocks that combine selective-scan dynamics with sliding-window attention. Two block variants are available:

- **`HybridMambaBlock`** — pure SSM (Mamba v1-style selective scan) with optional pre-norm
- **`HybridMamba2Block`** — parallel SSM + sliding-window attention branches fused with a learned gate, RoPE, GQA, and output normalisation (Mamba-2 / SSD-based)

Block implementations live under `foreblocks.custom_mamba.blocks`; package-level imports remain supported for compatibility.

## Installation requirements

The module is pure PyTorch by default. Optional acceleration layers:

| Backend | What it enables | How to activate |
|---|---|---|
| Triton | Fused causal conv and grouped-SSD scan | `pip install triton` |
| CUDA extension | Selective scan CUDA kernel | `precompile_selective_scan_extension()` |

Check availability at runtime:

```python
from foreblocks.custom_mamba import TRITON_AVAILABLE, extension_available

print(TRITON_AVAILABLE)       # True if triton is installed
print(extension_available())  # True if CUDA extension is built
```python

The cache is extended automatically when `T > max_seq_len`, so setting a generous upper bound is fine.

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `head_dim` | required | Dimension of each attention head. Must be even. |
| `base` | `10_000` | Frequency base (original paper default). |
| `max_seq_len` | `8192` | Pre-built cache length; extended on demand. |

## `HybridMambaBlock`

A single Mamba-style block. Expands `d_model` → `d_inner` via a causal depthwise conv, runs selective scan, then projects back. An optional pre-norm (`use_pre_norm=True`) stabilises training in deep stacks.

```python
import torch
from foreblocks.custom_mamba import HybridMambaBlock

block = HybridMambaBlock(
    d_model=256,
    d_inner=512,       # defaults to 2 * d_model
    d_state=16,
    d_conv=4,
    dt_rank=None,      # auto: max(4, ceil(d_model / 16))
    use_cuda_scan=True,
    use_pre_norm=True, # LayerNorm before in_proj — recommended
)

x = torch.randn(8, 64, 256)  # (batch, seq_len, d_model)
y = block(x)                 # same shape as x
```text
ssm_out  = SSD( LayerNorm(x) )
attn_out = SlidingWindowAttn( LayerNorm(x) )
gate     = sigmoid( Linear( LayerNorm(x) ) )
mixed    = gate * ssm_out + (1 − gate) * attn_out
output   = out_proj( LayerNorm(mixed) )
```python

### Constructor parameters

| Parameter | Default | Description |
|---|---|---|
| `d_model` | required | Model dimension |
| `d_inner` | `2 * d_model` | SSM inner dimension |
| `d_state` | `16` | SSM state dimension per head |
| `d_conv` | `4` | Causal conv kernel size in the SSM branch |
| `dt_rank` | auto | Low-rank Δt projection size |
| `num_heads` | `8` | Query heads for attention; head count for SSD |
| `n_kv_heads` | `None` | KV heads for GQA. `None` → standard MHA. Must divide `num_heads`. |
| `window_size` | `128` | Sliding-window size for local causal attention |
| `attn_dropout` | `0.0` | Attention dropout during training |
| `use_gated_delta` | `False` | Add per-head sigmoid gate on Δt in the SSD branch |
| `rope_base` | `10_000` | RoPE frequency base |
| `max_seq_len` | `8192` | Pre-built RoPE cache length |
| `n_sink_tokens` | `0` | Number of leading tokens every position may attend to outside the local window |
| `qk_norm` | `False` | Apply per-head RMSNorm to Q/K before RoPE for attention-logit stability |
| `qk_norm_eps` | `1e-6` | Epsilon for Q/K RMSNorm |
| `attn_logit_softcap` | `None` | Optional tanh soft-cap for attention logits before softmax |
| `layer_scale_init` | `None` | Optional initial per-channel output scale before the outer residual add |

## Grouped Query Attention (GQA)

Set `n_kv_heads` to a divisor of `num_heads` to enable GQA. With `num_heads=8, n_kv_heads=2` the model uses 4× fewer KV parameters and KV cache entries compared to MHA, matching the Llama 3 / Mistral configuration:

```python
block = HybridMamba2Block(
    d_model=512,
    num_heads=16,
    n_kv_heads=4,   # 4 query heads share each KV head
    window_size=256,
)
```text

For use as a time-series backbone, replace the embedding + LM-head with your own projection layers and feed patch embeddings or raw-feature vectors in place of `input_ids`.

## Diagnostics

```python
from foreblocks.custom_mamba import run_default_diagnostics, benchmark_block

run_default_diagnostics()   # quick correctness checks for ops on current device

stats = benchmark_block(d_model=256, seq_len=512, batch=8)
print(stats)  # wall-clock and memory stats
