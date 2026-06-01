# custom_att Triton attention scaffold

This package contains a Triton-first exact online-softmax attention implementation.
It is designed as a clean iteration surface for modern attention kernel tuning without shipping a custom C++/CUDA extension.

## What is implemented

### Core Kernels

- Triton forward and backward kernels for `[B,H,N,D]` tensors.
- Online log-sum-exp softmax, so it does **not** materialize the `N x N` attention matrix.
- Causal and non-causal modes.
- `float16` and `bfloat16` Triton fast paths for head dimensions `16, 32, 64, 96, 128, 256`.
- Deterministic Ada/RTX 4090-aware tile selection, avoiding broad cold-start
  autotuning while keeping tensor-core-friendly shapes.
- Hybrid backward delta handling: launch-bound shapes fuse `sum(out * d_out)`
  into the dQ/dK/dV kernels, while larger shapes keep the precomputed-delta
  path to avoid redundant row reductions.
- FlashAttention-2-style combined backward launch for aligned `D=64/128`,
  `N>=2048` Ada workloads, with shape-specific dQ/dK-dV tile choices.

### FA2 Optimizations

- **Persistent scheduling (forward)**: Grid-stride loop for CTA when tile count
  would leave SMs idle, maximizing occupancy on Ada GPUs.
- **Split-K dK/dV backward**: Parallelizes dK/dV accumulation across Q-tile splits
  for unbalanced causal workloads (N>=1024 on Ada).
- **Persistent dK/dV backward**: Grid-stride loop for K-tiles when tile count < 2x SMs,
  filling idle SMs on long-context sequences (N>=2048).
- **Persistent dQ backward**: Grid-stride loop for Q-tiles on long non-causal Ada workloads.
- **Fused softmax scaling (prescale_k)**: Scales K by log2e in the kernel to avoid
  per-tile multiplication, reducing shared memory pressure.
- **Fused delta computation**: Computes `D = sum(O * dO)` inside the dQ kernel for
  launch-bound shapes, removing a launch and global memory traffic.

### Advanced Features

- **FA2-style dropout**: Dropout applied to attention probabilities during training,
  with 1/(1-p) scaling for unbiased gradients.
- **KV-cache decode**: Optimized single-token generation with KV cache via
  PyTorch SDPA backend (already the fastest on modern GPUs).
- **Fused attention + RMSNorm**: `FlashAttnRMSNorm` module implementing the common
  LLM pattern of normalize -> attend -> norm -> add residual.
- **PyTorch `autograd.Function` wrapper** with correct torch/SDPA fallback paths
  for unsupported shapes and backends.

## What is not implemented yet

This is not a production Hopper FlashAttention-3 kernel yet. True FA3 requires:

1. WGMMA tensor-core matrix multiplications.
2. TMA global-to-shared asynchronous loading.
3. Warp-specialized producer/consumer scheduling.
4. Circular shared-memory K/V staging.
5. GEMM-softmax overlap / ping-pong scheduling.
6. FP8 e4m3 path with block quantization and incoherent processing.
7. A broader set of fused kernels than the current Triton implementation.

Those are large enough that the correct next implementation step is to add them incrementally on top of this exact baseline.

## Install

```bash
python -m pip install -v -e .
```

## Test

```bash
python test_custom_att.py
```

## Use

### Basic attention

```python
import torch
from custom_att import flash_attn_func

q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = flash_attn_func(q, k, v, causal=True)
```

### Attention with dropout

```python
from custom_att import flash_attn_dropout_func

o = flash_attn_dropout_func(q, k, v, causal=True, dropout_p=0.1)
```

### KV-cache decode (single-token generation)

```python
from custom_att import flash_attn_decode

q_dec = torch.randn(2, 8, 1, 64, device='cuda', dtype=torch.float16)  # [B, H, 1, D]
k_cache = torch.randn(16, 2048, 64, device='cuda', dtype=torch.float16)  # [B*H, max_seqlen, D]
v_cache = torch.randn(16, 2048, 64, device='cuda', dtype=torch.float16)
seqlens = torch.full((16,), 2048, device='cuda')  # [B*H]

out, lse = flash_attn_decode(q_dec, k_cache, v_cache, seqlens)
```

### Fused attention + RMSNorm module

```python
from custom_att import FlashAttnRMSNorm

module = FlashAttnRMSNorm(1024, n_heads=8).half().cuda()
x = torch.randn(2, 2048, 1024, device='cuda', dtype=torch.float16)
y = module(x)  # normalize -> attend -> norm -> residual
```

## Roadmap to a real Hopper FA3 kernel

1. Replace one-warp-per-row dot products by block-tiled QK and PV fragments.
2. Add shared-memory K/V tiles and double-buffering.
3. Introduce WGMMA fragments for QK and PV.
4. Split CTA warps into producer and consumer warpgroups.
5. Add TMA loads and barriers.
6. Add two-stage QK/softmax/PV pipelining.
7. Add persistent scheduling and load balancing.
8. Add FP8 quantized Q/K/V path.

The paper's core FA3 ideas are producer-consumer asynchrony, WGMMA/TMA overlap, softmax hidden under asynchronous GEMMs, and FP8 block quantization/incoherent processing.

## Roadmap for RTX 4090 / Ada

1. ~~Add a split-K / persistent backward path for long-context training.~~ ✅ Done
2. ~~Explore a persistent scheduler that maps multiple row/column tiles per CTA~~ ✅ Done
3. Keep tuning forward tiles for causal `D=128`, where PyTorch SDPA still wins
   on RTX 4090.
4. Add a true Triton decode kernel with block-tiled PV (pending Triton API improvements).
5. Add Grouped Query Attention (GQA) / Multi-Query Attention (MQA) support.
6. Add fused attention + dropout + RMSNorm end-to-end module.
