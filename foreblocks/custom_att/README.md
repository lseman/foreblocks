# custom_att Triton attention scaffold

This package contains a Triton-first exact online-softmax attention implementation.
It is designed as a clean iteration surface for modern attention kernel tuning without shipping a custom C++/CUDA extension.

## What is implemented

- Triton forward and backward kernels for `[B,H,N,D]` tensors.
- Online log-sum-exp softmax, so it does **not** materialize the `N x N` attention matrix.
- Causal and non-causal modes.
- `float16` and `bfloat16` Triton fast paths for head dimensions `32, 64, 128`.
- Deterministic Ada/RTX 4090-aware tile selection, avoiding broad cold-start
  autotuning while keeping tensor-core-friendly shapes.
- Hybrid backward delta handling: launch-bound shapes fuse `sum(out * d_out)`
  into the dQ/dK/dV kernels, while larger shapes keep the precomputed-delta
  path to avoid redundant row reductions.
- FlashAttention-2-style combined backward launch for aligned `D=64/128`,
  `N>=1024` Ada workloads, with shape-specific dQ/dK-dV tile choices.
- PyTorch `autograd.Function` wrapper with correct torch fallback paths for unsupported shapes.

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

On RTX 4090 / Ada (`sm_89`), WGMMA and TMA are not available. The practical
performance roadmap for this GPU is FlashAttention-2 style: better tile
selection, persistent scheduling, lower memory traffic, and specialized causal
paths for the shapes that dominate training throughput.

## Install

```bash
python -m pip install -v -e .
```

## Test

```bash
python test_custom_att.py
```

## Use

```python
import torch
from custom_att import flash_attn_func

q = torch.randn(2, 8, 1024, 64, device='cuda', dtype=torch.float16)
k = torch.randn_like(q)
v = torch.randn_like(q)
o = flash_attn_func(q, k, v, causal=True)
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

1. Add a split-K / persistent backward path for long-context training.
   The current hybrid fused-delta path reduces launch/global-memory overhead
   where it wins, and aligned `D=64/128` long-context paths use a combined
   dK/dV+dQ launch. The dQ path also has a conservative grid-stride persistent
   variant for long `D=128` non-causal Ada workloads.
2. Explore a persistent scheduler that maps multiple row/column tiles per CTA
   on Ada without introducing atomics on dK/dV.
3. Keep tuning forward tiles for causal `D=128`, where PyTorch SDPA still wins
   on RTX 4090.
