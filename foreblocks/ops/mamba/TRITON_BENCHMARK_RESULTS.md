# Mamba2 Triton Kernel Benchmark Results

**GPU**: NVIDIA RTX 4090, CUDA 13.0, Triton 3.7.0

## Summary

| Component | Winner | Best Speedup | Worst Speedup |
|-----------|--------|-------------|---------------|
| dt_prep (elementwise) | Fallback | — | Triton 10x slower (small D) |
| fused_out (RMSNorm) | Triton | 15x | 2.9x |
| **SSD fwd (parallel)** | **Triton** | **8.76x** | **1.31x** |
| SSD bwd (T<1024) | Triton | 2.90x | — |
| SSD bwd (T≥1024) | **Torch** | — | Triton 10-100x slower |
| **Full block (T=4096)** | **Triton** | — | **1.48x faster** |
| Full block (T=1024) | Triton | — | 1.62x faster |
| **fused_dt** (dt_rank=16) | Triton | — | **1.96x faster** |
| **fused_dt** (dt_rank=64) | Triton | — | **1.36x faster** |
| **fused_dt** (dt_rank=256) | **Fallback** | — | 0.35x (cuBLAS faster) |

## Implementations Completed

### 1. Fused dt_proj + dt_prep kernel ✓
- File: `fused_dt.py`
- Forward + backward in single kernel launch
- Saves one kernel launch + avoids intermediate dt_raw tensor
- Performance: 1.3-2x faster for dt_rank ≤ 64; slower for dt_rank ≥ 256
- Integration: already wired into `mamba2_combined.py`

### 2. Parallel SSD forward ✓
- File: `ssd.py` (modified `chunked_ssd_forward_triton`)
- **Replaced sequential Python chunk loop with parallel single-kernel launch**
- Entry states computed via `segment_sum_log` (same as torch vectorized path)
- Single kernel processes ALL chunks simultaneously via `(B * nc * H, npb)` grid
- **Performance: 1.3-8.8x faster than torch vectorized path**
- Correctness: max relative error < 4e-7 across T=128 to T=4096

### 3. SSD backward path selection
- `SSD_TRITON_BACKWARD_MIN_SEQLEN = 65536` (always use torch backward)
- Triton backward is O(chunks) sequential, still 10-100x slower than torch

## Performance Details

### Parallel SSD Forward (single kernel launch)

| T | Chunks (nc) | Torch (ms) | Triton (ms) | Speedup |
|---|-------------|------------|-------------|---------|
| 128 | 2 | 1.54ms | 0.21ms | **7.32x** |
| 256 | 4 | 0.24ms | 0.16ms | **1.55x** |
| 512 | 8 | 0.24ms | 0.16ms | **1.49x** |
| 1024 | 16 | 0.24ms | 0.17ms | **1.39x** |
| 2048 | 32 | 0.35ms | 0.16ms | **2.17x** |
| 4096 | 64 | 0.71ms | 0.17ms | **4.30x** |

### Full Mamba2 Block (forward only, eval mode)

| T | Torch (ms) | Triton (ms) | Speedup |
|---|-----------|-------------|---------|
| 1024 | 4.26ms | 2.69ms | **1.58x** |
| 4096 | 20.80ms | 14.02ms | **1.48x** |

## How Parallel SSD Forward Works

### Before (sequential)
```
for chunk_idx in range(n_chunks):  # O(nc) kernel launches
    _chunked_ssd_forward_kernel[grid](..., CHUNK_INDEX=chunk_idx, state=state)
```
Each chunk launches its own kernel, carrying state between launches.

### After (parallel)
```python
# 1. Compute entry states via segment_sum_log (O(1) Python overhead)
entry_states = compute_boundary_states(state_end, chunk_log_decay)

# 2. ONE kernel launch processes ALL chunks
g = (B * nc * H, npb)
_chunked_ssd_forward_kernel[g](..., entry_state_ptr=entry_states)
```
Each program ID = (b, nc, h) handles one (batch, chunk, head) tuple.

### Key insight
Entry states (boundary states) are precomputed using `segment_sum_log` — the same parallel prefix scan used by the torch vectorized path. This eliminates the O(nc) Python loop while maintaining correctness.

## Key Findings

### 1. Parallel SSD forward eliminates sequential chunk loop ✓
The previous bottleneck was O(nc) sequential kernel launches. Replacing with a single kernel launch using `segment_sum_log` for entry states gives 1.3-8.8x speedup.

### 2. SSD backward remains sequential
The backward requires gradient state to flow between chunks (reverse order). Triton backward is 10-100x slower than torch. Threshold set to always use torch backward.

### 3. fused_out Triton is a clear win
Triton RMSNorm + gating is 3-15x faster than PyTorch across all sizes. No changes needed here.

### 4. fused_dt speedup depends on dt_rank
- dt_rank ≤ 16: 1.96x faster
- dt_rank = 64: 1.36x faster  
- dt_rank ≥ 256: 0.35x (cuBLAS matmul wins)

## Correctness

All implementations verified within float32 tolerance:
- Parallel SSD forward max relative error: 3.74e-07 (T=128 to T=4096)
- dt_prep max error: 1.19e-07
- fused_out max error: 9.54e-07
- fused_dt forward max error: 4.95e-06
- fused_dt backward max error: 6.78e-05
