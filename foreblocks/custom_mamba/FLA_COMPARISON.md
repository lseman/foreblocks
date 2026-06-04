# Comparison: custom_mamba vs fla (flash-linear-attention) Mamba2

## Source

- **ours**: `foreblocks/custom_mamba/` — our selective state-space implementation
- **fla**: `flash-linear-attention/fla/layers/mamba2.py` + `fla/ops/` — the reference Mamba2 from fla-org

The fla Mamba2 implementation is the production-grade reference used in open-weight models (Mamba-3, etc.). Below are the key architectural differences and opportunities for improvement.

---

## 1. Architecture & Parameter Layout

### fla Mamba2 parameter layout

```
in_proj (hidden → intermediate_size + conv_dim + num_heads)
├── d_mlp (twice, unused in Mamba2 — residual from Mamba1)
├── gate (intermediate_size)
├── hidden_states_B_C (intermediate_size + 2 * n_groups * ssm_state_size)
│   ├── hidden_states (intermediate_size)  ← the "u" input
│   ├── B (n_groups * ssm_state_size)
│   └── C (n_groups * ssm_state_size)
└── dt (num_heads,)
```

### Current custom_mamba Mamba2 parameter layout

```
in_proj (d_model → d_inner + conv_dim + dt_rank)
├── z (d_inner)  ← gate for fused_out
├── conv_input (d_inner + 2*n_groups*d_state)
│   ├── u (d_inner)
│   ├── Bflat (n_groups * d_state)
│   └── Cflat (n_groups * d_state)
└── dt_hidden (dt_rank)
dt_proj (dt_rank → num_heads)
dt_bias (num_heads)
A_log (num_heads)
Dskip (num_heads, head_dim)
```

### Key differences

| Aspect | fla Mamba2 | custom_mamba |
|--------|-----------|--------------|
| **B, C shape** | `(n_groups, d_state)` — shared per group | `(n_groups, d_state)` expanded to heads |
| **D parameter** | Per-head or per-head_dim | Per-head_dim (`Dskip`) |
| **dt init** | Random per head, inverse_softplus stored | Random per head, inverse_softplus stored |
| **A parameter** | Per-head (scalar) | Per-head (scalar) |
| **Gating** | RMSNormGated (rmsnorm then silu(gate)) | fused_out (silu(z) * y + residual) |
| **Conv** | causal_conv1d CUDA package | Triton causal_conv1d |
| **SSM scan** | Chunked (mamba_chunk_scan_combined) | Chunked SSD with parallel prefix |
| **Grouping** | Explicit `n_groups` | Explicit `n_groups` |

---

## 2. Improvement Opportunities

### A. Structured (diagonal) A matrix — **IMPLEMENTED**

**fla approach:**
```python
A = -torch.exp(self.A_log.float())  # [num_heads] — scalar per head
A = A[:, None, ...][:, :, None].expand(-1, self.head_dim, self.ssm_state_size)
# A is [num_heads, head_dim, d_state] with A[h] on the diagonal
# This is the "diagonal A" from Mamba2 paper
```

**Previous custom_mamba approach:**
```python
A = -torch.exp(self.A_log)  # [d_inner, d_state] — dense per-feature
```

**Impact:** The diagonal A is the Mamba2 hallmark. It means A[h, j, k] = A[h] if j == k else 0. This allows the SSM to be factored as head × feature × state, reducing parameters and computation. Our dense A is more expressive but slower and more parameter-heavy.

**Current status:** `Mamba2Block` uses `A_log` as `[num_heads]` and feeds the
diagonal-A chunked SSD scan directly.

---

### B. Chunked SSM scan — **IMPLEMENTED / OPTIMIZE NEXT**

**fla approach:** Uses `mamba_chunk_scan_combined` (CUDA) or naive chunked scan:
```python
# Pad to chunk_size boundary
pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
hidden_states, A, B, C = [reshape_into_chunks(t, pad_size, chunk_size) ...]
# Per-chunk: compute intra-chunk diagonal + inter-chunk off-diagonal
L = torch.exp(segment_sum(A))  # L[i,j] = exp(sum(A[i:j]))
G = C * B  # "attention weights"
Y_diag = (G[..., None] * hidden_states).sum(dim=3)  # intra-chunk
decay_states = torch.exp(A_cumsum[..., -1:] - A_cumsum)  # inter-chunk decay
states = (B_decay * hidden_states).sum(dim=2)  # accumulate states
# Inter-chunk recurrence
decay_chunk = torch.exp(segment_sum(pad(A_cumsum[..., -1], (1, 0))))
new_states = (decay_chunk * states).sum(dim=1)
Y_off = (C_times_states.sum(-1) * state_decay_out)  # state → output
y = Y_diag + Y_off + D * hidden_states
```

**Previous custom_mamba approach:** Sequential scan:
```python
for t in range(T):
    state = abar * state + dt_t * bpar * u_t
    y_t = cpar * state + Dskip * u_t
```

**Current custom_mamba approach:** `Mamba2Block` uses diagonal-A `chunked_ssd_forward` with:
- intra-chunk SSD factorization for local outputs and chunk state summaries
- a full parallel prefix scan over chunk boundary states using FLA-style log-space `segment_sum`
- an optional Triton recurrent per-chunk kernel (`chunked_ssd_forward_triton`) for CUDA experiments

`Mamba2Block` now defaults training to the full parallel-prefix path and reserves
the per-chunk recurrent Triton SSD path for eval/inference. This matches the
current trade-off: training benefits more from parallel chunk-boundary
propagation, while inference benefits from avoiding the larger prefix tensors.

**Impact:** Chunked scan enables:
1. **Parallelism across chunks** — different chunks can be processed concurrently
2. **Lower memory bandwidth** — state passes between chunks are materialized once
3. **Better GPU utilization** — block-level parallelism within each chunk

**Remaining action:** fuse the full-prefix path in Triton/CUDA and add a fused backward. The algorithmic chunked scan is now present; the next speed step is kernel fusion.

---

### C. RMSNormGated (fused norm + gate) — **MEDIUM PRIORITY**

**fla approach:** Uses `RMSNormGated` with Triton kernel:
```python
# Forward: norm(x) * silu(gate)  (norm_before_gate=True)
# or: norm(x * silu(gate))  (norm_before_gate=False)
# Fully fused in a single Triton kernel with:
# - Per-group normalization
# - Optional bias
# - Optional upcast to float32
```

**Our approach:** Two-step process:
```python
# fused_out: x = y * silu(z) + residual; then RMSNorm
# This mixes gate + residual + norm
```

**Impact:** A fused `RMSNormGated` kernel would:
1. Avoid materializing the intermediate `x = y * silu(z) + residual` tensor
2. Allow the RMSNorm mean/sqrt to be computed on `y * silu(z) + residual` directly
3. Reduce memory traffic by ~1× (one kernel instead of two)

**Action:** Create `RMSNormGated` module with Triton kernel supporting `norm_before_gate` flag.

---

### D. D parameter as per-head skip — **IMPLEMENTED**

**fla approach:**
```python
D = torch.ones(num_heads)  # per-head scalar
D = D[:, None].expand(-1, head_dim)  # [num_heads, head_dim]
y = y + hidden_states * D  # [bs, T, H, P] + [bs, T, H, P]
```

**Previous custom_mamba approach:**
```python
Dskip = torch.ones(d_inner)  # per-feature
y = C * state + Dskip * u  # [bs, T, D]
```

**Impact:** Per-head D gives more expressivity. Since fla treats D as per-head (or per-head_dim), and A is also per-head, this is the "natural" Mamba2 parameterization.

**Current status:** `Mamba2Block` stores `Dskip` as `[num_heads, head_dim]`.

---

### E. dt_bias initialization — **IMPLEMENTED**

**fla approach:**
```python
dt = torch.exp(torch.rand(num_heads) * (log(dt_max) - log(dt_min)) + log(dt_min))
dt = torch.clamp(dt, min=dt_init_floor)
inv_dt = dt + torch.log(-torch.expm1(-dt))  # inverse softplus
self.dt_bias = nn.Parameter(inv_dt)
```

**Previous custom_mamba approach:**
```python
dt = torch.rand(d_inner) * (log(dt_max) - log(dt_min)) + log(dt_min)
self.dt_bias.copy_(inverse_softplus(dt.exp()))
```

**Impact:** fla uses `torch.rand` (uniform in log space) which gives a wider range of initial dt values. Our approach is similar but uses `d_inner` instead of `num_heads` for the shape.

**Current status:** `Mamba2Block` initializes `dt_bias` per head in log space.

---

### F. Grouped B/C (n_groups) — **IMPLEMENTED**

**fla approach:** B and C are projected to `(n_groups, ssm_state_size)`, then expanded to `(num_heads, ssm_state_size)` via repetition. This means `num_heads / n_groups` heads share the same B and C matrices.

**Previous custom_mamba approach:** B and C were fully projected to `(d_inner, d_state)` — no sharing.

**Impact:** Grouped B/C reduces parameters significantly:
- fla with `n_groups=1`: B, C = `(num_heads, ssm_state_size)` each
- fla with `n_groups=8`: B, C = `(8, ssm_state_size)` each — 8× fewer params

**Current status:** `Mamba2Block` projects grouped B/C and expands groups across heads.

---

### G. dt_limit (dt clamping) — **LOW PRIORITY**

**fla approach:**
```python
dt_limit = (0.0, float("inf"))  # or custom range
dt = torch.nn.functional.softplus(dt + dt_bias)
dt = torch.clamp(dt, dt_limit[0], dt_limit[1])
```

**Our approach:**
```python
dt = dt_prep(dt_raw, dt_bias, dt_min=dt_min, dt_max=dt_max)
# dt_prep applies softplus + clamp in Triton kernel
```

**Impact:** Our `dt_prep` kernel already does this correctly. No action needed.

---

### H. Causal Conv1d backend — **IMPROVED / LOW PRIORITY**

**fla approach:** Uses `causal_conv1d` from the causal-conv1d package (CUDA kernel) with a Triton fallback. Has explicit backend selection via env var `FLA_CONV_BACKEND`.

**Our approach:** Uses our own Triton causal_conv1d kernels for forward and
backward, with a PyTorch reference fallback.

**Impact:** The causal-conv1d CUDA package is still heavily optimized, but our
Triton path no longer falls back to autograd-over-reference for backward.

**Remaining action:** Consider adding the causal-conv1d CUDA package as an
optional dependency/backend for users who want its production kernel.

---

## 3. Summary: Recommended Priority Order

| # | Improvement | Effort | Expected Impact |
|---|-----------|--------|----------------|
| 1 | **Diagonal A matrix** | Done | Mamba2 correctness, fewer params |
| 2 | **Fused chunked SSM kernels** | High | 2-4× throughput, enables parallelism |
| 3 | **RMSNormGated fused kernel** | Medium | 15-20% layer speedup |
| 4 | **Grouped B/C (n_groups)** | Done | 2-8× fewer B/C params |
| 5 | **Per-head D parameter** | Done | Mamba2 correctness |
| 6 | **dt_bias init shape** | Done | Mamba2 alignment |
| 7 | **causal-conv1d CUDA fallback** | Optional | Potential speedup |

---

## 4. Module Mapping: fla → custom_mamba

| fla module | custom_mamba equivalent |
|-----------|----------------------|
| `fla/layers/mamba2.py::Mamba2` | `custom_mamba/blocks/mamba2.py::Mamba2Block` |
| `fla/ops/delta_rule/` | `custom_mamba/ops/ssd.py` |
| `fla/ops/common/fused_chunk.py` | *(not implemented)* |
| `fla/ops/common/fused_recurrent.py` | *(not implemented)* |
| `fla/ops/gated_delta_rule/` | *(not active; Mamba2Block uses standard SSD)* |
| `fla/modules/layernorm_gated.py` | `custom_mamba/blocks/norms.py::RMSNormWeightOnly` + `ops/triton_ops.py::fused_out` |
| `fla/modules/convolution.py` | `custom_mamba/ops/causal_conv1d.py` |
| `fla/modules/activations.py` | `torch.nn.functional.silu` |
| `fla/ops/common/gate.py` | *(not implemented)* |
| `fla/ops/utils/` | `custom_mamba/blocks/utils.py` |

---

## 5. Architecture Comparison Diagram

### fla Mamba2 (production)
```
x (B, T, hidden)
  │
  ├─ in_proj → [d_mlp, d_mlp, gate, conv_input, dt]
  │                │              │        │           │
  │           (unused)        (B,T,D)   (B,T,H)     (B,T,H)
  │                │              │        │           │
  │                │         causal_conv1d   │           │
  │                │              │        │           │
  │                │         [gate, B, C]   │           │
  │                │              │        │           │
  │                │         RMSNormGated   │           │
  │                │              │        │           │
  │                │         ┌────┴────────┘           │
  │                │         │                          │
  │         chunked_ssd_scan │                          │
  │         [u, dt, A(h), B(g), C(g), D(h)]            │
  │                │                          │
  │                └────────── y (B,T,D) ─────┘
  │                          │
  │                      out_proj
  │                          │
  y (B, T, hidden)
```

### custom_mamba (current)
```
x (B, T, d_model)
  │
  ├─ in_proj → [z, u, dt_hidden, B, C]
  │              │   │      │        │   │
  │           (B,T,D)  │    (B,T,R)  │   │
  │                    │             │   │
  │              conv1d (Triton)     │   │
  │                    │             │   │
  │              chunked_ssd_forward │   │
  │         [u, dt, A(f), B(f), C(f)]   │
  │                    │             │   │
  │                    └──── y (B,T,D) │
  │                                  │
  │                  fused_out(y, z, residual, norm)
  │                                  │
  │                              out_proj
  │                                  │
  y (B, T, d_model)
```

Key difference: fla uses **chunked SSD with diagonal A** (Mamba2 paper), while ours uses **sequential selective scan with dense A** (closer to original Mamba).
