# foreblocks.nn.attention

Multi-backend attention: dense SDPA/flash/flex kernels, sparse variants
(sliding window, NSA, MoBA, ProbSparse), linear-attention backends (GLA,
DeltaNet, GatedDeltaNet, GatedDeltaNet2, Kimi/KDA, RDA), spectral attention
(Fourier/DWT/autocorrelation), and paged/static KV caching with optional
attention-matching compaction.

## Directory structure

```text
attention/
├── __init__.py             # Lazy public exports
├── config.py, enums.py     # Grouped configuration and typed choices
├── compat.py               # Legacy configuration conversion
├── layer.py                # AttentionLayer composition interface
├── multihead.py            # MultiAttention initialization and orchestration
├── features.py             # Feature initialization, score/output transforms, masks
├── cache/                  # KV providers, storage, selection, decode, compaction
│   └── runtime.py          # Private cache lifecycle and compaction coordination
├── execution/              # Backend capabilities and kernel dispatch
├── preparation/            # QKV projections, positions, masks, shared pipeline
│   └── position.py         # RoPE/SyPE transforms and the position extension point
├── variants/               # Dense, sparse, windowed, and spectral strategies
└── algorithms/
    ├── linear/             # Linear-attention backends and common execution
    └── spectral/           # Autocorrelation, Fourier, and wavelet algorithms
```

`multihead.py` owns construction and the public forward interface. Private
feature, position, and cache mixins group behavior by responsibility. Feature
initialization registers parameters directly on the owning `MultiAttention`,
without adding nested modules, so parameter names and checkpoint paths stay
unchanged. The constructor reads the grouped config directly rather than
copying its fields into an intermediate flat configuration.
`variants/` chooses an attention algorithm; `execution/` chooses its kernel;
`preparation/` handles tensor and mask preparation; `cache/` handles KV state.
Optional external adapters belong under `foreblocks.integrations`.

Package exports, built-in variant factories, and linear backend selection import
implementations on demand. Importing configs does not load `MultiAttention`,
and constructing standard attention does not load linear or spectral algorithms.
The same public import paths still resolve to the concrete implementation classes.
Backend capability discovery lives in `execution/backends.py`.

Transformer layers build these modules through
`foreblocks.nn.transformer.attention_backends`. Keep transformer stack and
layer orchestration in `transformer/`, and shared attention behavior here.

## Causal execution

Uncached forwards use a scalar zero position offset instead of creating cache
position tensors. Eligible causal SDPA calls use `is_causal=True` without a
dense additive mask. Explicit attention/padding masks retain their composition
path; cached offsets and custom score transforms retain the existing dispatch
rules. Cross-attention does not acquire a causal mask from this fast path.

Run `benchmarks/attention/bench_causal_sdpa.py` to compare eager and SDPA full
attention latency and peak temporary CUDA memory on the same model weights.
This measures attention forwards, not a full transformer training workload.

## Linear attention state and masks

Pass a mutable `layer_state={}` on the first streaming call and reuse it across
prefill chunks and decode steps. Preserve the complete returned dictionary when
one is returned: it includes short-convolution history as well as recurrent
memory. `ModernLinearAttention.reset_state(state)` clears the selected backend's
memory, convolution history, and RoPE position while preserving unrelated entries.
Cached RDA and GLA use absolute RoPE offsets across calls. State retains its
autograd history; detach its tensors explicitly when truncating backpropagation.

Linear backends accept empty masks and redundant causal masks (boolean blocked
entries, or additive zero/negative-infinity entries). Arbitrary pairwise masks
raise `ValueError`, since the recurrent algorithms cannot represent them.
RDA supports noncausal execution; the gated/delta backends remain causal scans.

All linear backends accept boolean `key_padding_mask` tensors shaped `[B,T]`
on the input device. In causal self-attention, `True` denotes an absent step:
it produces zero output and does not write, erase, decay, or advance recurrent
memory, short-convolution history, or RoPE positions. Left padding, right
padding, interior gaps, and entirely padded sequences follow the same rule.
Masked inputs are cleared before projections so padded NaNs cannot contaminate
valid outputs or gradients. After masked calls, RDA/GLA position counters can
be per-sequence tensors because each batch member may have consumed a different
number of valid tokens. Unmasked execution retains its existing fast paths;
masked short convolutions compact valid tokens per sequence before dispatch.

For noncausal RDA and the gated-delta cross-attention approximation, the mask
applies to encoder keys/values only; query positions are independent of that
memory mask.

RFF projections are registered buffers and move with the module's device and
dtype. Their random basis is saved in checkpoints. Kimi short convolution uses
one causal padding operation and one SiLU activation.

`benchmarks/attention/bench_linear_attention.py` reports module auto-dispatch
latency separately from kernel-only latency. Kernel comparisons use identical
inputs, numerical output/state checks, and Q/K/V gradient checks with
`--backward`. Its eager sequential reference is a correctness baseline, not an
optimized performance target. FLA is optional; unavailable kernels are reported
explicitly. The local recurrent Triton linear scan is dispatched automatically
only for single-token calls; longer scans can be benchmarked through the explicit
kernel API.

The FlexAttention backend builds a `BlockMask` and compiles the attention call.
It uses SDPA when training attention dropout is nonzero or head dimensions are
below 16, preserving semantics for operations/shapes unsupported by FlexAttention.

### Accelerated linear and MoE blocks

The recurrent linear Triton API accepts `initial_state=(key_sum, kv_sum)` and
`output_final_state=True`. It returns `(output, (next_key_sum, next_kv_sum))`;
the new state uses fp32 and does not mutate or alias the input state. This is an
inference API. RDA cached single-token calls select it automatically; training
and longer cached chunks keep the differentiable torch scan.

`fused_rmsnorm_sigmoid_gate` supports gradients for inputs, gates, and per-head
weights. Pass `gate_activation="sigmoid"` or `"silu"` for raw gate logits;
the default `"identity"` preserves the original preactivated-gate convention.
Gated Delta, GDN2, and Kimi use fused sigmoid gating in training and inference.

Linear-attention short convolutions share Mamba's differentiable Triton
convolution during training. Their inference kernel combines convolution,
SiLU, and raw-history updates without concatenating a history tensor. Q/K
normalization remains a separate operation. Set
`FOREBLOCKS_DISABLE_TRITON_SHORT_CONV=1` to force the torch reference.

`grouped_mm_varM(..., backend="persistent")` uses a fixed worker pool over
actual expert tiles, including for input gradients. `backend="legacy"` keeps
the rectangular launch; `"auto"` uses persistent scheduling for at least 16
experts and at most 128 packed rows per expert on average. Larger groups retain
the legacy launch because the measured persistent advantage was limited to
short groups. The explicit selection supports other hardware/workload tuning.

Run `PYTHONPATH=src python benchmarks/attention/bench_acceleration.py --output
results.json` for value-checked timings and peak temporary memory. It covers
sparse windows, norm/gate forward and backward, stateful decode, grouped GEMM,
and short-convolution updates. Compilation and autotuning are warmed first.

## Grouped KV heads and local windows

SyPE rotates queries and keys with their own head counts. KV projections and
caches retain `n_kv_heads`; head repetition happens only for attention execution.
Sliding-window attention supports both grouped-query and multi-query layouts.

Window masks mark blocked positions internally and are inverted for SDPA's
allowed-position convention. Manual query chunks cover the union of their local
windows, honor each sequence's absolute cache offset, and return full attention
weights when requested. Fully masked rows produce zero attention updates and
finite gradients in both execution paths.

On CUDA, the `auto` and `flex_attention` backends use direct block-sparse
predicates for sliding and dilated windows when dropout is inactive, head
dimensions are at least 16, and weights are not requested. This path does not
construct a dense window mask or repeat KV heads. Static predicates cache up
to eight block masks per implementation; offsets, user masks, and padding
rebuild the mask so streaming calls cannot reuse stale positions. Dropout,
small heads, CPU execution, and attention-weight requests retain the existing
paths. Dilated windows honor absolute cached query positions in both paths.

## Naming convention

This package shares `transformer/`'s conventions (see
`../transformer/README.md`) for the roles they have in common. Where this
package needed its own answer, it's recorded here — don't reintroduce the
inconsistency these fixed.

**`*Owner` Protocols** — every Protocol describing a collaborator object
handed to a strategy/mixin ends in `Owner`: `LazyAttentionOwner`,
`KernelDispatchOwner`, `AttentionOwner`, `ProjectionOwner`,
`_GatedDeltaOwner`. This applies whether the object is passed as a
constructor argument held by reference (`QKVProjector.context:
ProjectionOwner`) or mixed in as `self` (`_GatedDeltaOwner` for
`GatedDeltaExecutionMixin`) — same convention, same reasoning as
`transformer/`'s `DecoderOwner`. `MultiAttentionOwner` (`variants/base.py`)
is the sole concrete adapter implementing `AttentionOwner`; its name mirrors
the Protocol it implements rather than being called an "adapter" or
"context" — do the same for any future adapter of an `*Owner` Protocol.

**Backend class names always end in `Backend`**: `DeltaNetBackend`,
`GLABackend`, `RDABackend`, `GatedDeltaNetBackend`, `GatedDeltaNet2Backend`,
`KimiAttentionBackend`. There used to be three different conventions here
(bare model name, `*Backend`, `*Attention`) plus two aliases
(`GatedDeltaBackend = GatedDeltaNet`, `KimiBackend = KimiAttention`) bolted
on so the dispatch map in `wrapper.py` could pretend they were uniform. The
aliases are gone — `wrapper.py`'s `_BACKEND_MAP` now references the real
class names directly. If you add a 7th linear-attention backend, name its
class `<Name>Backend` and skip the alias step entirely.

**Variant implementations always end in `AttentionImpl`**:
`StandardAttentionImpl`, `SlidingWindowAttentionImpl`,
`DilatedSlidingWindowAttentionImpl`, `MoBAAttentionImpl`, `NSAAttentionImpl`,
`ProbSparseAttentionImpl`, `SoftpickAttentionImpl`, `SpectralAttentionImpl`.
This was already consistent except one outlier (`NSAImpl`, missing
"Attention") which is now `NSAAttentionImpl`.

**Verb prefixes for "construct a new instance"**: `build_*` (or a bare
`.build()` staticmethod) is the only verb for pure construction —
`NormWrapper.build`, `FeatureMapRegistry.build`,
`LayerAttentionBackendSpec.build`, `build_attention_mask`. Two verbs that
look like construction but aren't are deliberately different:
`PagedKVCache.ensure()` is get-or-create (idempotent lookup against
existing `layer_state`, not unconditional construction — keep calling it
`ensure`, not `build`), and `DensePagedStorage.allocate()` /
`LatentPagedStorage.allocate()` specifically reserve tensor memory —
`allocate` is the more precise term here than the generic `build`, so it
stays. Don't "fix" either of these to `build_*`; they're named for a reason.

## Known exceptions (not naming bugs — read before "fixing")

**`kimi.py` doesn't use `gated_common.py`'s shared classes.**
`GatedDeltaExecutionMixin`/`CausalDepthwiseConv`/`HeadRMSNorm` are shared by
`GatedDeltaNetBackend` and `GatedDeltaNet2Backend` (both were previously
wrapping them in pointless pass-only subclasses — removed; they now use the
shared classes directly). `KimiAttentionBackend` has its own
`_ShortConv1d`/`_HeadwiseRMSNorm` instead. This is intentional, not
leftover duplication: `_ShortConv1d` supports streaming-decode arguments
(`cache`, `output_final_state`) that `CausalDepthwiseConv` doesn't expose in
that form; packed `cu_seqlens` inputs are rejected. `_HeadwiseRMSNorm`'s weight-broadcast shape assumption differs
from `HeadRMSNorm`'s. Don't collapse these onto the shared classes without
first giving `CausalDepthwiseConv`/`HeadRMSNorm` a superset interface and
running a numerical-parity check against `kimi.py`'s current output — the
two aren't drop-in equivalent today.

**Compaction config lives in `AttentionCacheConfig`, not a separate
dataclass.** `AttentionMatchingCompactor` used to take its own
`AttentionMatchingConfig` with fields (`keep_ratio`, `trigger_len`, ...)
that duplicated `AttentionCacheConfig.matching_*` under different names —
three copies of the same five numbers by the time they reached the
compactor. `AttentionMatchingCompactor.__init__` now takes those five
values as keyword arguments directly; `MultiAttention` passes its already-
unpacked `self.attention_matching_*` attributes straight through. If you
add a new compaction knob, add it to `AttentionCacheConfig.matching_*` and
thread it the same way — don't reintroduce a parallel config object.

## Scope of the two attention interfaces

`MultiAttention` is the configured attention engine used by transformer layers.
`AttentionLayer` in `layer.py` is the separate public interface used by block
compositions and the root package. Its projection layout, method names, and
mask behavior differ; it is not an alias for `MultiAttention`. Keep its public
constructor and checkpoint structure separate until an explicit migration is
provided. Algorithm implementations remain grouped by mathematical family;
cache storage stays independent of transformer stack state and generation.
