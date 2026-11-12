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
├── features.py             # Private score/output transforms and feature masks
├── cache/                  # KV providers, storage, selection, decode, compaction
├── execution/              # Backend capabilities and kernel dispatch
├── preparation/            # QKV projections, positions, masks, shared pipeline
├── variants/               # Dense, sparse, windowed, and spectral strategies
└── algorithms/
    ├── linear/             # Linear-attention backends and common execution
    └── spectral/           # Autocorrelation, Fourier, and wavelet algorithms
```

`multihead.py` owns initialization, parameter registration, and the public
forward interface. The private feature mixin registers no additional modules,
so moving its methods does not alter parameter names or checkpoint paths.
`variants/` chooses an attention algorithm; `execution/` chooses its kernel;
`preparation/` handles tensor and mask preparation; `cache/` handles KV state.
Optional external adapters belong under `foreblocks.integrations`.

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
(`cache`, `output_final_state`, `cu_seqlens`) that `CausalDepthwiseConv`
doesn't, and `_HeadwiseRMSNorm`'s weight-broadcast shape assumption differs
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
