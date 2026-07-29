# foreblocks.modules.attention

Multi-backend attention: dense SDPA/flash/flex kernels, sparse variants
(sliding window, NSA, MoBA, ProbSparse), linear-attention backends (GLA,
DeltaNet, GatedDeltaNet, GatedDeltaNet2, Kimi/KDA, RDA), spectral attention
(Fourier/DWT/autocorrelation), and paged/static KV caching with optional
attention-matching compaction.

## Directory structure

```
attention/
├── __init__.py           # Stable public facade — deliberately small, see its own docstring
├── config.py               # AttentionConfig + 5 grouped sub-configs (shape/cache/position/variant/features)
├── enums.py                  # Closed StrEnum choice sets (PositionEncoding, QKNorm, ...)
├── compat.py                   # Legacy flat-kwargs -> AttentionConfig shim
├── multi_att.py                  # MultiAttention: the top-level nn.Module wiring everything below together
├── cache/                  # KV storage and retrieval
│   ├── base.py                # KVCacheProtocol + state-dict (de)serialization helpers
│   ├── kv.py                    # KVProvider ABC + Dense/Static/Paged implementations
│   ├── paged.py                   # PagedKVCache — vLLM-style block-pool cache
│   ├── storage.py                   # Dense/Latent block allocation strategies
│   ├── selection.py                   # AttentionCacheSelector: picks a KVProvider from config
│   ├── decode_stream.py                 # Streaming online-softmax decode over a PagedKVCache
│   └── compaction.py                      # AttentionMatchingCompactor: KV compaction policy
├── execution/               # Kernel selection and dispatch
│   ├── backends.py             # AttentionBackendRegistry: sdpa/flash/flex/eager capability specs
│   └── dispatch.py               # AttentionKernelDispatcher: runtime kernel choice
├── preparation/              # Input-side transforms, shared by all variants
│   ├── projections.py           # QKVProjector: model-space -> per-head Q/K/V
│   ├── position.py                # PositionEncodingApplier: named RoPE/ALiBi/SyPE transform pipeline
│   ├── masking.py                   # AttentionMaskProcessor + mask-building functions
│   └── pipeline.py                    # QKVPipeline: composes the three steps above
├── variants/                 # Pluggable AttentionImpl strategies (the "how do we score/weight" layer)
│   ├── base.py                  # AttentionImpl / AttentionOwner Protocols + MultiAttentionOwner adapter
│   ├── registry.py                # AttentionVariantRegistry: name -> AttentionImpl factory
│   ├── standard.py                  # Dense SDPA + GQA + production KV-cache paths
│   ├── sliding_window.py, dilated_sliding_window.py
│   ├── prob_sparse.py, moba.py, nsa.py, softpick.py
│   └── spectral.py                  # Dispatches into implementations/ (Fourier/DWT/autocorrelation)
└── implementations/          # Concrete algorithms too heavy/optional to import eagerly
    ├── autocor_att.py, dwt_att.py, frequency_att.py    # Spectral attention (Autoformer/DWT/FEDformer)
    └── linear_att/                # O(L·d²) recurrent-state backends, unified by ModernLinearAttention
        ├── base.py                  # RoPEMixin, FeatureMapRegistry — shared across backends
        ├── gated_common.py             # CausalDepthwiseConv, HeadRMSNorm, GatedDeltaExecutionMixin
        │                                — shared by GatedDeltaNetBackend and GatedDeltaNet2Backend
        ├── deltanet.py, gla.py, rda.py     # DeltaNetBackend, GLABackend, RDABackend
        ├── gated_delta.py, gated_deltanet2.py  # GatedDeltaNetBackend, GatedDeltaNet2Backend
        ├── kimi.py                          # KimiAttentionBackend (KDA) — intentionally NOT built on
        │                                      gated_common.py, see "Known exceptions" below
        └── wrapper.py                          # ModernLinearAttention: string-dispatch over all 6
```

`cache/`, `execution/`, `preparation/`, and `variants/` depend on nothing
above them in this tree except each other and `config.py`/`enums.py`.
`implementations/` depends on nothing else in this package (each backend
file is self-contained). `multi_att.py` is the only file that imports across
all five subpackages — it is the composition root, not a shared dependency.
There is no PEP 562 lazy-facade cycle inside `attention/` itself (unlike
`transformer/`, which needs one internally); the only cross-package cycle is
`attention/variants/softpick.py`'s deferred import reaching into
`foreblocks.models.transformer.third_party`, and `transformer/`'s own lazy
facades handle the transformer↔attention side of that cycle. Don't add a
new eager import from `attention/` back into `transformer/` — follow
`softpick.py`'s function-local deferred-import pattern if you need one.

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
