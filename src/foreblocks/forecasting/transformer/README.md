# foreblocks.forecasting.transformer

Modular transformer encoder/decoder stack: pluggable attention backends
(standard, linear, GLA, DeltaNet, GatedDeltaNet, SyPE, ...), GateSkip residual
gating, manifold hyper-connections (mHC), attention-residual accumulation,
Mixture-of-Depths routing, patch tokenization, and incremental KV-cache
decoding (greedy / beam / speculative).

## Directory structure

```
transformer/
├── config.py            # TransformerConfig, AttentionMode, ResidualConfig, CacheConfig, GenerationConfig
├── generation.py        # Re-export submodule (for backward compat)
├── tuner.py             # TransformerTuner: signal-analysis-driven auto-config
├── tuner_utils.py       # Signal-analysis utility functions extracted from tuner.py
├── core/                 # Layer + model classes (the public nn.Module surface)
│   ├── base.py            # BaseTransformerLayer, BaseTransformer (shared ABC)
│   ├── attention_backends.py  # LazyAttentionBackendMixin + backend registry
│   ├── construction.py    # build_positional_encoder, build_layer_modules
│   ├── encoder.py          # TransformerEncoderLayer, TransformerEncoder
│   ├── decoder.py          # TransformerDecoderLayer, TransformerDecoder
│   └── mixing.py           # sequence-then-variate attention for [B,V,N,D]
├── features/              # Standalone nn.Module building blocks, no cross-imports between them
│   ├── mhc.py               # MHCHyperConnection + stream init/collapse/norm helpers
│   ├── residuals.py          # AttentionResidual, BlockAttentionResidual
│   ├── patching.py            # PatchTokenizer / PatchDetokenizer (PatchTST-style)
│   ├── sype.py                 # AdaptiveWarp, SyPERotator (symplectic positional encoding)
│   └── fusions.py                # Fused dropout+residual(+norm) kernels
└── runtime/                # Non-nn.Module execution plumbing consumed by core/
    ├── execution.py          # *Strategy / *Mixin objects that run a sublayer block
    ├── forward.py              # prepare_*/execute_*/build_* — one encoder/decoder forward pass
    ├── contiguous.py           # masked-horizon preparation, projection, and quantile loss
    ├── decoding.py              # GenerationEngine, beam_search, speculative_decode
    ├── routing.py                # Mixture-of-Depths gather/scatter helpers
    ├── cache.py                   # DecoderCacheManager
    ├── contracts.py                # DecoderOwner (stable Protocol other runtime modules depend on)
    ├── state.py                     # DecoderState / DecoderLayerState / AttentionCacheState
    ├── residual_state.py             # AttentionResidualState + accumulation helpers
    └── outputs.py                     # TransformerEncoderOutput / DecoderOutput / GenerationOutput
```

`core/` depends on `features/` and `runtime/`. `runtime/` depends on
`features/` but never on `core/`. `features/` depends on neither — each file
there is a self-contained `nn.Module`. This one-way dependency is why
`core/base.py`, `core/__init__.py`, `features/__init__.py`, and
`runtime/__init__.py` all use a `TYPE_CHECKING` + PEP 562 `__getattr__` lazy
re-export: `core/base.py` and `core/{encoder,decoder}.py` import each other
(the base class needs to type-hint the concrete stack classes; the stack
classes subclass the base), and `foreblocks.attention` imports back
into `transformer/features` and `transformer/core`, forming a genuine import
cycle. Don't eagerly import across those seams at module scope — follow the
existing lazy-facade pattern in the `__init__.py` you're touching.

## Multivariate mixing

Set `TransformerConfig.use_variate_attention=True` to preserve input channels
as `[B, V, N, D]` inside the encoder. Each layer attends over the `N` sequence
positions independently per variate, then over the `V` variates independently
per sequence position. The default `variate_fuse="linear"` restores the usual
`[B, N, D]` encoder output; use `"mean"` or `"none"` for masked averaging or
an unfused `[B, V, N, D]` output. Variate attention is always non-causal.

```python
from foreblocks.forecasting.transformer import TransformerConfig
from foreblocks.forecasting.transformer.core.encoder import TransformerEncoder

config = TransformerConfig(
    input_size=num_variates,
    use_variate_attention=True,
)
model = TransformerEncoder(config)
encoded = model(values)
output = encoded.last_hidden_state  # values [B,N,V] -> output [B,N,D]
```

`MixingTransformer` remains available as the lower-level embedding block for
callers that already own `[B, V, N, D]` tensors.

### Single-pass contiguous decoding

Enable `use_contiguous_patch_decoding=True` with non-overlapping patches. The
forecast path appends learned missing-value tokens for the complete horizon and
runs the bidirectional encoder/mixing stack once. Targets and past-only
covariates are hidden over the horizon; past-future covariates remain visible.

```python
config = TransformerConfig(
    input_size=num_targets + num_past_only + num_known_future,
    use_variate_attention=True,
    use_contiguous_patch_decoding=True,
    patch_encoder=True,
    patch_len=32,
    patch_stride=32,
)
model = TransformerEncoder(config)
quantile_forecast = model.decode_contiguous(
    target,                       # [B, context, targets]
    horizon=128,
    past_only_covariates=past,    # [B, context, past-only]
    past_future_covariates=known, # [B, context + horizon, known-future]
)  # [B, horizon, targets, 9]
```

`forecast_contiguous` is the differentiable training entry point;
`contiguous_quantile_loss` computes pinball loss for the configured quantiles.

## Naming convention

These conventions are enforced package-wide. If you're adding a new
`prepare_*`/`*Owner`/`*Strategy` symbol, match the existing one instead of
inventing a fourth variant.

**`Prepared*` dataclasses** — the return type of a public `prepare_*`
function is always named `Prepared<Thing>` (`PreparedEncoderInput`,
`PreparedDecoderState`). If a helper's result isn't worth a dataclass (it's
private, or genuinely just one tensor), don't force-fit `Prepared*` — name it
for what it returns instead (see `_prepare_mtp_base` in `core/decoder.py`,
which returns a bare tensor; that's the deliberate exception, not a
convention violation).

**`*Owner` Protocols** — every `Protocol` describing "the object a free
function or strategy method is handed as its first argument / `self`" ends
in `Owner` (`LazyAttentionOwner`, `ExecutionOwner`, `LayerInvokeOwner`,
`DecoderStackOwner`, `EncoderPreparationOwner`, `DecoderOwner`,
`RoutingOwner`, `MHCConnectionOwner`). This applies uniformly whether the
protocol describes a `self`-type consumed by a mixin method or an external
collaborator object held by reference (e.g. `GenerationEngine.decoder:
DecoderOwner`) — both are "the thing this code calls back into." Don't
introduce `*Protocol` or bare interface names for this role; rename to
`*Owner` instead.

**`*Strategy` vs `*Mixin` vs `*Cfg`/`*Config`** — three distinct roles, not
synonyms:
- **`*Strategy`**: a composed object, held as an attribute, with behavior
  methods (`ModelLayerInvokeStrategy.run_encoder_layer`,
  `LayerExecutionStrategy.run_block`). Use when the caller needs to hold a
  reference and call into it repeatedly, or when the behavior varies by a
  runtime flag captured in the object (e.g. `use_checkpoint`, `use_mhc`).
- **`*Mixin`**: behavior inherited directly into a layer class
  (`ResidualBlockMixin`, `MHCExecutionMixin`, `LazyAttentionBackendMixin`).
  Use when the methods need direct access to the layer's own attributes via
  `self`, rather than through an `Owner` protocol passed as an argument.
- **`*Cfg`/`*Config`**: plain data, no behavior (`ResidualRunCfg`,
  `ResidualConfig`, `CacheConfig`, `GenerationConfig`, `TunerConfig`). Never
  add methods beyond validation (`__post_init__`) or trivial derivation
  properties.

There is no `*Policy` class anywhere in this package — don't add one, and
don't describe a `*Config`/`*Strategy`/`*Mixin` as a "policy" in a docstring;
say what it actually is (a config, a strategy, a mixin).

**Verb prefixes** on functions and staticmethods:
- **`build_*`**: constructs and returns an object — an `nn.Module`
  (`build_layer_attention_backend`, `build_positional_encoder`,
  `NormWrapper.build`) or an assembled data object
  (`build_decoder_output`, `build_layer_modules`). No side effects beyond
  construction.
- **`prepare_*`**: normalizes/transforms input data into a `Prepared*`
  struct. No model construction, no layer invocation.
- **`run_*`** / **`execute_*`**: performs the actual layer/sublayer
  invocation — effectful, not idempotent to call twice with the same state
  (`run_encoder_layer`, `run_decoder_layer`, `run_block`, `run_mod_layer`,
  `execute_decoder_layer`).

**`MHC*` naming** — `MHCHyperConnection` (`features/mhc.py`) is the concrete
learnable module. `MHCConnectionOwner` (`runtime/execution.py`) is the
`Protocol` it satisfies. `MHCExecutionMixin` (`runtime/execution.py`) is the
mixin that drives one through a layer's forward pass. Keep the three-way
split — module vs. protocol vs. execution mixin — rather than collapsing
them; they live at different layers of the `core` → `runtime` → `features`
dependency direction described above.

**Encoder/decoder asymmetry is intentional, not a naming bug.** The decoder
has state the encoder doesn't (KV cache, incremental decoding), so it has
more machinery (`DecoderState`, `DecoderCacheManager`, `DecoderLayerResult`,
`build_decoder_output`) with no encoder counterpart. Don't manufacture a
parallel `EncoderLayerResult`/`build_encoder_output` just to make the two
symmetric — only add one if the encoder actually grows a matching need.
