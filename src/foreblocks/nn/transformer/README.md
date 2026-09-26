# foreblocks.nn.transformer

Modular transformer encoder/decoder stack: pluggable attention backends
(standard, linear, GLA, DeltaNet, GatedDeltaNet, SyPE, ...), GateSkip residual
gating, manifold hyper-connections (mHC), attention-residual accumulation,
Mixture-of-Depths routing, patch tokenization, and incremental KV-cache
decoding (greedy / beam / speculative).

```python
from foreblocks.nn.transformer import TransformerDecoder, TransformerEncoder

encoder = TransformerEncoder(input_size=4, d_model=64, n_heads=4, num_layers=2)
decoder = TransformerDecoder(encoder.config, input_size=2, output_size=2)
memory = encoder(x).last_hidden_state
forecast = decoder(tgt, memory).last_hidden_state
```

## Directory structure

```text
transformer/
├── __init__.py                  # Lazy public exports
├── config.py                    # TransformerConfig (flat settings), GenerationConfig
├── base.py                      # BaseTransformer: stack construction and shared forward steps
├── encoder.py                   # TransformerEncoder: encoder stack and forecasting
├── decoder.py                   # TransformerDecoder: decoder stack and incremental interface
├── layers/                      # Individual layers; no dependency on model stacks
│   ├── base.py                  # BaseTransformerLayer: FFN/MoE and residual-policy modules
│   ├── encoder.py               # TransformerEncoderLayer: self-attention and FFN
│   ├── decoder.py               # TransformerDecoderLayer: self/cross-attention and FFN
│   ├── mixing.py                # MixingTransformer: sequence-then-variate attention for [B,V,N,D]
│   └── axis_attention.py        # Sequence/variate attention reshaping and masks
├── attention_backends.py        # Lazy attention backend selection and materialization
└── runtime/                     # Execution helpers without concrete stack imports
    ├── execution.py             # Layer and sublayer execution strategies/mixins
    ├── forward.py               # Input/state preparation and per-layer execution
    ├── contiguous.py            # Masked-horizon preparation, projection, and quantile loss
    ├── decoding.py              # GenerationEngine, beam search, speculative decoding
    ├── routing.py               # Mixture-of-Depths gather/scatter helpers
    ├── mtp.py                   # MTP target validation and horizon alignment
    ├── cache.py                 # DecoderCacheManager
    ├── contracts.py             # DecoderOwner protocol
    ├── state.py                 # Typed incremental decoder state
    ├── residual_state.py        # Attention-residual accumulation state
    └── outputs.py               # Structured encoder, decoder, and generation outputs
```

The stack modules (`base.py`, `encoder.py`, `decoder.py`) compose individual
`layers/` and call `runtime/` helpers. Layers depend on the config, attention
backends, and runtime execution, without importing the concrete model stacks.
Runtime helpers use protocols when they need a stack collaborator. Reusable
neural modules live in sibling packages such as `foreblocks.nn.attention`,
`foreblocks.nn.embeddings`, `foreblocks.nn.residual`, and `foreblocks.nn.routing`.

Keep per-layer construction and forward behavior in `layers/`; keep layer-stack
orchestration, input/output adaptation, and public model methods in the stack
modules. Tensor preparation, cache handling, and generation algorithms belong in
`runtime/`. The package and `layers/` expose classes lazily, so importing a
layer does not load an encoder or decoder stack.

## Configuration

`TransformerConfig` is one frozen dataclass of flat, typed settings — there
are no nested config objects to build. Every module accepts a config, keyword
overrides, or both; keywords are applied over the config with
`dataclasses.replace`, and the original config is never mutated:

```python
from foreblocks.nn.transformer import TransformerConfig, TransformerEncoder

encoder = TransformerEncoder(input_size=4, d_model=64, residual="mhc")

config = TransformerConfig(
    d_model=64, n_heads=4, attention="gla", attention_pattern="hybrid"
)
encoder = TransformerEncoder(config, input_size=4)
assert TransformerEncoder(config).config is config
```

| Group | Settings |
| --- | --- |
| Shape | `input_size`, `output_size`, `d_model`, `n_heads`, `n_kv_heads`, `num_layers`, `ff_dim`, `dropout`, `activation`, `swiglu`, `max_seq_len` |
| Attention | `attention`, `attention_pattern`, `attention_kernel`, `frequency_modes`, `attention_options` |
| Positions | `position`, `position_scale`, `rope_base`, `rope_scaling`, `rope_scaling_factor`, `time_encoding` |
| Normalization | `norm`, `norm_placement`, `norm_eps`, `final_norm` |
| Mixture of experts | `moe_experts` (0 = dense), `moe_top_k`, `moe_latent`, `moe_latent_dim`, `moe_latent_ff_dim`, `moe_aux_weight`, `moe_options` |
| Residual | `residual`, `gate_*`, `mhc_*`, `mod_aux_weight`, `attention_residual_*` |
| Execution | `share_layers`, `gradient_checkpointing`, `init_std`, `depth_scaled_init` |
| Encoder input | `patching`, `patch_len`, `patch_stride`, `patch_pad_end`, `channel_fuse`, `variate_*`, `contiguous_decoding`, `quantiles` |
| Decoder | `informer`, `label_len`, `kv_cache` |

Settings that only some backends use go through two validated mappings:
`attention_options` (fields of `foreblocks.nn.attention.config`, e.g.
`window_size`, `qk_norm`, `use_mla`) and `moe_options` (arguments of
`FeedForwardBlock`, e.g. `num_shared`, `router_type`). Unknown names raise.

Validation happens when the config is built: invalid values, unknown names,
and incompatible combinations (for example `variate_attention` with a
non-standard residual, or `gradient_checkpointing` with `residual="mhc"`) raise
immediately. `validate_for("encoder" | "decoder")` adds role checks.

Derived views keep one source of truth:

- `config.attention_config(cross=False)` builds the nested `AttentionConfig`
  consumed by `foreblocks.nn.attention` (shape, cache, position, variant,
  features).
- `config.layer_attention(index)` and `config.for_layer(index, dropout=...)`
  resolve the per-depth backend of `attention_pattern` and a scheduled
  dropout. Stacks build each layer from `config.for_layer(index)`, so
  `layer.config` records the layer's effective settings.

`config.to_dict()` / `TransformerConfig.from_dict()` round-trip through JSON,
and configs pickle as plain data.

Modules read settings from `self.config`; nothing is copied onto the module.
Live collaborators are stack constructor arguments, never config fields:
`pos_encoder=`, `gate_scheduler=` (GateSkip budget annealing),
`mod_scheduler=` (Mixture-of-Depths keep rates), and `dropout_schedule=`
(per-layer dropout). A shared layer requires a constant dropout schedule.

### Residual policies

`residual` selects exactly one policy, and layers only build that policy's
modules: `gateskip` adds `gate_*` modules, `mhc` adds `mhc_conn_*` hyper
connections, `attention` adds `*_input_residual` modules, and `mod` adds
per-depth routers on the stack.

### Auxiliary losses

Structured encoder and decoder outputs expose differentiable `aux_loss` tensors.
Include them in the training objective, for example
`loss = prediction_loss + output.aux_loss`. MoE and GateSkip terms are collected
per executed layer invocation (including shared layers), averaged over executed
depths, and scaled by `moe_aux_weight`; the depth-router term is scaled by
`mod_aux_weight` and added separately. Checkpointing returns the auxiliary loss
with the hidden states so its gradients survive recomputation.

## Multivariate mixing

Set `variate_attention=True` to preserve input channels as `[B, V, N, D]`
inside the encoder. Each layer attends over the `N` sequence positions
independently per variate, then over the `V` variates independently per
sequence position. The default `variate_fuse="linear"` restores the usual
`[B, N, D]` encoder output; use `"mean"` or `"none"` for masked averaging or
an unfused `[B, V, N, D]` output. Variate attention is always non-causal and
uses no position encoding unless `variate_position=True`.

```python
model = TransformerEncoder(input_size=num_variates, variate_attention=True)
output = model(values).last_hidden_state  # values [B,N,V] -> output [B,N,D]
```

`MixingTransformer` is the lower-level layer for callers that already own
`[B, V, N, D]` tensors.

### Single-pass contiguous decoding

Enable `contiguous_decoding=True` with non-overlapping patches. The forecast
path appends learned missing-value tokens for the complete horizon and runs the
bidirectional encoder/mixing stack once. Targets and past-only covariates are
hidden over the horizon; past-future covariates remain visible.

```python
model = TransformerEncoder(
    input_size=num_targets + num_past_only + num_known_future,
    variate_attention=True,
    contiguous_decoding=True,
    patch_len=32,
    patch_stride=32,
)
quantile_forecast = model.forecast_contiguous(
    target,  # [B, context, targets]
    horizon=128,
    past_only_covariates=past,  # [B, context, past-only]
    past_future_covariates=known,  # [B, context + horizon, known-future]
)  # [B, horizon, targets, len(quantiles)]
```

`contiguous_quantile_loss` computes pinball loss for the configured quantiles.

## Naming convention

These conventions are enforced package-wide. If you're adding a new
`prepare_*`/`*Owner`/`*Strategy` symbol, match the existing one instead of
inventing a fourth variant.

**`Prepared*` dataclasses** — the return type of a public `prepare_*`
function is always named `Prepared<Thing>` (`PreparedEncoderInput`,
`PreparedDecoderState`). If a helper's result isn't worth a dataclass (it's
private, or genuinely just one tensor), don't force-fit `Prepared*` — name it
for what it returns instead. `build_decoder_mtp_targets` in `runtime/mtp.py`,
for example, constructs a bare target tensor rather than a preparation
dataclass.

**`*Owner` Protocols** — every `Protocol` describing "the object a free
function or strategy method is handed as its first argument / `self`" ends
in `Owner` (`LazyAttentionOwner`, `ExecutionOwner`, `LayerInvokeOwner`,
`StackOwner`, `EncoderPreparationOwner`, `DecoderOwner`, `RoutingOwner`,
`MHCConnectionOwner`). This applies uniformly whether the protocol describes a
`self`-type consumed by a mixin method or an external collaborator held by
reference (e.g. `GenerationEngine.decoder: DecoderOwner`). Don't introduce
`*Protocol` or bare interface names for this role.

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
  `TransformerConfig`, `GenerationConfig`). Beyond validation, a config only
  derives values from its own fields (`attention_config`, `for_layer`).

There is no `*Policy` class anywhere in this package — don't add one. The
`residual` setting names a residual policy, but it is a config value, not a
class.

**Verb prefixes** on functions and staticmethods:
- **`build_*`**: constructs and returns an object — an `nn.Module`
  (`build_layer_attention_backend`, `NormWrapper.build`) or an assembled data
  object (`build_decoder_output`). No side effects beyond construction.
- **`prepare_*`**: normalizes/transforms input data into a `Prepared*`
  struct. No model construction, no layer invocation.
- **`run_*`** / **`execute_*`**: performs the actual layer/sublayer
  invocation — effectful, not idempotent to call twice with the same state
  (`run_encoder_layer`, `run_decoder_layer`, `run_block`, `run_mod_layer`,
  `execute_encoder_layer`, `execute_decoder_layer`).

**`MHC*` naming** — `MHCHyperConnection` (`foreblocks.nn.residual.hyper_connections`)
is the concrete learnable module. `MHCConnectionOwner` (`runtime/execution.py`)
is the `Protocol` it satisfies. `MHCExecutionMixin` (`runtime/execution.py`) is
the mixin that drives one through a layer's forward pass. Keep the three-way
split — module vs. protocol vs. execution mixin.

**Encoder/decoder structure.** Both stacks run their layers through
`BaseTransformer._run_layer_stack` and return per-layer `LayerResult`s from
`execute_encoder_layer` / `execute_decoder_layer`. The decoder has state the
encoder doesn't (KV cache, incremental decoding), so `DecoderState`,
`DecoderCacheManager`, and `build_decoder_output` have no encoder counterpart;
only add one if the encoder grows a matching need.
