# Transformer Guide

ForeBlocks ships a flexible encoder-decoder transformer stack centered on `TransformerEncoder` and `TransformerDecoder`.

The current implementation supports:

- multiple attention backends and per-layer attention routing
- encoder and decoder patching
- CT-PatchTST-style encoder tokenization
- paper-style Attention Residuals
- GateSkip
- Mixture-of-Depths (MoD)
- mHC residual stream mixing
- MoE feedforward blocks
- gradient checkpointing and shared-layer reuse

Related docs:

- [Documentation Overview](overview.md)
- [Getting Started](getting-started.md)
- [Custom Blocks](custom_blocks.md)
- [MoE](moe.md)
- [DARTS](darts.md)

## Import

```python
from foreblocks import TransformerEncoder, TransformerDecoder
```

## Mental model

The safest path is:

1. start with a dense encoder-decoder transformer
2. patch the encoder if the source sequence is long
3. keep the decoder timestep-level
4. verify full-sequence and autoregressive inference
5. only then add MoE, GateSkip, MoD, or mHC

For most time-series setups, the best default is:

- `patch_encoder=True`
- `patch_decoder=False`
- `attention_mode="standard"`
- `norm_strategy="pre_norm"`
- `custom_norm="rms"`

## Baseline encoder

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    attention_mode="standard",
    norm_strategy="pre_norm",
    custom_norm="rms",
    patch_encoder=True,
    patch_len=16,
    patch_stride=8,
)
```

## Baseline decoder

```python
decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    d_model=256,
    nhead=8,
    num_layers=4,
    patch_decoder=False,
    informer_like=False,
)
```

## Constructor groups

### Backbone

- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `dropout`
- `max_seq_len`

### Normalization and FFN

- `norm_strategy`: `pre_norm`, `post_norm`, or `sandwich_norm`
- `custom_norm`: `rms`, `layer`, and other norm-factory variants
- `use_final_norm`
- `use_swiglu`

### Attention selection

- `att_type`
- `attention_mode`
- `freq_modes`

Supported `attention_mode` values currently include:

- `standard`
- `linear`
- `sype`
- `hybrid`
- `kimi`
- `hybrid_kimi`
- `kimi_3to1`
- `gated_delta`
- `hybrid_gdn`
- `gdn_3to1`

Important behavior:

- if `attention_mode="standard"` but `att_type` is a routed type such as `linear`, `sype`, `kimi`, or `gated_delta`, the model promotes `attention_mode` automatically

### Patching

- `patch_encoder`
- `patch_decoder`
- `patch_len`
- `patch_stride`
- `patch_pad_end`

### Efficiency

- `use_gradient_checkpointing`
- `share_layers`

### Advanced modules

- Attention Residuals:
  `use_attention_residual`, `attn_residual_type`, `attention_residual_block_size`
- GateSkip:
  `use_gateskip`, `gate_budget`, `gate_lambda`
- MoD:
  `use_mod`, `mod_mode`, `mod_lambda`, `mod_budget_scheduler`
- mHC:
  `use_mhc`, `mhc_n_streams`, `mhc_sinkhorn_iters`, `mhc_collapse`
- MoE:
  `use_moe`, `num_experts`, `top_k`, `moe_aux_lambda`

## Recommended patching strategy

The recommended pattern for forecasting is:

- `patch_encoder=True`
- `patch_decoder=False`

Why:

- the encoder benefits from shorter token sequences
- the decoder stays easier to reason about
- autoregressive decoding stays compatible with `forward_one_step(...)`

`patch_decoder=True` is supported for full-sequence decoding, but it is not compatible with KV-cached incremental decoding.

When the encoder is patched, the memory sequence length becomes patch-token length. The decoder validates that `memory_key_padding_mask` matches the actual memory length, so patched and unpatched masks cannot be mixed silently.

## CT-PatchTST encoder mode

The encoder also supports a channel-token PatchTST-style path:

```python
encoder = TransformerEncoder(
    input_size=8,
    ct_patchtst=True,
    ct_patch_len=16,
    ct_patch_stride=8,
    ct_patch_pad_end=True,
    ct_patch_fuse="linear",  # or "mean"
    d_model=256,
)
```

This path:

- patchifies across time per channel
- embeds each channel patch
- fuses channels into transformer tokens

Use it when timestep-level tokenization is too expensive for long multivariate histories.

## Inference modes

### Full-sequence encoder-decoder

This is the default path for standard sequence-to-sequence forecasting:

```python
memory = encoder(src, src_key_padding_mask=src_kpm)
out = decoder(
    tgt,
    memory,
    memory_key_padding_mask=src_kpm_or_patchified_memory_kpm,
)
```

### Informer-like decoding

`model_type="informer-like"` changes defaults so that:

- encoder time encoding is enabled
- decoder informer-like behavior is enabled
- decoder prompt masking follows `label_len`

Typical setup:

```python
decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    model_type="informer-like",
    label_len=12,
    d_model=256,
    nhead=8,
    num_layers=4,
)
```

`label_len` controls how much of the decoder input is treated as observed prompt.

Important behavior:

- set `label_len` explicitly for true Informer-style masking
- when `label_len <= 0`, the implementation now skips the automatic Informer padding mask instead of masking the whole decoder input

### Autoregressive decoding

`forward_one_step(...)` is intended for KV-cached autoregressive decoding.

```python
step_out, state = decoder.forward_one_step(tgt_prefix, memory)
step_out, state = decoder.forward_one_step(
    next_token,
    memory,
    incremental_state=state,
    memory_key_padding_mask=memory_kpm,
)
```

Recommended usage:

- first call: pass the available prefix
- later calls: pass either the growing prefix or only the newest token
- once cache exists, the implementation consumes only the newest step

Current constraints:

- requires `patch_decoder=False`
- does not support `use_mod=True`
- does not support `use_mhc=True`

## Active-position masks for time series

Both GateSkip and MoD operate over active positions. The public runtime input is:

- encoder: `gateskip_active_mask`
- decoder: `gateskip_active_mask`

For time series, the intended meaning is:

- `True`: this timestep or token participates in budgeting or routing
- `False`: inactive position such as padding or masked-out region

Default behavior:

- encoder: active positions are derived from `src_key_padding_mask` when available
- decoder: active positions are derived from the user-provided target padding mask
- the auto-generated Informer forecast mask is intentionally not treated as inactivity for GateSkip or MoD

With patching enabled, the active mask is patchified too, so routing stays aligned with patch tokens.

## Attention Residuals

The transformer now implements paper-style Attention Residuals rather than the older local residual trick.

Controls:

- `use_attention_residual`
- `attn_residual_type`: `full` or `block`
- `attention_residual_block_size`

Behavior:

- `full`: aggregates over the running layer history
- `block`: aggregates over block summaries

Notes:

- this is enabled by default
- it replaces the normal residual path for the affected blocks

Current compatibility rules:

- not compatible with `use_gateskip=True`
- not compatible with `use_mhc=True`
- not compatible with `use_mod=True`

If you want GateSkip, MoD, or mHC, disable Attention Residuals explicitly:

```python
use_attention_residual=False
```

## GateSkip

GateSkip applies residual gating at the sublayer level.

Controls:

- `use_gateskip`
- `gate_budget`
- `gate_lambda`

For time series, GateSkip budgets over valid positions rather than LM-style EOS handling.

Recommendation:

- keep it off until the dense baseline is stable
- when using it, pass an explicit `gateskip_active_mask` if you want forecast-only gating rather than all valid positions

Current compatibility rules:

- not wired together with Attention Residuals
- not wired together with MoD

## Mixture-of-Depths

The transformer supports paper-style MoD token routing.

Controls:

- `use_mod`
- `mod_mode`
- `mod_lambda`
- `mod_budget_scheduler`

Current behavior:

- only `mod_mode="token"` is supported
- routing is top-k over active positions
- packed routed tokens are processed and scattered back

For time series:

- routing is timestep or patch-token routing
- default active positions are all valid positions
- if you want forecast-only routing, provide an explicit `gateskip_active_mask`

Current compatibility rules:

- not compatible with Attention Residuals
- not compatible with GateSkip
- not compatible with mHC
- not supported in `forward_one_step(...)`

## mHC residual streams

mHC adds multiple residual streams and dynamic hyper-connections between them.

Controls:

- `use_mhc`
- `mhc_n_streams`
- `mhc_sinkhorn_iters`
- `mhc_collapse`: `first` or `mean`

Current behavior:

- paper-style stream init is `(x, 0, ..., 0)`
- stream read/write and residual mixing are token-wise and input-dependent
- `mhc_collapse="first"` is the safest default

Current compatibility rules:

- not compatible with Attention Residuals
- not compatible with MoD
- not supported in decoder KV-cached autoregressive decoding

Use it as a research feature rather than a first production default.

## MoE in transformer layers

MoE is enabled at the feedforward block level:

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    use_moe=True,
    num_experts=8,
    top_k=2,
    moe_aux_lambda=1.0,
)
```

See the dedicated guide for routing and auxiliary-loss details:

- [MoE Guide](moe.md)

## Integration with `ForecastingModel`

```python
from foreblocks import ForecastingModel, TransformerEncoder, TransformerDecoder

encoder = TransformerEncoder(
    input_size=8,
    d_model=128,
    nhead=4,
    num_layers=3,
    patch_encoder=True,
    patch_len=16,
    patch_stride=8,
)

decoder = TransformerDecoder(
    input_size=1,
    output_size=1,
    d_model=128,
    nhead=4,
    num_layers=3,
    patch_decoder=False,
    informer_like=False,
)

model = ForecastingModel(
    encoder=encoder,
    decoder=decoder,
    forecasting_strategy="transformer_seq2seq",
    model_type="transformer",
    target_len=24,
    output_size=1,
)
```

## Recommended tuning order

1. Get a plain encoder-decoder transformer running with `standard` attention.
2. Enable encoder patching if source sequence length is large.
3. Verify full-sequence and autoregressive inference.
4. Explore `attention_mode` variants.
5. Add MoE only after the dense baseline is stable.
6. Add GateSkip, MoD, or mHC only for targeted experiments.

## Troubleshooting

- `Sequence length exceeds max_seq_len`: increase `max_seq_len` or enable patching.
- Decoder/memory mask mismatch: the encoder may be patched while the memory padding mask was not patchified consistently.
- `patch_decoder=True` with KV caching: unsupported; keep decoder patching off for autoregressive decoding.
- `forward_one_step(...)` errors with MoD or mHC: those features are intentionally disabled in the incremental path.
- Attention Residuals with GateSkip, MoD, or mHC: unsupported in the current implementation.
- Informer-like mode behaving like plain decoding: set `label_len` explicitly.
- OOM: reduce `d_model`, `num_layers`, `dim_feedforward`, or enable gradient checkpointing.
