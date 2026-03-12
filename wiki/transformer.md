# Transformer Guide

ForeBlocks ships a broad transformer stack centered on `TransformerEncoder` and `TransformerDecoder`.

The implementation supports:

- multiple self-attention kernels
- per-layer attention routing
- encoder/decoder patching
- CT-PatchTST-style encoder tokenization
- MoE feedforward layers
- dynamic layer skipping
- mHC residual stream mixing
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

The current stack has three levels of control:

1. backbone dimensions and normalization
2. tokenization and attention-routing behavior
3. optional advanced modules such as MoE, mHC, and layer skipping

The safest baseline is:

- patch the encoder
- keep the decoder timestep-level
- use `pre_norm`
- start with standard attention
- leave MoE, mHC, and dynamic skipping off until the basic path works

## Baseline encoder

```python
encoder = TransformerEncoder(
    input_size=8,
    d_model=256,
    nhead=8,
    num_layers=4,
    dim_feedforward=1024,
    dropout=0.1,
    att_type="standard",
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
    label_len=12,
    informer_like=False,
    patch_decoder=False,
)
```

## Core constructor groups

### Dimensions and depth

- `d_model`
- `nhead`
- `num_layers`
- `dim_feedforward`
- `dropout`
- `max_seq_len`

### Normalization and residual behavior

- `norm_strategy`: `pre_norm`, `post_norm`, or `sandwich_norm`
- `custom_norm`: `rms`, `layer`, and other norm-factory variants
- `use_final_norm`
- `use_swiglu`

### Attention selection

- `att_type`: the base attention family for standard layers
- `attention_mode`: how attention types are assigned across layers

Supported routed modes in the current implementation include:

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

- if `attention_mode="standard"` but `att_type` is a routed type such as `linear`, `sype`, `kimi`, or `gated_delta`, the model promotes `attention_mode` automatically so the requested path is actually used

### Patching and tokenization

- `patch_encoder`
- `patch_decoder`
- `patch_len`
- `patch_stride`
- `patch_pad_end`

Encoder patching is the recommended default. The encoder returns patch-memory tokens without unpatching, and the decoder cross-attends to that patch memory.

### Efficiency and parameter sharing

- `use_gradient_checkpointing`
- `share_layers`

### Advanced modules

- `use_moe`, `num_experts`, `top_k`, `moe_aux_lambda`
- `use_layer_skipping`, `layer_skip_mode`, `layer_skip_temperature`, `layer_skip_lambda`
- `use_mhc`, `mhc_n_streams`, `mhc_sinkhorn_iters`, `mhc_temperature`, `mhc_collapse`

## Attention routing patterns

`attention_mode` controls how attention kernels are assigned across the layer stack.

Common choices:

- `standard`: all layers use standard attention
- `linear`: all layers use linear attention
- `hybrid`: early layers use linear attention, final layer uses standard attention
- `hybrid_kimi`: early layers use Kimi attention, final layer uses standard attention
- `kimi_3to1`: three Kimi layers followed by one standard layer in repeating groups
- `hybrid_gdn`: early layers use Gated DeltaNet, final layer uses standard attention

Use these routed modes when:

- sequence length is large
- you want cheaper early layers with a stronger final layer
- you are experimenting with linear or state-space-like attention variants

## Patching strategy

The current implementation is explicit about patching behavior.

### Recommended pattern

- `patch_encoder=True`
- `patch_decoder=False`

Why:

- the encoder benefits from shorter token sequences
- the decoder remains easier to reason about
- autoregressive decoding remains compatible with `forward_one_step(...)`

### Decoder patching caveat

`patch_decoder=True` is supported for non-incremental decoding, but it is not compatible with KV-cached incremental decoding.

### Memory mask alignment

When the encoder is patched, the memory sequence length changes from timestep length to patch-token length. The implementation validates that `memory_key_padding_mask` matches the actual memory length, so patched and unpatched masks cannot be mixed accidentally.

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
- embeds each channel-patch
- fuses channels into transformer tokens

Use it when long input sequences make timestep-level tokenization too expensive.

## Informer-like mode

`model_type="informer-like"` changes defaults in the current implementation:

- encoder time encoding is enabled
- decoder informer-like behavior is enabled
- decoder prompt masking behavior follows `label_len`

Typical decoder setup:

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

## Decoder behavior and constraints

### Prompting

The decoder consumes:

- `tgt`: decoder prompt sequence
- `memory`: encoder output sequence

`label_len` controls how much prompt is treated as observed context in informer-like decoding.

### Incremental decoding

`forward_one_step(...)` is intended for autoregressive decoding with KV caching.

Current constraints:

- requires `patch_decoder=False`
- does not support dynamic layer skipping
- mHC is not supported with incremental decoder state

### MTP targets

The decoder supports optional multi-token prediction targets for MoE FFNs in decoder layers. That is an advanced path and should only be enabled when you intentionally want auxiliary decoder-horizon supervision inside the FFN block.

## Dynamic layer skipping

The transformer base supports MoD-style layer skipping.

Key controls:

- `use_layer_skipping`
- `layer_skip_mode`: `seq` or `token`
- `layer_skip_temperature`
- `layer_skip_hard`
- `layer_skip_lambda`

Current behavior:

- `seq` mode can skip whole layers and save compute
- `token` mode is behavioral mixing, not a true compute-saving token-pruning path

Recommendation:

- leave this off until you have a stable baseline
- prefer `seq` mode first if you want actual compute savings

## mHC residual streams

mHC adds multiple residual streams internally and mixes them with a Sinkhorn-constrained residual mixer.

Key controls:

- `use_mhc`
- `mhc_n_streams`
- `mhc_sinkhorn_iters`
- `mhc_temperature`
- `mhc_collapse`: `first` or `mean`

Use it for research exploration, not as a first-line production default. It changes the residual dynamics substantially and has more runtime constraints than the plain transformer path.

## MoE in transformer layers

MoE is enabled at the feedforward block level through the transformer constructors:

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

See the dedicated guide for the routing and auxiliary-loss details:

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
2. Enable encoder patching if sequence length is large.
3. Explore `attention_mode` variants.
4. Add MoE only after the dense baseline is stable.
5. Add layer skipping or mHC only for targeted experiments.

## Troubleshooting

- `Sequence length exceeds max_seq_len`: increase `max_seq_len` or enable patching.
- Decoder/memory mask mismatch: this often means the encoder is patched but the memory padding mask was not patchified consistently.
- `patch_decoder=True` with KV caching: unsupported; keep decoder patching off for autoregressive decoding.
- `forward_one_step(...)` errors with layer skipping or mHC: those features are intentionally disabled in the incremental path.
- OOM: reduce `d_model`, `num_layers`, `dim_feedforward`, or enable gradient checkpointing.
