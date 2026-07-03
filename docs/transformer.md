---
title: Transformer Guide
description: Transformer stack — backbones, attention variants, MoE, norms, and embeddings.
editLink: true
---


[[toc]]
# Transformer Guide

ForeBlocks ships a flexible encoder-decoder transformer stack centered on `TransformerEncoder` and `TransformerDecoder`.

The current implementation supports:

- **Training optimization**: Layer-wise LR decay (LLRD) + warmup-cosine scheduler for SOTA fine-tuning
- **Per-layer dropout schedule**: Depth-scaled attention dropout (deeper layers → higher dropout)
- multiple attention backends and per-layer attention routing
- encoder and decoder patching
- CT-PatchTST-style encoder tokenization
- paper-style Attention Residuals
- GateSkip (residual path gating)
- Mixture-of-Depths (MoD) dynamic layer skipping
- mHC residual stream mixing (manifold-constrained hyper-connections)
- MoE feedforward blocks with multiple routers and load-balancing
- gradient checkpointing and shared-layer reuse

Related docs:

- [Documentation Overview](overview)
- [Getting Started](getting-started)
- [Custom Blocks](custom_blocks)
- **[Advanced Transformer Features](transformer-advanced)** — LLRD, per-layer dropout, GateSkip, MoD, mHC, attention variants
- **[Advanced MoE](moe-advanced)** — routers, load-balancing, expert types, production tuning
- [MoE](moe)
- [DARTS](darts)

## Import

```python
from foreblocks import TransformerEncoder, TransformerDecoder
```text

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
```text

#### `attention_mode` — routing schedule

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
```text

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
```text

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
```text

See the dedicated guide for routing and auxiliary-loss details:

- [MoE Guide](moe)

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
