---
title: Transformer Guide
description: Transformer stack — backbones, attention variants, MoE, norms, and embeddings.
editLink: true
---


[[toc]]
# Transformer Guide

ForeBlocks ships a flexible encoder-decoder transformer stack centered on `TransformerEncoder` and `TransformerDecoder`.

The current implementation supports:

- multiple attention backends, mixed across depth
- encoder patching (shared or per-channel) and variate mixing
- one residual policy per model: standard, GateSkip, mHC, Mixture-of-Depths, or Attention Residuals
- MoE feedforward blocks with multiple routers and load-balancing
- per-layer dropout schedules, gradient checkpointing, and shared-layer reuse
- incremental KV-cache decoding (greedy, beam, speculative)

Related docs:

- [Documentation Overview](overview)
- [Getting Started](getting-started)
- [Custom Blocks](custom_blocks)
- **[Advanced Transformer Features](transformer-advanced)** — LLRD, per-layer dropout, GateSkip, MoD, mHC, attention variants
- **[Advanced MoE](moe-advanced)** — routers, load-balancing, expert types, production tuning
- [MoE](moe)
- [DARTS](darts)

## Building a model

Every setting is a keyword argument. Pass them directly, collect them in a
`TransformerConfig`, or both — keywords override the config:

```python
from foreblocks import TransformerDecoder, TransformerEncoder
from foreblocks.nn.transformer import TransformerConfig

encoder = TransformerEncoder(input_size=8, d_model=256, n_heads=8, num_layers=4)

config = TransformerConfig(d_model=256, n_heads=8, num_layers=4, residual="gateskip")
encoder = TransformerEncoder(config, input_size=8)
decoder = TransformerDecoder(config, input_size=1, output_size=1)
```

A config is frozen, plain data: `config.to_dict()` is JSON-serializable and
`TransformerConfig.from_dict(...)` restores it. Unknown names and invalid values
raise immediately.

Live objects are constructor arguments rather than config fields:
`pos_encoder=`, `gate_scheduler=`, `mod_scheduler=`, and `dropout_schedule=`.

## Settings

| Group | Settings |
| --- | --- |
| Shape | `input_size`, `output_size`, `d_model`, `n_heads`, `n_kv_heads`, `num_layers`, `ff_dim`, `dropout`, `activation`, `swiglu`, `max_seq_len` |
| Attention | `attention`, `attention_pattern`, `attention_kernel`, `frequency_modes`, `attention_options` |
| Positions | `position`, `position_scale`, `rope_base`, `rope_scaling`, `rope_scaling_factor`, `time_encoding` |
| Normalization | `norm` (`rms`/`layer`), `norm_placement` (`pre`/`post`/`sandwich`), `norm_eps`, `final_norm` |
| Mixture of experts | `moe_experts` (0 = dense), `moe_top_k`, `moe_latent`, `moe_latent_dim`, `moe_latent_ff_dim`, `moe_aux_weight`, `moe_options` |
| Residual | `residual`, `gate_budget`, `gate_aux_weight`, `mhc_streams`, `mhc_sinkhorn_iters`, `mhc_collapse`, `mod_aux_weight`, `attention_residual_mode`, `attention_residual_block_size` |
| Execution | `share_layers`, `gradient_checkpointing`, `init_std`, `depth_scaled_init` |
| Encoder input | `patching`, `patch_len`, `patch_stride`, `patch_pad_end`, `channel_fuse`, `variate_attention`, `variate_fuse`, `variate_position`, `contiguous_decoding`, `quantiles` |
| Decoder | `informer`, `label_len`, `kv_cache` |

## Attention

`attention` names the backend of every layer. It is either a recurrent or
linear backend — `linear`, `gla`, `deltanet`, `gated_deltanet`,
`gated_delta`, `kimi` — or a softmax variant: `standard`, `sype`,
`prob_sparse`, `frequency`, `sliding_window`, `dilated_window`, `moba`, `nsa`,
`softpick`, `autocor`, `dwt`, ...

`attention_pattern` mixes that backend with standard attention across depth:

- `uniform` (default): every layer uses `attention`
- `hybrid`: every layer but the last uses `attention`
- `3to1`: three of every four layers use `attention`

```python
encoder = TransformerEncoder(
    input_size=8, d_model=256, num_layers=8,
    attention="gla", attention_pattern="3to1",
)
```

Rarely used attention settings go through `attention_options`, keyed by the
field names of `foreblocks.nn.attention.config` (`window_size`, `qk_norm`,
`logit_softcap`, `use_mla`, `attention_matching`, ...):

```python
encoder = TransformerEncoder(
    input_size=8, attention="sliding_window",
    attention_options={"window_size": 128, "qk_norm": True},
)
```

## Residual policies

Exactly one residual policy is active per model:

| `residual=` | Behavior | Settings |
| --- | --- | --- |
| `standard` | Ordinary residual connections | — |
| `gateskip` | Learned per-token gates on each sublayer update | `gate_budget`, `gate_aux_weight`, `gate_scheduler=` |
| `mhc` | Manifold-constrained hyper-connections over parallel streams | `mhc_streams`, `mhc_sinkhorn_iters`, `mhc_collapse` |
| `mod` | Mixture-of-Depths: each layer processes only routed tokens | `mod_aux_weight`, `mod_scheduler=` |
| `attention` | Attention Residuals over earlier layer outputs | `attention_residual_mode` (`full`/`block`), `attention_residual_block_size` |

`mhc` and `attention` cannot be combined with `gradient_checkpointing`; `mhc`
and `mod` do not support KV-cached decoding.

## Encoder patching

`patching` selects how the encoder tokenizes `[B, T, C]` input:

- `shared` (default): project each step, then embed patches of `patch_len` steps every `patch_stride` steps
- `channel`: embed each channel's patches and fuse channels per patch (`channel_fuse="linear"` or `"mean"`)
- `none`: one token per step

When the encoder is patched, the memory sequence length becomes the number of
patches. The decoder validates that `memory_key_padding_mask` matches the
memory length, so patched and unpatched masks cannot be mixed silently.

`variate_attention=True` keeps each channel as its own token stream and mixes
across channels in every layer; with `contiguous_decoding=True` the encoder
forecasts a whole horizon in one pass (`encoder.forecast_contiguous(...)`).

## Informer-style decoding

`informer=True` makes decoder self-attention non-causal and masks positions
after `label_len` as padding, so the decoder reads a known prefix and fills
the horizon in one pass:

```python
decoder = TransformerDecoder(
    input_size=1, output_size=1, d_model=256, n_heads=8, num_layers=4,
    informer=True, label_len=12, time_encoding=True,
)
```

The default (`informer=False`) is an ordinary causal decoder, which supports
incremental decoding through `prefill`, `decode`, `forward_one_step`, and
`generate`.

## Active-position masks

GateSkip and Mixture-of-Depths operate over active positions, passed to
`forward` as `active_mask` (`True` marks a position that participates). By
default it is derived from the caller's padding mask. The decoder's automatic
Informer horizon mask is intentionally not treated as inactivity. With
patching, the active mask is patchified too, so routing stays aligned with
patch tokens.

## Integration with `ForecastingModel`

See the [Custom Blocks Guide](custom_blocks) for wiring transformers into `ForecastingModel`.

## Transformer Tuner

`foreblocks.tuning.TransformerTuner` recommends patch lengths, attention, and
preprocessing from series characteristics:

- **Lempel-Ziv complexity** analysis for sequence structure
- **Continuous Wavelet Transform (CWT) energy** features for frequency domain analysis
