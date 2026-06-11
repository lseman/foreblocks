---
title: Package Reorganization — Migration Map
description: Old → new import mappings for the ops/layers/modules/models/sequence reshape.
editLink: true
---

# Package Reorganization — Migration Map

This is the authoritative old → new mapping for the `ops / layers / modules / models / sequence`
reshape of `foreblocks/`. Renames are **hard** (no compatibility shims); every import site in
`foreblocks/`, `tests/`, `examples/`, and `darts/` is rewritten in the same change.

Relative imports were first normalized to absolute `foreblocks.…` form so moves are mechanical.

## Module-prefix mapping

| Old import prefix | New import prefix | Notes |
| --- | --- | --- |
| `foreblocks.transformer.kernels` | `foreblocks.ops.kernels` | triton/general kernels |
| `foreblocks.transformer.attention.kernels` | `foreblocks.ops.attention` | fla_*, fused_rope, paged_decode, chunked linear |
| `foreblocks.transformer.norms.triton_backend` | `foreblocks.ops.norms_triton` | triton norm backend (compute) |
| `foreblocks.custom_mamba.ops` | `foreblocks.ops.mamba` | ssd, causal_conv1d, mamba2_combined, rms_norm, rotary, triton_ops |
| `foreblocks.custom_raven.ops` | `foreblocks.ops.raven` | raven backend ops |
| `foreblocks.transformer.norms` | `foreblocks.layers.norms` | group/layer/rms/temporal/revin nn.Modules |
| `foreblocks.transformer.embeddings` | `foreblocks.layers.embeddings` | rotary, alibi, positional, time embeds |
| `foreblocks.layers.graph` | `foreblocks.layers.graph` | unchanged (already a layer family) |
| `foreblocks.transformer.attention` | `foreblocks.modules.attention` | multi_att, variants, modules/linear_att, cache, utils |
| `foreblocks.transformer.moe` | `foreblocks.modules.moe` | experts, routers, ff |
| `foreblocks.transformer.skip` | `foreblocks.modules.skip` | gateskip, mod |
| `foreblocks.blocks.popular` | `foreblocks.models.popular` | nbeats, nha, timesnet (merged) |
| `foreblocks.blocks` | `foreblocks.modules.blocks` | tcn, ode, fourier, wavelets, xlstm, enc_dec, … |
| `foreblocks.core.heads` | `foreblocks.modules.heads` | head families + head modules |
| `foreblocks.transformer.popular` | `foreblocks.models.popular` | informer, autoformer, … (merged with blocks.popular) |
| `foreblocks.transformer` | `foreblocks.models.transformer` | transformer, tf_*, patching, fusions, sype, mhc, transformer_tuner |
| `foreblocks.custom_mamba.blocks` | `foreblocks.sequence.mamba` | HybridMamba family |
| `foreblocks.custom_mamba` | `foreblocks.sequence.mamba` | package root |
| `foreblocks.mamba` | `foreblocks.sequence.mamba` | older Mamba backbone |
| `foreblocks.custom_raven` | `foreblocks.sequence.raven` | raven blocks + configuration |
| `foreblocks.kan` | `foreblocks.kan` | **unchanged** (kept top-level per decision) |
| `foreblocks.custom_att` | `foreblocks.experimental.attention_kernels` | vendored sub-project (own setup.py) |

Unchanged top-level: `core` (model, att, sampling, extend), `data`, `training`, `evaluation`,
`ts_handler`, `mltracker`, `ui`, `studio`, `third_party`, `config.py`, `models` (forecasting,
graph_forecasting stay; `popular/` + `transformer/` added under it).

## Ordering note

Longer/more-specific prefixes are applied before shorter ones (e.g. `transformer.attention.kernels`
before `transformer.attention` before `transformer`) so nested moves don't get double-rewritten.
