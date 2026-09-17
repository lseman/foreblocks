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
| `foreblocks.transformer.attention` | `foreblocks.attention` | multi_att, variants, modules/linear_att, cache, utils |
| `foreblocks.transformer.moe` | `foreblocks.modules.moe` | experts, routers, ff |
| `foreblocks.transformer.skip` | `foreblocks.modules.skip` | gateskip, mod |
| `foreblocks.blocks.popular` | `foreblocks.models.popular` | nbeats, nha, timesnet (merged) |
| `foreblocks.blocks` | `foreblocks.modules.blocks` | tcn, ode, fourier, wavelets, xlstm, enc_dec, … |
| `foreblocks.core.heads` | `foreblocks.modules.heads` | head families + head modules |
| `foreblocks.transformer.popular` | `foreblocks.models.popular` | informer, autoformer, … (merged with blocks.popular) |
| `foreblocks.transformer` | `foreblocks.models.transformer` | transformer, tf_*, patching, fusions, sype, mhc, transformer_tuner |
| `foreblocks.custom_mamba.blocks` | `foreblocks.models.sequence.mamba` | HybridMamba family |
| `foreblocks.custom_mamba` | `foreblocks.models.sequence.mamba` | package root |
| `foreblocks.mamba` | `foreblocks.models.sequence.mamba` | older Mamba backbone |
| `foreblocks.custom_raven` | `foreblocks.models.sequence.raven` | raven blocks + configuration |
| `foreblocks.kan` | `foreblocks.models.kan` | moved under `models/` in a later pass (see below) |
| `foreblocks.custom_att` | `foreblocks.ops.experimental.attention_kernels` | vendored sub-project (own setup.py) |

Unchanged top-level (at the time of this reshape): `core` (model, att, sampling, extend), `data`,
`training`, `evaluation`, `ts_handler`, `mltracker`, `ui`, `third_party`, `config.py`, `models`
(forecasting, graph_forecasting stay; `popular/` + `transformer/` added under it).

## Ordering note

Longer/more-specific prefixes are applied before shorter ones (e.g. `transformer.attention.kernels`
before `transformer.attention` before `transformer`) so nested moves don't get double-rewritten.

## Second pass — root declutter + placement fixes

A later, smaller pass fixed drift and placement inconsistencies that accumulated after the
reshape above:

| Old import / path | New import / path | Notes |
| --- | --- | --- |
| `foreblocks.anomaly` | `foreblocks.models.anomaly` | nested under `models/` — it's an applications layer on `models/`+`core/`, not a foundational tier |
| `foreblocks.ui` | `foreblocks.studio` | renamed to avoid colliding with the `apps/webui` frontend it serves specs to |
| `src/tree/` | `projects/tree/` | standalone C++/CUDA library, never part of the packaged wheel |
| `scheduling/` (repo root) | `projects/scheduling/` | standalone RL/GNN side-project, unrelated to forecasting |
| `src/mltracker/dashboard_v2/` | `apps/mltracker-dashboard/` | frontend placement now consistent with `apps/webui/` |
| `src/tree/include/tree/neural_odst.py` | `projects/tree/neural_odst.py` | stray Python file out of a C++ `include/` header directory |
| `tests/test_*.py` (flat) | `tests/{darts,foretools,foreblocks/<tier>}/test_*.py` | test files split into subdirectories mirroring the `src/` package tiers |

Also fixed in this pass (packaging config, not a code move):

- `pyproject.toml`'s `[tool.setuptools.packages.find].include` was missing `mltracker`/`mltracker.*`
  despite the `mltracker-tui` console-script entry point depending on it — the wheel silently
  shipped a broken entry point. Now included.
- `MANIFEST.in`'s stale `recursive-include foreblocks/studio/dist *` (referring to Studio assets
  that moved to `apps/webui/dist` and are no longer package data) was removed.
- The CI-generated VitePress build under `site/docs/` (~11MB) was untracked from git — it's
  rebuilt from scratch by `.github/workflows/docs.yml` on every deploy. Only `site/landing/`
  (hand-authored source) stays tracked.
