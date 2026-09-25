---
title: DARTS Guide
description: Differentiable neural architecture search for time-series forecasting.
editLink: true
---


[[toc]]
# ForeBlocks DARTS Guide

ForeBlocks includes a staged neural architecture search subsystem for time-series forecasting. The DARTS stack is built around `DARTSTrainer`, but it is not just a single differentiable loop. It combines:

- search-space sampling
- zero-cost candidate screening
- bilevel DARTS training on promoted candidates
- conversion to a discrete fixed model
- final retraining and optional result analysis

## Install

DARTS requires the `darts` extra plus the base foreblocks package:

```bash
pip install "foreblocks[darts]"
```

For foreblocks imports (`ForecastingModel`, `Trainer`, `create_dataloaders`, etc.) see the [Imports Reference](shared/imports).

## When to use DARTS here

Use the DARTS subsystem when:

- your operation pool is already fairly well defined, but you do not know which combinations work best
- you want a cheaper staged NAS workflow instead of fully training every candidate
- you want to inspect alpha evolution, promoted candidates, and final retrained metrics

Do not start here if:

- your base training loop is not working yet
- you have not verified tensor shapes with a normal `ForecastingModel` run
- you are still deciding between totally different problem formulations

## Quick start

```python
from darts import DARTSTrainer

trainer = DARTSTrainer(
    input_dim=5,
    hidden_dims=[32, 64, 128],
    forecast_horizon=24,
    seq_length=48,
    device="auto",
)

results = trainer.multi_fidelity_search(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_candidates=20,
    search_epochs=20,
    final_epochs=80,
    max_samples=32,
    top_k=5,
)

best_model = results["final_model"]
trainer.save_best_model("best_darts_model.pth")
```

Useful knobs:

- `max_samples`
- `num_batches`
- `fast_mode`
- `ablation`
- `n_random`
- `random_sigma`
- `seed`

### `train_darts_model(...)`

```python
from darts import DARTSTrainConfig

train_config = DARTSTrainConfig(
    epochs=30,
    arch_learning_rate=3e-3,
    model_learning_rate=1e-3,
    use_bilevel_optimization=True,
)
search_results = trainer.train_darts_model(
    model=candidate,
    train_loader=train_loader,
    val_loader=val_loader,
    train_config=train_config,
)
```

Use this after the mixed architecture has converged enough that you want a discrete artifact for final training or export.

### `train_final_model(...)`

```python
final_results = trainer.train_final_model(
    model=fixed_model,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    epochs=100,
)
```

## Result structure

The end-to-end search dictionary includes:

- `final_model`
- `candidates`
- `top_candidates`
- `trained_candidates`
- `best_candidate`
- `final_results`
- `search_config`
- `stats` when `collect_stats=True`

In practice, the most useful objects to inspect are:

- `results["best_candidate"]`
- `results["final_results"]["final_metrics"]`
- `results["trained_candidates"]`
- `results["candidates"]` if you are debugging promotion behavior

## Configuration surface

The config dataclasses in `darts.config` map cleanly onto the staged workflow:

| Config | What it controls |
| --- | --- |
| `DARTSSearchSpaceConfig` | op pool, architecture modes, hidden dims, cells, nodes, family grouping |
| `DARTSTrainConfig` | bilevel search phase, regularization, pruning, temperature schedule |
| `FinalTrainConfig` | fixed-model retraining budget and optimizer behavior |
| `MultiFidelitySearchConfig` | candidate count, promotion size, epoch budgets, worker/stats settings |
| `AblationSearchConfig` | zero-cost weighting ablations |
| `RobustPoolSearchConfig` | sensitivity to alternative op-pool definitions |

### Default search-space ingredients

The default architecture modes are:

- `encoder_decoder`
- `encoder_only`
- `decoder_only`

The default operation pool (`darts.config.DEFAULT_OPS`, grouped by family in
`DEFAULT_OP_FAMILIES`) includes:

- `Identity`
- `ResidualMLP`, `GRN`, `TimeMixer`, `NBeats` (`mlp` family)
- `TimeConv`, `TCN`, `ConvMixer`, `MultiScaleConv`, `PyramidConv` (`conv` family)
- `Fourier`, `Wavelet`, `DLinear`, `TimesNet` (`frequency` family)
- `PatchEmbed`, `InvertedAttention` (`attention` family)
- `SwiGLU`, `GeGLU`, `GatedGELU` (`gated_ffn` family)
- `SSM` (`ssm` family — a lightweight S4D-style diagonal state-space block)

Every operation's constructor, family, and efficiency prior is defined once
in `darts.architecture.ops.registry.OP_REGISTRY`; the search-space config and
`MixedOp` both derive from it, so it is the single place to add a new op.

## Recommended tuning order

When the search behaves poorly, change things in this order:

1. shrink or clarify the search space
2. verify zero-cost metrics are ranking plausible candidates
3. reduce `top_k` and search budgets only after phase 2 looks sensible
4. tune bilevel parameters like `arch_learning_rate`, `model_learning_rate`, and `architecture_update_freq`
5. only then add more advanced regularization or pruning rules

Useful early knobs:

- `hidden_dims`
- `cell_range`
- `node_range`
- `min_ops`
- `max_ops`
- `num_candidates`
- `top_k`
- `search_epochs`

## Practical tips

- Run a tiny end-to-end search first, even if the result quality is poor. That verifies your search loop, result structure, and save/load path.
- Keep `num_candidates`, `top_k`, and `search_epochs` small while you are validating the search space.
- Use `evaluate_zero_cost_metrics(...)` directly before trusting a longer search run.
- Treat `collect_stats=True` as a benchmarking tool, not a default setting for every run.
- Save the best discrete model with `trainer.save_best_model(...)` once you have a search worth keeping.

## Related pages

- [Run A DARTS Search](tutorials/darts-multifidelity-search)
- [DARTS Search Pipeline](architecture/darts-pipeline)
- [Transformer Guide](transformer)
- [Custom Blocks Guide](custom_blocks)
- [Imports Reference](shared/imports)
- [Troubleshooting](troubleshooting)
