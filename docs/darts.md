# ForeBlocks DARTS Guide

ForeBlocks includes a staged neural architecture search subsystem for time-series forecasting. The DARTS stack is built around `DARTSTrainer`, but it is not just a single differentiable loop. It combines:

- search-space sampling
- zero-cost candidate screening
- bilevel DARTS training on promoted candidates
- conversion to a discrete fixed model
- final retraining and optional result analysis

## Install

For the DARTS workflow itself, including the analyzer and richer search-result visuals:

```bash
pip install "foreblocks[darts]"
```

## Public imports

```python
from foreblocks.darts import (
    DARTSTrainer,
    DARTSConfig,
    DARTSTrainConfig,
    FinalTrainConfig,
    MultiFildelitySearchConfig,
    AblationSearchConfig,
    RobustPoolSearchConfig,
)
```

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
from foreblocks.darts import DARTSTrainer

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

This is the highest-level API. It runs candidate generation, ranking, short DARTS training, discrete derivation, and final retraining in one call.

## Mental model

The public entry point is `DARTSTrainer`, but the implementation is intentionally split into focused modules:

- `foreblocks/darts/search/zero_cost.py`: cheap candidate scoring
- `foreblocks/darts/training/darts_loop.py`: bilevel search training
- `foreblocks/darts/training/final_trainer.py`: fixed-model retraining
- `foreblocks/darts/search/multi_fidelity.py`: staged orchestration
- `foreblocks/darts/evaluation/analyzer.py`: post-search analysis

That means you can use the trainer in two styles:

- as one end-to-end NAS pipeline via `multi_fidelity_search(...)`
- as a lower-level orchestration layer where you call each phase manually

## Search phases

The multi-fidelity flow is:

1. generate `num_candidates` random architectures from the configured search space
2. score them with zero-cost metrics
3. keep the top `top_k`
4. run short bilevel DARTS training on the promoted candidates
5. derive a discrete architecture for each searched mixed model
6. select the best candidate by validation behavior
7. retrain the best fixed model and report final metrics

This is closer to a staged NAS pipeline than to a single bare DARTS loop.

## Core APIs

| Method | Use it when | Returns |
| --- | --- | --- |
| `evaluate_zero_cost_metrics(...)` | you want a cheap score before expensive training | weighted zero-cost metrics |
| `evaluate_zero_cost_metrics_raw(...)` | you want the underlying raw zero-cost signals | raw metrics without weighting |
| `train_darts_model(...)` | you want to search a single candidate with bilevel optimization | searched model, losses, alphas, metrics |
| `derive_final_architecture(...)` | you want a discrete model from a mixed searched one | fixed model |
| `train_final_model(...)` | you want to retrain a fixed model independently | final metrics and training info |
| `multi_fidelity_search(...)` | you want the end-to-end staged workflow | full search result dictionary |
| `ablation_weight_search(...)` | you want to study zero-cost weighting choices | ablation artifacts and rankings |
| `robust_initial_pool_over_op_pools(...)` | you want to test sensitivity to op-pool composition | robustness report |

### `evaluate_zero_cost_metrics(...)`

```python
metrics = trainer.evaluate_zero_cost_metrics(
    model=candidate,
    dataloader=val_loader,
    max_samples=32,
    num_batches=1,
    fast_mode=True,
)
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
search_results = trainer.train_darts_model(
    model=candidate,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    arch_learning_rate=3e-3,
    model_learning_rate=1e-3,
    use_bilevel_optimization=True,
)
```

This runs the differentiable search phase for a single candidate and returns a dictionary with the searched model, train/validation losses, alpha traces, and final metrics.

### `derive_final_architecture(...)`

```python
fixed_model = trainer.derive_final_architecture(search_results["model"])
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

### `multi_fidelity_search(...)`

```python
results = trainer.multi_fidelity_search(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    num_candidates=10,
    search_epochs=10,
    final_epochs=100,
    max_samples=32,
    top_k=5,
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

The config dataclasses in `foreblocks.darts.config` map cleanly onto the staged workflow:

| Config | What it controls |
| --- | --- |
| `DARTSSearchSpaceConfig` | op pool, architecture modes, hidden dims, cells, nodes, family grouping |
| `DARTSTrainConfig` | bilevel search phase, regularization, pruning, temperature schedule |
| `FinalTrainConfig` | fixed-model retraining budget and optimizer behavior |
| `MultiFildelitySearchConfig` | candidate count, promotion size, epoch budgets, worker/stats settings |
| `AblationSearchConfig` | zero-cost weighting ablations |
| `RobustPoolSearchConfig` | sensitivity to alternative op-pool definitions |

### Default search-space ingredients

The default architecture modes are:

- `encoder_decoder`
- `encoder_only`
- `decoder_only`
- `mamba`

The default operation pool includes:

- `Identity`
- `TimeConv`
- `GRN`
- `Wavelet`
- `Fourier`
- `TCN`
- `ResidualMLP`
- `ConvMixer`
- `MultiScaleConv`
- `PyramidConv`
- `PatchEmbed`
- `InvertedAttention`
- `DLinear`
- `TimeMixer`
- `NBeats`
- `TimesNet`

Important note:

- `mamba` is represented as an architecture mode rather than as a normal operation in the default op list

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

- [Run A DARTS Search](tutorials/darts-multifidelity-search.md)
- [DARTS Search Pipeline](architecture/darts-pipeline.md)
- [Transformer Guide](transformer.md)
- [Custom Blocks Guide](custom_blocks.md)
- [Troubleshooting](troubleshooting.md)
