# ForeBlocks DARTS Guide

ForeBlocks includes a substantial neural architecture search subsystem for time-series forecasting.

The current DARTS stack combines:

- random candidate generation
- zero-cost screening
- bilevel DARTS training on promoted candidates
- architecture derivation
- final retraining of the best discrete model

Related docs:

- [Documentation Overview](overview.md)
- [Getting Started](getting-started.md)
- [Custom Blocks](custom_blocks.md)
- [Transformer](transformer.md)

## Import

```python
from foreblocks.darts import DARTSTrainer
```

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
    top_k=5,
)

best_model = results["final_model"]
trainer.save_best_model("best_darts_model.pth")
```

## Mental model

The public entry point is `DARTSTrainer`, but the search pipeline is split internally into focused modules:

- search-space configuration
- zero-cost candidate scoring
- bilevel training
- finalization into a discrete architecture
- final training and evaluation

You can use the trainer at two levels:

- as a single end-to-end search pipeline via `multi_fidelity_search(...)`
- as a lower-level orchestration layer if you want to inspect and tune each phase manually

## Search pipeline

The current multi-fidelity flow is:

1. randomly generate `num_candidates` candidate architectures
2. evaluate them with zero-cost metrics
3. keep the top `top_k`
4. run short DARTS bilevel training on promoted candidates
5. derive a discrete architecture from each searched mixed model
6. select the best candidate by validation behavior
7. retrain the best final model

The multi-fidelity implementation is closer to a staged NAS pipeline than to a single bare DARTS loop.

## Core public methods

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

Use this when you want to compare candidate architectures cheaply before any expensive training.

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

This runs the differentiable search phase on a single candidate and returns a dictionary containing the searched model, losses, alpha traces, and final metrics.

### `derive_final_architecture(...)`

```python
fixed_model = trainer.derive_final_architecture(search_results["model"])
```

This converts the mixed architecture into a discrete architecture suitable for final training.

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

Use this when you want to retrain a fixed architecture independently of the full search pipeline.

### `multi_fidelity_search(...)`

```python
results = trainer.multi_fidelity_search(
    train_loader,
    val_loader,
    test_loader,
    num_candidates=10,
    search_epochs=10,
    final_epochs=100,
    max_samples=32,
    top_k=5,
)
```

This is the easiest end-to-end API. The returned dictionary includes:

- `final_model`
- `candidates`
- `top_candidates`
- `trained_candidates`
- `best_candidate`
- `final_results`
- `search_config`

## Search-space controls

The search subsystem exposes configuration dataclasses in `foreblocks.darts.config`.

Important groups:

### `DARTSSearchSpaceConfig`

Controls:

- operation pool
- architecture modes
- hidden-dimension choices
- number of cells and nodes
- family-level operation grouping

The current default architecture modes are:

- `encoder_decoder`
- `encoder_only`
- `decoder_only`
- `mamba`

### `DARTSTrainConfig`

Controls the bilevel search phase:

- search epochs
- architecture and model learning rates
- weight decay
- warmup epochs
- architecture update frequency
- pruning / progressive shrinking
- temperature schedule
- extra regularization terms

### `FinalTrainConfig`

Controls retraining of the fixed final model:

- epochs
- learning rate
- weight decay
- patience
- one-cycle scheduling

### `MultiFildelitySearchConfig`

Controls the staged NAS pipeline:

- number of candidates
- short-search epoch budget
- final-training epoch budget
- top-k promotion size
- stats collection and worker settings

## Default operation pool

The current default operation pool includes:

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

- `mamba` is represented as an architecture mode, not as a regular op in the default op list

## Candidate generation

Candidate generation in the orchestrator selects:

- operation subsets
- hidden dimension
- number of cells
- number of nodes
- architecture mode
- transformer self-attention type
- FFN variant

The search is therefore not limited to simple op-edge selection. It can alter higher-level modeling structure too.

## Practical tuning advice

### Fast iteration

Use:

- `num_candidates=8`
- `search_epochs=8`
- `top_k=3`
- smaller `hidden_dims`

This is useful for debugging the search loop itself.

### Balanced search

Use:

- `num_candidates=20`
- `search_epochs=20`
- `top_k=5`

This is the most reasonable starting point for a serious experiment.

### When the search is unstable

Try this order:

1. lower `arch_learning_rate`
2. increase `warmup_epochs`
3. disable or delay progressive shrinking
4. reduce regularization complexity
5. reduce search-space breadth

## Bilevel search details

The current `train_darts_model(...)` surface includes several advanced controls:

- `warmup_epochs`
- `architecture_update_freq`
- `use_bilevel_optimization`
- `progressive_shrinking`
- hybrid pruning controls
- Hessian penalty controls
- edge diversity and identity-cap regularization
- edge sharpening schedule
- `arch_grad_ema_beta`
- `beta_darts_weight`
- `moe_balance_weight`
- `transformer_exploration_weight`

This means the search loop is already tuned for more than plain textbook DARTS. It mixes bilevel search with several stabilizing regularizers and pruning heuristics.

## Results and persistence

Useful post-search utilities:

- `trainer.save_best_model(...)`
- `trainer.plot_search_summary(...)` style methods if you build around the returned artifacts
- `StreamlinedDARTSAnalyzer` for inspecting search results

The saved best-model checkpoint includes:

- final model state
- candidate config
- final metrics

## Recommended workflow

1. Confirm your data loaders and forecasting task work without NAS.
2. Run a small `multi_fidelity_search(...)`.
3. Inspect top candidates and promoted candidates.
4. Save the best final model.
5. Only then expand the search space or final training budget.

## Troubleshooting

- Search is too slow: reduce `num_candidates`, `search_epochs`, and operation breadth first.
- Zero-cost ranking looks noisy: increase `max_samples` slightly and reduce ablation randomness.
- Promoted candidates all look similar: narrow the op pool less aggressively or revisit family diversity settings.
- Bilevel optimization is unstable: lower architecture LR, add more warmup, and simplify pruning.
- Final model underperforms search-time expectations: retrain from scratch and inspect whether the discrete architecture differs sharply from the mixed model.

## Related pages

- [Transformer Guide](transformer.md)
- [Public API](reference/public-api.md)
- [Repository Map](reference/repository-map.md)
