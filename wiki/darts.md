# ForeBlocks DARTS Guide

Neural architecture search for time-series models with zero-cost screening, bilevel DARTS optimization, and final retraining.

Related docs:
- [Documentation Overview](overview.md)
- [Getting Started](getting-started.md)
- [Custom Blocks](custom_blocks.md)
- [Transformer](transformer.md)

---

## Quick start

```python
from foreblocks.darts import DARTSTrainer

trainer = DARTSTrainer(
    input_dim=5,
    hidden_dims=[32, 64, 128],
    forecast_horizon=24,
    seq_length=48,
    device="auto",   # resolves to cuda if available
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

---

## Search pipeline

1. Generate candidate architectures.
2. Rank candidates with zero-cost metrics.
3. Run `train_darts_model(...)` on top candidates.
4. Derive fixed architecture with `derive_final_architecture(...)`.
5. Train/evaluate final model with `train_final_model(...)`.

---

## Public API

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

Useful parameters: `ablation`, `n_random`, `random_sigma`, `seed`.

### `train_darts_model(...)`

```python
search = trainer.train_darts_model(
    model=candidate,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=30,
    arch_learning_rate=3e-3,
    model_learning_rate=1e-3,
    use_swa=False,
    use_bilevel_optimization=True,
)
```

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

---

## Default operation pool

If `all_ops` is not provided, the trainer uses:

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
- `Mamba`
- `InvertedAttention`

You can override with:

```python
trainer = DARTSTrainer(..., all_ops=["Identity", "TimeConv", "TCN"])
```

---

## Recommended settings

- Fast iteration: `num_candidates=8`, `search_epochs=8`, `top_k=3`.
- Balanced search: `num_candidates=20`, `search_epochs=20`, `top_k=5`.
- If unstable, lower `arch_learning_rate` and disable pruning first.

---

## Notes

- `multi_fidelity_search(...)` forwards extra keyword args to the orchestrator, so advanced knobs can be passed directly.
- Trainer methods return dictionaries with training curves/metrics; persist them for reproducible NAS reports.
