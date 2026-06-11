---
title: Evaluation & Metrics
description: Model evaluation, metrics, and benchmarking utilities.
editLink: true
---


[[toc]]
# Evaluation & Metrics

`foreblocks.evaluation.ModelEvaluator` wraps a trained `Trainer` and provides batched inference, rolling cross-validation, loss curve plotting, and a human-readable training summary.

## Setup

`ModelEvaluator` takes a `Trainer` instance directly — it reuses the model, device, and training history stored on it.

```python
from foreblocks.training import Trainer
from foreblocks.evaluation import ModelEvaluator

trainer = Trainer(model, ...)
trainer.fit(train_loader, val_loader, epochs=50)

evaluator = ModelEvaluator(trainer)
```python

`use_amp=True` enables automatic mixed precision on CUDA; it is silently ignored on CPU.

## Point metrics

```python
y_test = torch.randn(200, 24, 7)   # (samples, horizon, channels)
metrics = evaluator.compute_metrics(X_test, y_test)

# {'mse': ..., 'rmse': ..., 'mae': ..., 'mape': ...}
print(metrics)
```text

Return dict keys:

| Key | Type | Description |
|---|---|---|
| `overall` | `dict` | Aggregate MAE / RMSE / MAPE / MSE over all windows |
| `window_metrics` | `list[dict]` | Per-window metrics including `start_idx` / `end_idx` |
| `predictions` | `Tensor` | Concatenated predictions across all windows |
| `targets` | `Tensor` | Concatenated targets across all windows |
| `n_windows` | `int` | Number of windows actually evaluated |
| `total_points` | `int` | Total sample count |

::: info Model is not retrained per fold
This is a walk-forward evaluation of a fixed model, not k-fold retraining. Use it to assess generalisation across temporal shifts, not for model selection.
:::

## Plots

All plotting methods require `matplotlib`:

pip install foreblocks[plotting]

### Cross-validation results

```python
fig = evaluator.plot_cv_results(cv, figsize=(15, 8))
fig.savefig("cv_results.png")
```toml

Three subplots: train/val loss, learning rate schedule, and (if using distillation) task vs. distillation loss components.

## Training summary

```python
evaluator.print_summary()
