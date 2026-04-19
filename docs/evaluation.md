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
```

## Batched prediction

```python
import torch

X_test = torch.randn(200, 96, 7)   # (samples, input_len, channels)
preds = evaluator.predict(X_test, batch_size=256, use_amp=True)
# preds: (200, horizon, channels)
```

`use_amp=True` enables automatic mixed precision on CUDA; it is silently ignored on CPU.

## Point metrics

```python
y_test = torch.randn(200, 24, 7)   # (samples, horizon, channels)
metrics = evaluator.compute_metrics(X_test, y_test)

# {'mse': ..., 'rmse': ..., 'mae': ..., 'mape': ...}
print(metrics)
```

| Metric | Formula |
|---|---|
| MSE | mean((ŷ − y)²) |
| RMSE | √MSE |
| MAE | mean(|ŷ − y|) |
| MAPE | mean(|ŷ − y| / (|y| + ε)) × 100 |

## Rolling cross-validation

`cross_validation` slides a window of size `horizon` across the dataset and evaluates the already-trained model on each window.

```python
cv = evaluator.cross_validation(
    X=X_test,
    y=y_test,
    n_windows=10,
    horizon=24,
    step_size=None,   # defaults to horizon (non-overlapping)
    batch_size=256,
)

print(cv['overall'])          # aggregate metrics
print(cv['window_metrics'])   # list of per-window dicts
preds_all = cv['predictions'] # concatenated predictions tensor
```

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

```
pip install foreblocks[plotting]
```

### Cross-validation results

```python
fig = evaluator.plot_cv_results(cv, figsize=(15, 8))
fig.savefig("cv_results.png")
```

Produces a 2×2 grid: per-window MAE, RMSE, and MAPE curves with overall means, plus a text summary box.

### Learning curves

```python
fig = evaluator.plot_learning_curves(figsize=(15, 5))
```

Three subplots: train/val loss, learning rate schedule, and (if using distillation) task vs. distillation loss components.

## Training summary

```python
evaluator.print_summary()
```

Prints epoch count, final and best validation loss, and model size (parameter count and memory footprint in MB if the model exposes `get_model_size()`).
