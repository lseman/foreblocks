# Troubleshooting

This page collects the most common first-run issues for `foreblocks`.

## Import errors for optional packages

Some subsystems require optional extras that are not part of the base install.

| Symptom | Likely fix |
| --- | --- |
| Plotting helpers complain about `matplotlib` | `pip install "foreblocks[plotting]"` |
| `TimeSeriesHandler` or preprocessing utilities fail to import scientific packages | `pip install "foreblocks[preprocessing]"` |
| DARTS trainer, analyzer, or search visuals fail to import | `pip install "foreblocks[darts]"` |
| MLTracker API or UI modules fail to import | `pip install "foreblocks[mltracker]"` |
| VMD utilities or Optuna search features fail to import | `pip install "foreblocks[vmd]"` |
| Wavelet-related blocks are unavailable | `pip install "foreblocks[wavelets]"` |
| External benchmark integrations fail to import | `pip install "foreblocks[benchmark]"` |
| Changepoint-detection helpers are unavailable | `pip install "foreblocks[foreminer]"` |

If you want every optional runtime dependency:

```bash
pip install "foreblocks[all]"
```

## Training starts MLTracker when you only want a local smoke test

`Trainer` can initialize MLTracker automatically.

Disable it for quick local checks:

```python
trainer = Trainer(model, config=TrainingConfig(use_amp=False), auto_track=False)
```

## Shape mismatch in the direct forecasting path

The simplest path expects:

- `X`: `[N, T, F]`
- direct head output: same shape as the target tensor

If your head returns `[N, horizon]`, your `y` should also be `[N, horizon]`.

## Shape mismatch in seq2seq or transformer workflows

Most encoder/decoder workflows use:

- inputs: `[N, T, F]`
- targets: `[N, H, D]`

If you are building custom encoder or decoder blocks, verify the hidden size, target horizon, and feature dimension contracts before debugging the training loop.

Relevant guides:

- [Custom Blocks Guide](custom_blocks.md)
- [Transformer Guide](transformer.md)

## `TimeSeriesHandler.transform(...)` does not work on a fresh instance

`transform(...)` uses fitted preprocessing state.

The normal order is:

```python
pre = TimeSeriesHandler(window_size=24, horizon=6)
X_train, y_train, processed, time_feat = pre.fit_transform(train_data)
X_val = pre.transform(val_data)
```

Call `fit_transform(...)` first, then reuse the same instance for validation or test data.

## Predictions are still normalized or transformed

If you trained on preprocessed outputs, map predictions back with:

```python
pred_real = pre.inverse_transform(pred_scaled)
```

This is especially important when normalization, differencing, or detrending is enabled.

## Mixed precision causes issues on CPU or during debugging

For small local runs and debugging sessions, prefer:

```python
config = TrainingConfig(use_amp=False)
```

That removes AMP from the equation and makes failures easier to interpret.

## Docs build errors locally

The docs site uses `mkdocs-material` plus `pymdownx` Markdown extensions.

Install the docs dependency first:

```bash
pip install mkdocs-material
```

Then run:

```bash
mkdocs serve
```

## Still not sure where to start

Use this order:

1. [Getting Started](getting-started.md)
2. [Public API](reference/public-api.md)
3. [Configuration](reference/configuration.md)
4. The subsystem guide closest to your workflow
