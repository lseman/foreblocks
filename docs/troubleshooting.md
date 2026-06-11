---
title: Troubleshooting
description: Common issues, error messages, and solutions.
editLink: true
---


[[toc]]
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
```toml

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

- [Custom Blocks Guide](custom_blocks)
- [Transformer Guide](transformer)

## `TimeSeriesHandler.transform(...)` does not work on a fresh instance

`transform(...)` uses fitted preprocessing state.

The normal order is:

```python
pre = TimeSeriesHandler(window_size=24, horizon=6)
X_train, y_train, processed, time_feat = pre.fit_transform(train_data)
X_val = pre.transform(val_data)
```toml

This is especially important when normalization, differencing, or detrending is enabled.

## Mixed precision causes issues on CPU or during debugging

For small local runs and debugging sessions, prefer:

```python
config = TrainingConfig(use_amp=False)
```bash

Then run:

```bash
npm run docs:dev
