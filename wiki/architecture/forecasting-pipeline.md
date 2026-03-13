# Forecasting Pipeline

This page describes the main execution flow for a typical `foreblocks` forecasting workload.

## Pipeline stages

1. data preparation
2. dataloader construction
3. model assembly
4. training
5. evaluation
6. optional calibration, search, or synthetic-data loops

## Data preparation

You can start from either:

- pre-windowed tensors or NumPy arrays
- a raw multivariate time series shaped `[T, D]`

If you start from raw series data, `TimeSeriesHandler` can create:

- processed series
- input windows
- forecast targets
- optional time features

## Model assembly

The central abstraction is `ForecastingModel`.

Common paths:

- direct forecasting with a custom `head`
- recurrent seq2seq with encoder/decoder blocks
- transformer seq2seq with transformer backbones

The direct path is the simplest. Decoder-based paths require tighter agreement between encoder outputs, decoder inputs, and target/output dimensions.

## Training

`Trainer` is the orchestration layer for:

- optimization
- validation
- early stopping
- scheduler stepping
- optional MLTracker integration
- optional NAS handling
- optional conformal machinery

For local smoke tests, pass `auto_track=False`.

## Evaluation

`ModelEvaluator` provides:

- prediction batching
- metrics computation
- rolling-window cross-validation helpers
- learning-curve plots

## Optional branches

- use `foretools/tsgen` to create synthetic datasets
- use `foreblocks/darts` to search over architectures
- use conformal support in `Trainer` if you need intervals

## Related pages

- [Train A Direct Model](../tutorials/train-direct-model.md)
- [Preprocessor Guide](../preprocessor.md)
- [DARTS Guide](../darts.md)
