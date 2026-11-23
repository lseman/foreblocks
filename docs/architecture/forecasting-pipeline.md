---
title: Forecasting Pipeline
description: End-to-end execution flow for a typical foreblocks forecasting workload.
editLink: true
---

# Forecasting Pipeline

This page describes the main execution flow for a typical `foreblocks` forecasting workload.

## End-to-end flow

```mermaid
flowchart TD
    A[Raw series] --> B[TimeSeriesHandler]
    B --> C[Prepared inputs and targets]
    C -->|create_dataloaders| D[Training and validation loaders]
    E[ForecastingModel] --> F[Trainer]
    D -->|train| F
    F --> G[ModelEvaluator]
    G --> H[Metrics and rolling cross-validation]
    G --> I[Learning curves]
    F -->|optional calibration| J[Conformal prediction intervals]
```

Inputs typically have shape `[samples, history, features]` and targets have
shape `[samples, horizon, outputs]`. Pass the prepared arrays to
`create_dataloaders`, construct `Trainer(model)`, and call `trainer.train`
with the loaders. Architecture search has its own training workflow.
[DARTS search pipeline](darts-pipeline.md)
