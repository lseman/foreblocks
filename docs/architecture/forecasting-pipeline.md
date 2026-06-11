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
    A([Raw series\nT × D]) -->|fit_transform| B[TimeSeriesHandler]
    B --> C[X: N×T×F\ny: N×H×D]
    C -->|create_dataloaders| D[(DataLoader\ntrain / val / test)]

    D --> E[ForecastingModel\nhead + backbone]
    E --> F[Trainer]
    F -->|train loop| G{converged?}
    G -->|no| F
    G -->|yes| H[ModelEvaluator]

    H --> I[metrics\nMAE · RMSE · MAPE]
    H --> J[CV windows\nrolling evaluation]
    H --> K[learning curve\nplots]

    E -.->|optional| L[ConformalPredictionEngine\ncalibrate → predict intervals]
    E -.->|optional| M[DARTSSearcher\narchitecture search]
