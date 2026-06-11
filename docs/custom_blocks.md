---
title: Custom Blocks Guide
description: ForecastingModel customization — preprocessing, normalization, output post-processing.
editLink: true
---


[[toc]]
# ForeBlocks Custom Blocks Guide

This guide covers the current `ForecastingModel` customization points for feature preprocessing, normalization, and output post-processing.

Related docs:
- [Documentation Overview](overview)
- [Getting Started](getting-started)
- [Transformer](transformer)
- [Preprocessor](preprocessor)
- [DARTS](darts)

---

## Core idea

`ForecastingModel` is a modular wrapper around an encoder/decoder/head stack.
You can inject processing blocks at stable extension points without rewriting training code.

### Supported forecasting strategies

- `seq2seq`
- `autoregressive`
- `direct`
- `transformer_seq2seq`

### Supported model types

- `lstm`
- `transformer`
- `informer-like`
- `head_only`

---

## Processing blocks (constructor)

```python
from foreblocks import ForecastingModel

model = ForecastingModel(
    encoder=...,
    decoder=...,
    forecasting_strategy="seq2seq",
    model_type="lstm",
    target_len=24,
    output_size=1,
    input_preprocessor=...,      # default: Identity
    input_normalization=...,     # default: Identity
    output_block=...,            # default: Identity
    output_normalization=...,    # default: Identity
    output_postprocessor=...,    # default: Identity
    input_skip_connection=False,
)
```text

Remove with:

```python
model.remove_head("output_norm")
```python

## Minimal custom preprocessor example

```python
import torch
import torch.nn as nn


class ConvPre(nn.Module):
    def __init__(self, in_features: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_features, hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden, in_features, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        y = x.transpose(1, 2)
        y = self.net(y)
        return y.transpose(1, 2)


model = ForecastingModel(
    encoder=...,
    decoder=...,
    forecasting_strategy="seq2seq",
    model_type="lstm",
    target_len=24,
    output_size=1,
    input_preprocessor=ConvPre(in_features=8, hidden=32),
    input_skip_connection=True,
)
