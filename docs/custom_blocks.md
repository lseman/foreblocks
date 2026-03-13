# ForeBlocks Custom Blocks Guide

This guide covers the current `ForecastingModel` customization points for feature preprocessing, normalization, graph blocks, and output post-processing.

Related docs:
- [Documentation Overview](overview.md)
- [Getting Started](getting-started.md)
- [Transformer](transformer.md)
- [Preprocessor](preprocessor.md)
- [DARTS](darts.md)

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
```

### Block order

1. `input_preprocessor`
2. optional input skip connection
3. `input_normalization`
4. encoder/decoder (or direct head path)
5. `output_block`
6. `output_normalization`
7. `output_postprocessor`

---

## Add/replace blocks after model creation

Use `add_head(...)` to modify blocks in-place.

### Valid `position` values

- `encoder`
- `decoder`
- `attention`
- `input`
- `output`
- `input_norm`
- `output_norm`
- `head`

```python
import torch.nn as nn

model.add_head(nn.LayerNorm(64), position="input_norm")
model.add_head(nn.Sequential(nn.Linear(1, 1)), position="output")
```

Remove with:

```python
model.remove_head("output_norm")
```

Inspect with:

```python
print(model.list_heads())
```

---

## Graph block injection

Register graph modules at these stages:

- `pre_encoder`
- `post_encoder`
- `post_decoder`

```python
model.add_graph_block(graph_block=my_graph_layer, where="pre_encoder")
```

Expected graph block interface:
- inputs: `(x, adj)`
- `x` shape: `[B, T, N, F]` or `[B, N, F]`
- `adj` shape: `[N, N]` or `[B, N, N]`

---

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
```

---

## Practical recommendations

- Start with identity blocks and add one custom block at a time.
- Enable `input_skip_connection=True` when your preprocessor is aggressive.
- For long horizons, pair custom blocks with transformer backbones (`transformer_seq2seq`).
- Keep post-processing simple (monotonic constraints, scaling, clipping) and test it independently.

---

## Version notes

This page reflects the current `foreblocks.core.model.ForecastingModel` API.
If you are migrating from older examples that mention `TimeSeriesSeq2Seq`, switch to `ForecastingModel`.
