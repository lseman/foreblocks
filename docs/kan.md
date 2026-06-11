---
title: KAN Backbone
description: Kolmogorov-Arnold Network backbone for time-series forecasting.
editLink: true
---


[[toc]]
# KAN Backbone

ForeBlocks ships a Kolmogorov–Arnold Network (KAN) backbone for time-series
forecasting in `foreblocks.kan`. Instead of fixed activation functions on
neurons, KAN places learnable, basis-expanded functions on the edges of the
network. This implementation patches the input series, expands each patch
through one or more **orthogonal polynomial families**, and optionally routes
between families with a token-level mixture-of-experts (MoKAN) router.

The KAN backbone is a specialist subsystem: it is **not** part of the top-level
`foreblocks` public API. Import it directly from `foreblocks.kan`.

## When to use it

- You want a non-Transformer, non-recurrent backbone with strong inductive bias
  for smooth/periodic structure.
- You want to mix several basis families (e.g. Chebyshev + Fourier + wavelet)
  and let a router pick per token.
- You are comparing alternative function approximators against the standard
  forecasting stack.

If your baseline `ForecastingModel` run is not working yet, start there first —
see [Getting Started](getting-started).

## Minimal example

`KANModel` consumes a `[B, T, C]` series and returns a `[B, H, C]` forecast,
where `T = context_window`, `H = target_window`, and `C = c_in`.

```python
import torch
from foreblocks.kan import KANModel

model = KANModel(
    c_in=4,              # number of input channels (features)
    context_window=48,   # input length T
    target_window=24,    # forecast horizon H
    patch_len=16,        # patch size
    stride=8,            # patch stride
    d_model=64,
    depth=2,             # number of KAN blocks
    revin=True,          # reversible instance norm on inputs
)

x = torch.randn(8, 48, 4)   # [B, T, C]
y = model(x)                # [B, 24, 4] -> [B, H, C]
```python

Pass a subset via the `families` argument; per-family hyperparameters are
exposed as keyword arguments (for example `jacobi_alpha`, `jacobi_beta`,
`wavelet_num`, `fourier_base_freq`). Fine-grained layer behaviour can be
configured with `PolyLayerConfig`.

## Mixture of families (MoKAN routing)

When multiple families are active, a token-level router (`TokenRouter`,
configured with `RouterConfig`) selects the top-`k` families per token. The
relevant `KANModel` knobs are:

- `top_k` — number of families selected per token (default `2`)
- `router_temperature`, `router_hidden` — router softmax temperature and width
- `load_balance_coef` — auxiliary load-balancing loss weight

## Key constructor arguments

| Argument | Purpose |
| --- | --- |
| `c_in`, `context_window`, `target_window` | input channels, input length, forecast horizon |
| `patch_len`, `stride`, `padding_patch` | patching of the input series |
| `d_model`, `depth` | hidden width and number of KAN blocks |
| `families` | sequence of `PolyFamily` instances to expand through |
| `revin`, `affine`, `subtract_last` | reversible instance normalization options |
| `top_k`, `router_*`, `load_balance_coef` | MoKAN routing controls |
| `head_*`, `block_*`, `final_norm` | forecast head and block regularization |

## Related pages

- [Transformer Guide](transformer) — the standard backbone and attention variants
- [Hybrid Mamba Guide](hybrid-mamba) — the SSM-based backbone
- [Getting Started](getting-started)
