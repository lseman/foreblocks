---
title: BOHB Search
description: Budgeted black-box optimization with Hyperband-style successive halving and configurable TPE.
editLink: true
---


[[toc]]
# BOHB Search

`foretools/bohb` provides budgeted black-box optimization built on Hyperband-style successive halving and a multi-fidelity TPE proposer. Serial runs use synchronous successive halving; `parallel_jobs > 1` uses ASHA.

The default sampler (`sampler="mftpe"`) is a vectorized multivariate TPE. It is fit on the largest budget with enough results, and batch proposals are spread with a constant liar. The original per-parameter TPE, with trust regions, qNEI/Thompson batching and a GP surrogate, is still available as `sampler="legacy"` and is configured through `TPEConf`.

Use it when you want to tune hyperparameters, compare search strategies, or benchmark objective functions outside the differentiable architecture-search flow covered by `darts`.

## Import surface

```python
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

## Config space format

The search space is a dictionary of parameter names to tuple specs.

Supported parameter types:

| Type | Example | Meaning |
| --- | --- | --- |
| `float` | `("float", (0.0, 1.0))` | continuous uniform range |
| `float` with log scaling | `("float", (1e-5, 1e-1, "log"))` | continuous log-uniform style range |
| `int` | `("int", (16, 256))` | integer range |
| `choice` | `("choice", ["adam", "adamw", "sgd"])` | categorical choice |

Any entry can take a third options element to make it conditional, for example `("float", (0.0, 0.99), {"condition": {"parent": "opt", "values": ["sgd"]}})`.

## Sampler options

```python
bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    sampler_options={"n_ei_candidates": 64, "random_fraction": 0.1, "bandwidth_factor": 1.5},
)
```

## Objective function signatures

`BOHB` accepts either of these objective signatures:

- `objective(config, budget) -> float`
- `objective(config, budget, trial) -> float`

Use the three-argument form when you want intermediate reporting and early pruning.

### Example with `Trial.report()`

```python
from foretools.bohb.trial import TrialPruned

def objective(config, budget, trial):
    loss = 1.0
    for epoch in range(int(budget)):
        loss *= 0.85
        trial.report(epoch, loss)
    return float(loss)

try:
    best_config, best_loss = BOHB(
        config_space={"width": ("int", (32, 256))},
        evaluate_fn=objective,
        min_budget=1,
        max_budget=9,
    ).run()
except TrialPruned:
    pass
```

`BOHB` catches `TrialPruned` itself, so the `try` block is optional. A pruned trial's last reported loss is still passed to the sampler.

With `sampler="legacy"`, `tpe_overrides={...}` is lighter than constructing a full `TPEConf`.

## Pruning configuration

`PruningConfig` exposes the pruning thresholds that BOHB uses for both completed evaluations and intermediate `Trial.report()` calls. A completed evaluation that is "pruned" is still recorded; it is only kept from being promoted to the next budget.

```python
from foretools.bohb import BOHB, PruningConfig

pruning = PruningConfig(
    final_min_history=12,
    final_prob_base_balanced=0.55,
    step_min_history=10,
    step_progress_tolerance=0.10,
    step_quantile_balanced=0.95,
)

bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    pruning_conf=pruning,
)
```

## Inspecting results

The BOHB instance keeps a full optimization history plus ranked configurations.

```python
history = bohb.get_optimization_history()
top_configs = bohb.get_top_configs(5)
```

## Plotting utilities

`OptimizationPlotter` builds lightweight analysis charts directly from a BOHB run.

```python
from foretools.bohb.plotter import OptimizationPlotter

plotter = OptimizationPlotter.from_bohb(bohb)
plotter.plot_optimization_history()
plotter.plot_budget_vs_loss()
plotter.plot_bracket_best()
plotter.plot_param_effect("lr")
plotter.plot_param_importance(top_k=5)
plotter.plot_parallel_coordinates()
```
