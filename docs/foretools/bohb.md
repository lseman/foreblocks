# BOHB Search

`foretools/bohb` provides budgeted black-box optimization built around Hyperband-style successive halving and a configurable TPE proposer.

Use it when you want to tune hyperparameters, compare search strategies, or benchmark objective functions outside the differentiable architecture-search flow covered by `foreblocks.darts`.

## Import surface

```python
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

## Core BOHB workflow

```python
from foretools.bohb import BOHB

config_space = {
    "lr": ("float", (1e-5, 1e-1, "log")),
    "hidden": ("int", (32, 256)),
    "dropout": ("float", (0.0, 0.5)),
    "optimizer": ("choice", ["adam", "adamw", "sgd"]),
}

def objective(config, budget):
    lr_penalty = (config["lr"] - 1e-2) ** 2
    hidden_bonus = abs(config["hidden"] - 128) / 256
    dropout_penalty = abs(config["dropout"] - 0.2)
    optimizer_penalty = 0.0 if config["optimizer"] == "adamw" else 0.1
    fidelity_gain = 1.0 / max(budget, 1.0)
    return float(lr_penalty + hidden_bonus + dropout_penalty + optimizer_penalty + fidelity_gain)

bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    min_budget=1,
    max_budget=27,
    eta=3,
    n_iterations=5,
    verbose=True,
)

best_config, best_loss = bohb.run()
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

In practice you normally catch `TrialPruned` inside the objective and return a fallback loss or let BOHB's error handling manage it.

## Main BOHB controls

| Argument | Role |
| --- | --- |
| `min_budget`, `max_budget` | define the resource range used by Hyperband |
| `eta` | successive-halving downsampling factor |
| `n_iterations` | number of outer BOHB iterations |
| `parallel_jobs` | concurrent evaluations via thread pool |
| `early_prune` | enables trial-level pruning logic |
| `handle_errors` | converts objective failures into skipped trials instead of crashing |
| `prior_trials_jsonl` | warm-start source for historical observations |
| `history_export_jsonl` | export completed evaluations for reuse |
| `tpe_conf`, `tpe_overrides` | control the TPE sampler behavior |
| `pruning_conf`, `pruning_overrides` | tune final-loss and `Trial.report()` pruning thresholds |

## TPE configuration

`TPEConf` groups the large TPE configuration surface into sections:

- `gamma`: good-vs-bad split behavior and exploration pressure
- `bandwidth`: KDE smoothing and local bandwidth options
- `prior`: prior mixing and startup behavior
- `constraints`: hard and soft constraints, repair attempts, rejection limits
- `batch`: batched candidate selection behavior
- `trust_region`: trust-region-style local search controls

Example:

```python
from foretools.bohb import BOHB, TPEConf

conf = TPEConf(
    gamma={"gamma": 0.15, "gamma_strategy": "sqrt", "dual_gamma": False},
    bandwidth={"bandwidth_factor": 0.7, "local_bandwidth": True, "local_bandwidth_k": 7},
    trust_region={"trust_region_enabled": True, "trust_region_init_length": 1.0},
)

bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    max_budget=27,
    min_budget=1,
    tpe_conf=conf,
)
```

If you only need a few overrides, `tpe_overrides={...}` is lighter than constructing a full config object.

## Pruning configuration

`PruningConfig` exposes the pruning thresholds that BOHB uses for both completed evaluations and intermediate `Trial.report()` calls.

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

For smaller adjustments, `pruning_overrides={...}` is usually enough:

```python
bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    pruning_overrides={
        "step_min_history": 12,
        "step_quantile_balanced": 0.94,
    },
)
```

## Inspecting results

The BOHB instance keeps a full optimization history plus ranked configurations.

```python
history = bohb.get_optimization_history()
top_configs = bohb.get_top_configs(5)
```

You can also export the history:

```python
bohb.save_history_jsonl("bohb_history.jsonl")
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

These plots are useful for answering different questions:

- `plot_optimization_history()`: is the search still improving?
- `plot_budget_vs_loss()`: does more budget systematically help?
- `plot_bracket_best()`: which Hyperband brackets carried the best results?
- `plot_param_effect()`: how does one parameter relate to loss?
- `plot_param_importance()`: which parameters matter most by a simple proxy?
- `plot_parallel_coordinates()`: what do the good regions look like jointly?

## BOHB vs DARTS

Use `foretools/bohb` when:

- the search space is black-box or mixed discrete/continuous
- the objective is an arbitrary training script or evaluation function
- you want multi-fidelity tuning across epochs, data size, or training budget

Use `foreblocks.darts` when:

- you are searching over differentiable neural architecture choices inside the model stack
- you need bilevel architecture optimization rather than outer-loop black-box tuning

## Practical guidance

- Start with a cheap `max_budget` that still separates bad from good configurations.
- Keep `parallel_jobs=1` first so objective behavior is easy to debug.
- Use log-scaled floats for learning rates and weight decays.
- Tune `PruningConfig` conservatively at first; pruning is still workload-specific and can bias search if thresholds are too tight.
- When documenting results or transferring runs, export the history JSONL and keep the exact config space definition with it.

## Related pages

- [Optimize With BOHB](../tutorials/optimize-with-bohb.md)
- [DARTS Guide](../darts.md)
- [Foretools Overview](index.md)
