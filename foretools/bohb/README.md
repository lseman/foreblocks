# BOHB

This directory contains a BOHB-style optimizer that combines Hyperband resource allocation with a configurable TPE sampler.

The canonical documentation now lives in the repository wiki:

- `wiki/foretools/bohb.md`
- `wiki/tutorials/optimize-with-bohb.md`

This README keeps a short, code-accurate reference close to the implementation.

## Import surface

```python
from foretools.bohb import BOHB, PruningConfig, TPEConf
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import TrialPruned
```

## Quick start

```python
from foretools.bohb import BOHB

config_space = {
    "lr": ("float", (1e-5, 1e-1, "log")),
    "batch_size": ("choice", [16, 32, 64, 128]),
    "dropout": ("float", (0.0, 0.5)),
    "hidden": ("int", (16, 256)),
}

def objective(config, budget):
    lr_penalty = abs(config["lr"] - 1e-2)
    batch_penalty = 0.0 if config["batch_size"] == 64 else 0.1
    dropout_penalty = abs(config["dropout"] - 0.2)
    hidden_penalty = abs(config["hidden"] - 128) / 128
    budget_bonus = 1.0 / max(budget, 1.0)
    return float(lr_penalty + batch_penalty + dropout_penalty + hidden_penalty + budget_bonus)

bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    min_budget=1,
    max_budget=27,
    eta=3,
    n_iterations=4,
    verbose=True,
)

best_config, best_loss = bohb.run()
print(best_config, best_loss)
```

## Objective signatures

Supported objective forms:

- `objective(config, budget) -> float`
- `objective(config, budget, trial) -> float`

The three-argument form enables `Trial.report(step, loss)` and BOHB-side pruning.

## Config space

Supported parameter specs:

- `("float", (lo, hi))`
- `("float", (lo, hi, "log"))`
- `("int", (lo, hi))`
- `("choice", [v1, v2, ...])`

## Results and plotting

```python
history = bohb.get_optimization_history()
top_configs = bohb.get_top_configs(5)

plotter = OptimizationPlotter.from_bohb(bohb)
plotter.plot_optimization_history()
plotter.plot_budget_vs_loss()
plotter.plot_param_importance()
```

## Notes

- `run()` takes no arguments. Configure `n_iterations` when constructing `BOHB`.
- The constructor argument is `evaluate_fn`, not `evaluate`.
- `TPEConf` and `tpe_overrides` control the sampler configuration.
- `PruningConfig` and `pruning_overrides` control final-loss and intermediate-step pruning behavior.
- `history_export_jsonl` and `prior_trials_jsonl` support reuse across runs.
