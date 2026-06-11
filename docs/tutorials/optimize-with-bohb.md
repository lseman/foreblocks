---
title: Optimize With BOHB
description: Define a BOHB search space, run budget-aware optimization, inspect results.
editLink: true
---

# Optimize With BOHB

This tutorial shows the shortest complete BOHB workflow in `foretools`: define a search space, write a budget-aware objective, run the optimizer, and inspect the result history.

## Minimal runnable example

```python
from foretools.bohb import BOHB
from foretools.bohb.plotter import OptimizationPlotter

config_space = {
    "lr": ("float", (1e-5, 1e-1, "log")),
    "hidden": ("int", (32, 256)),
    "dropout": ("float", (0.0, 0.5)),
    "optimizer": ("choice", ["adam", "adamw", "sgd"]),
}

def objective(config, budget):
    lr_target = 1e-2
    hidden_target = 128
    dropout_target = 0.2

    lr_penalty = abs(config["lr"] - lr_target)
    hidden_penalty = abs(config["hidden"] - hidden_target) / hidden_target
    dropout_penalty = abs(config["dropout"] - dropout_target)
    optimizer_penalty = 0.0 if config["optimizer"] == "adamw" else 0.15
    budget_bonus = 1.0 / max(budget, 1.0)

    return float(lr_penalty + hidden_penalty + dropout_penalty + optimizer_penalty + budget_bonus)

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

plotter = OptimizationPlotter.from_bohb(bohb)
plotter.plot_optimization_history()
plotter.plot_param_importance()
```text

This enables BOHB's trial-level pruning hook. Keep in mind that the current pruning rule is intentionally simple, so validate it against your workload before relying on it heavily.

## What to inspect after the run

Start with:

```python
history = bohb.get_optimization_history()
top5 = bohb.get_top_configs(5)
