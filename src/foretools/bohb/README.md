# BOHB

A BOHB hyperparameter optimizer: **Hyperband** resource allocation with a
vectorized **multi-fidelity multivariate TPE** sampler, synchronous successive
halving for serial runs, and **ASHA** asynchronous promotion when
`parallel_jobs > 1`. It suits problems where evaluations are expensive and
you want to stop bad configurations early.

## Features

- **Hyperband + successive halving**: budgets ramp up as configurations prove
  themselves. Serial runs promote the exact top `n_{i+1}` per rung (classic
  SH). Parallel runs use ASHA: a config is promoted once it ranks in the top
  `k // eta` of its rung, and higher-rung work is dispatched first.
- **Multi-fidelity TPE sampler** (`mftpe.py`, the default):
  - The density model is fit on the largest budget that has enough results
    (BOHB rule).
  - Multivariate product kernels capture parameter interactions.
  - Ints are modelled as integer bins; categoricals use an Aitchison-Aitken
    kernel.
  - Conditional parameters are supported.
  - Batches use a constant liar, and startup points are scrambled Sobol.
  - Fully numpy-vectorized.
- **Two-level pruning**: final-loss pruning (the result is kept, only
  promotion is blocked) and intermediate step-level pruning via
  `Trial.report`. Step-pruned trials still feed their last report to the
  sampler.
- **Transfer learning**: warm-start from prior runs and export history
  via JSONL.
- **Parallel evaluation**: `parallel_jobs > 1` runs candidates concurrently.
- **Plotting helpers**: optimization history, budget-vs-loss, parameter
  importance, parallel coordinates.
- **Convergence detection**: automatic early stopping when optimization
  plateaus, based on no-improvement count, plateau detection, and variance.
- **Legacy sampler** (`sampler="legacy"`): the original per-parameter TPE.
  It adds a per-parameter trust region, qNEI, Thompson and
  local-penalization batch selection, a GP surrogate, and LOO-CV bandwidths.
  All of it is configured through `TPEConf`.

## Samplers

| `sampler`          | Notes                                                        |
| ------------------ | ------------------------------------------------------------ |
| `"mftpe"` (default) | Fast, vectorized multi-fidelity TPE. Tune with `sampler_options`. |
| `"legacy"`         | Original TPE. Tune with `tpe_conf` / `tpe_overrides`.          |

`sampler_options` for `"mftpe"` (see `MultiFidelityTPE` for all):

| Option                | Default   | Purpose                                                   |
| --------------------- | --------- | --------------------------------------------------------- |
| `gamma`               | `top_n_percent / 100` | Fraction of results forming the "good" density l(x). |
| `n_ei_candidates`     | `64`      | Samples drawn from l(x) per proposal.                     |
| `min_points_in_model` | `d + 2`   | Results needed at a budget before it is modelled.         |
| `random_fraction`     | `0.1`     | Share of proposals drawn uniformly at random.             |
| `bandwidth_factor`    | `1.5`     | Multiplier on the Scott-rule kernel bandwidth.            |
| `constant_liar`       | `True`    | Spread batch proposals by treating picks as pending bad points. |

Hard and soft constraints set via `TPEConf.constraints` / `tpe_overrides`
(`hard_constraints`, `soft_constraints`, `soft_penalty_weight`) apply to both
samplers. So do `n_ei_candidates`, `min_bandwidth` and `n_startup_trials`
(which maps to `min_points_in_model`) in `tpe_overrides`.

Conditional parameters take a third, options element:

```python
config_space = {
    "opt":      ("choice", ["sgd", "adam"]),
    "momentum": ("float", (0.0, 0.99), {"condition": {"parent": "opt", "values": ["sgd"]}}),
}
```

## Import surface

```python
from foretools.bohb import (
    BOHB,
    HyperbandScheduler,
    PruningConfig,
    TPEConf,
)
from foretools.bohb.plotter import OptimizationPlotter
from foretools.bohb.trial import Trial, TrialPruned
```

## Quick start

```python
from foretools.bohb import BOHB

config_space = {
    "lr":         ("float",  (1e-5, 1e-1, "log")),
    "batch_size": ("choice", [16, 32, 64, 128]),
    "dropout":    ("float",  (0.0, 0.5)),
    "hidden":     ("int",    (16, 256)),
}

def objective(config, budget):
    # `budget` is the resource (epochs, samples, ...) granted to this trial.
    return train_and_score(config, epochs=int(budget))

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

`run()` takes no arguments — configure iterations via `n_iterations`.

## Config space

Each entry is `(type, spec)`:

| Type     | Spec                          | Notes                          |
| -------- | ----------------------------- | ------------------------------ |
| `float`  | `(lo, hi)`                    | Uniform on `[lo, hi]`          |
| `float`  | `(lo, hi, "log")`             | Log-uniform on `[lo, hi]`      |
| `int`    | `(lo, hi)`                    | Inclusive integer range        |
| `choice` | `[v1, v2, ...]`               | Categorical                    |

## Objective signatures

Both forms are supported and auto-detected:

- `objective(config, budget) -> float` — minimal form.
- `objective(config, budget, trial) -> float` — enables intermediate
  reporting and step-level pruning.

```python
from foretools.bohb.trial import TrialPruned

def objective(config, budget, trial):
    model = build(config)
    for epoch in range(int(budget)):
        loss = train_one_epoch(model)
        trial.report(epoch, loss)   # may raise TrialPruned
    return final_validation_loss(model)
```

`trial.report(step, loss)` checks against the per-step pruning policy and
raises `TrialPruned` when the trial should be stopped. BOHB catches that
internally — you don't need to wrap it.

## Constructor reference (key arguments)

| Argument                | Default        | Purpose                                                |
| ----------------------- | -------------- | ------------------------------------------------------ |
| `config_space`          | —              | See above.                                             |
| `evaluate_fn`           | —              | The objective. Lower loss is better.                   |
| `min_budget`            | `1.0`          | Smallest resource per trial.                           |
| `max_budget`            | `81.0`         | Largest resource per trial.                            |
| `eta`                   | `3`            | Hyperband halving factor (≥ 2).                        |
| `n_iterations`          | `10`           | Number of full Hyperband iterations.                   |
| `top_n_percent`         | `15`           | TPE good/bad split (in percent).                       |
| `pruning_mode`          | `"conservative"` | `"conservative"`, `"balanced"`, or `"aggressive"`.    |
| `early_prune`           | `True`         | Enable final-loss pruning.                             |
| `parallel_jobs`         | `1`            | Concurrent evaluations per rung.                       |
| `seed`                  | `None`         | RNG seed.                                              |
| `sampler`               | `"mftpe"`      | `"mftpe"` (default) or `"legacy"`.                      |
| `sampler_options`       | `None`         | Keyword overrides for `MultiFidelityTPE`.              |
| `tpe_conf` / `tpe_overrides`         | `None`     | `TPEConf` instance and/or dict of overrides.   |
| `pruning_conf` / `pruning_overrides` | `None`     | `PruningConfig` instance and/or dict of overrides. |
| `prior_trials_jsonl`    | `None`         | Warm-start from a JSONL history file.                  |
| `history_export_jsonl`  | `None`         | Path to dump full history at end of `run()`.           |
| `handle_errors`         | `True`         | Catch and record exceptions instead of crashing.       |
| `verbose`               | `True`         | Print progress per iteration/bracket.                  |
| `max_no_improvement_rounds` | `None`     | Early stop: no significant improvement in N rounds.    |
| `convergence_threshold` | `1e-6`        | Threshold for convergence detection.                   |
| `min_improvement_frac`  | `0.001`       | Minimum improvement fraction to avoid premature stop.  |
| `convergence_lookback`  | `10`          | Number of recent rounds to check for convergence.      |

## Pruning

Two independent pruning paths share `pruning_mode`:

- **Final-loss pruning** (between trials): blocks promotion of trials whose
  final loss is worse than a moving quantile of historical losses. The result
  is still recorded and modelled. Gated by `early_prune`.
- **Step-level pruning** (within a trial): triggered by `trial.report` and
  uses cohort statistics at the same step/progress.

Tune via `pruning_overrides={...}` or by passing a fully-built
`PruningConfig`. See `foretools/bohb/pruning.py` for every knob.

## Transfer learning

```python
# Run A: persist history
bohb_a = BOHB(..., history_export_jsonl="runs/a.jsonl")
bohb_a.run()

# Run B: warm-start TPE from run A
bohb_b = BOHB(..., prior_trials_jsonl="runs/a.jsonl")
bohb_b.run()
```

JSONL records include `config`, `budget`, `loss`, `iteration`, `bracket`,
and `round` per trial.

## Batch selection strategies

These apply to `sampler="legacy"` only. Use `TPEConf.batch` to configure the batch selector:

```python
from foretools.bohb.tpe import TPEConf

conf = TPEConf()
conf.batch['batch_strategy'] = 'ts'  # Thompson sampling
conf.batch['constant_liar'] = True
conf.batch['liar_strategy'] = 'gp'   # GP-conditional liar
```

Available strategies:

| Strategy | Description |
| -------- | ----------- |
| `diversity` / `greedy_diversity` | Greedy selection with diversity bonus (default) |
| `ts` / `thompson` | Thompson sampling with uncertainty-aware selection |
| `lp` / `local_penalization` | Distance-based local penalization |
| `qnei` / `qnei_batch` | qNoisy Expected Improvement (greedy batch) |

Additional batch options:
- `constant_liar`: Inject fake values for pending evaluations
- `liar_strategy`: `"mean"`, `"median"`, `"worst"`, `"gp"`, `"gp_median"`, `"gp_quantile"`
- `n_fantasies`: MC samples for qNEI
- `alpha`: Thompson sampling exploration strength
- `n_samples`: Thompson sampling stochastic samples

## Convergence detection

BOHB can automatically detect when optimization has converged and stop early:

```python
bohb = BOHB(
    config_space=config_space,
    evaluate_fn=objective,
    n_iterations=20,              # Planned max
    max_no_improvement_rounds=5,  # Stop after 5 rounds without improvement
    verbose=True,
)
best_config, best_loss = bohb.run()
# If converged: "Rounds: 8/20" and "Converged: Yes — no improvement for 6 rounds"
```

Three signals are checked (at least 2 must agree):
1. **No-improvement count**: rounds without significant improvement
2. **Plateau**: recent rounds show near-zero range and tiny improvement
3. **Low variance**: recent improvements are consistently near zero

Access diagnostics:
```python
conv = bohb.check_convergence()  # Dict with 'converged', 'reason', 'details'
```

## Results and plotting

```python
history     = bohb.get_optimization_history()  # list[dict]
top_configs = bohb.get_top_configs(5)          # [(config, loss), ...]

plotter = OptimizationPlotter.from_bohb(bohb)
plotter.plot_optimization_history()
plotter.plot_budget_vs_loss()
plotter.plot_bracket_best()
plotter.plot_param_importance()
plotter.plot_parallel_coordinates()
plotter.plot_param_effect("lr")
```

`OptimizationPlotter` requires `matplotlib`.

## See also

- Tutorial: [docs/tutorials/optimize-with-bohb.md](../../docs/tutorials/optimize-with-bohb.md)
- Source: [bohb.py](bohb.py), [mftpe.py](mftpe.py), [tpe.py](tpe.py), [hyperband.py](hyperband.py),
  [pruning.py](pruning.py), [trial.py](trial.py)
