# BOHB: Bayesian Optimization & Hyperband (with SOTA TPE)

A robust, high-performance implementation of **BOHB** (Bayesian Optimization and Hyperband) featuring a State-of-the-Art (SOTA) **Tree-structured Parzen Estimator (TPE)**.

This implementation enhances standard TPE with modern techniques to handle non-Gaussian distributions, dependencies between parameters, and large-scale optimization tasks efficiently.

## 🚀 Key Features

### SOTA TPE Enhancements
- **Input Warping (Yeo-Johnson)**: Automatically transforms non-Gaussian parameter distributions to improve Kernel Density Estimation (KDE) accuracy.
- **Adaptive Multivariate TPE**: Auto-detects when to switch from univariate to multivariate KDE based on sample size ($N \ge 5 \times D$), capturing parameter dependencies.
- **Adaptive TPE (ATPE)**:
    - **Dynamic Gamma**: Automatically adjusts the exploration/exploitation trade-off based on optimization progress.
    - **Global Filtering**: Uses Z-score and clustering to filter out outliers and noise from the observation history *before* modeling.
    - **Parameter Blocking**: Greedily fixes parameters that are strongly correlated with loss (excessive exploration of converged params is prevented).
- **Categorical Handling**: Supports continuous, integer, and categorical parameters with smoothing and embedding options.

### Algorithms
- **BOHB**: Combines Hyperband's efficient resource allocation with TPE's Bayesian guidance.
- **Hyperband**: Pure random search with aggressive early stopping (bracket-based).
- **TPE**: Standalone Bayesian optimization.

## 📦 Installation

Requires `numpy` and `scipy`.

```bash
pip install numpy scipy
```

## ⚡ Quick Start

### Basic Usage (BOHB)

```python
from bohb import BOHB
import time

# Define your objective function
def objective(config, budget):
    # Simulate training
    loss = (config["x"] - 2)**2 + (config["y"] + 1)**2
    # Add noise or resource scaling if needed
    return loss

# Define search space
config_space = {
    "x": ("float", (-5.0, 5.0)),
    "y": ("float", (-5.0, 5.0)),
    "method": ("choice", ["sgd", "adam"]),
}

# Initialize BOHB
bohb = BOHB(
    config_space=config_space,
    evaluate=objective,
    max_budget=27,
    min_budget=1,
    eta=3,
    parallel_jobs=1,  # Set >1 for parallel execution
    verbose=True
)

# Run optimization
best_config, best_loss = bohb.run(n_iterations=5)

print(f"Best Config: {best_config}")
print(f"Best Loss: {best_loss}")
```

### Enabling SOTA Features (ATPE)

To use the advanced Adaptive TPE features (enabled by default in `TPE` but configurable):

```python
from bohb import BOHB

bohb = BOHB(
    ...,
    # Enable ATPE (default True in TPE, explicitly set here for clarity)
    atpe=True,
    atpe_params={
        "filter_type": "zscore",   # Filter outliers
        "filter_threshold": 1.5,   # Z-score threshold
    },
    blocking_threshold=0.8,        # Correlation threshold for parameter blocking
    multivariate="auto",           # Auto-enable multivariate KDE
    warping=True                   # Enable Yeo-Johnson warping
)
```

## 📊 Benchmarking

A benchmarking script is included to verify performance on synthetic functions.

```bash
python benchmark_atpe.py
```

## 🛠 Project Structure

- **`bohb.py`**: Main BOHB and Hyperband implementation.
- **`tpe.py`**: The enhanced Tree-structured Parzen Estimator core.
- **`numerics.py`**: Numerical utilities including safe log, robust scaling, and Yeo-Johnson transforms.
- **`param_models.py`**: Probability density models (Float, Int, Categorical).
- **`plotter.py`**: Visualization utilities for optimization history.
- **`batch_selectors.py`**: Strategies for selecting batches of candidates (Diversity, Local Penalization).

## 📄 License

MIT License.
