from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None
    optim = None

from .utils import safe_log


def realistic_nn_objective(config: Dict[str, Any], budget: float) -> float:
    """
    Simulate training dynamics:
    - Some hyperparameter optimum
    - Diminishing returns with budget
    - Noise decreases with budget
    """
    lr = float(config["lr"])
    batch_size = int(config.get("batch_size", 32))
    dropout = float(config.get("dropout", 0.0))

    optimal_lr = 1e-2
    optimal_batch_size = 64
    optimal_dropout = 0.3

    lr_penalty = (math.log10(lr) - math.log10(optimal_lr)) ** 2
    bs_penalty = ((batch_size - optimal_batch_size) ** 2) / 1000.0
    do_penalty = (dropout - optimal_dropout) ** 2

    base_loss = lr_penalty + bs_penalty + do_penalty + 0.1

    improvement = 1.0 / (1.0 + 0.5 * safe_log(budget))
    noise = np.random.normal(0.0, 0.02 / math.sqrt(max(budget, 1e-12)))

    final_loss = base_loss * improvement + float(noise)
    return max(1e-3, float(final_loss))


def torch_mlp_objective(
    config: Dict[str, Any],
    budget: float,
    trial: Optional[Any] = None,
) -> float:
    """
    Small Torch MLP objective for BOHB demos.
    budget controls epochs (rounded to int >= 1).
    """
    if torch is None:
        raise RuntimeError(
            "PyTorch is not installed. Install torch to use this objective."
        )

    torch.manual_seed(42)
    np.random.seed(42)

    lr = float(config["lr"])
    hidden = int(config.get("hidden", 64))
    dropout = float(config.get("dropout", 0.2))
    batch_size = int(config.get("batch_size", 64))

    n_train = 1024
    n_val = 256
    n_features = 20
    X = torch.randn(n_train + n_val, n_features)
    true_w = torch.randn(n_features, 1) * 0.5
    y = X @ true_w + 0.1 * torch.randn(n_train + n_val, 1)
    X_train, X_val = X[:n_train], X[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model = nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(hidden, 1),
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    epochs = max(1, int(round(float(budget))))
    model.train()
    for ep in range(epochs):
        idx = torch.randint(0, n_train, (batch_size,))
        xb = X_train[idx]
        yb = y_train[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

        if trial is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            model.train()
            trial.report(ep, val_loss)

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val).item()
    return float(val_loss)
