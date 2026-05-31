"""
Multi-fidelity architecture search pipeline.

Phases:
1. Parallel zero-cost evaluation of ``num_candidates`` random architectures.
2. Select top-*k* candidates by aggregate score.
3. Short DARTS training + architecture derivation for each top candidate.
4. Select the best derived model by validation loss.
5. Full final training of the best model.

Public entry-point: :func:`run_multi_fidelity_search`.
"""

from __future__ import annotations

import concurrent.futures
import copy
import datetime
import logging
import os
import time
from typing import Any

import torch

from ..utils.training import reset_model_parameters
from .candidate_scoring import rescore_candidates_poolwise
from .stats_reporting import append_whatif_estimates, mean_std, save_csv, save_json

# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------


def bilevel_lr_sensitivity(
    trainer,
    model_factory,
    train_loader,
    val_loader,
    *,
    model_lrs=(1e-4, 3e-4, 1e-3, 3e-3),
    arch_lrs=(3e-4, 1e-3, 3e-3, 1e-2),
    seeds=(0, 1, 2),
    epochs: int = 30,
    save_csv_path: str | None = None,
):
    """
    Grid-search over (``model_lr``, ``arch_lr``, ``seed``) configurations.

    For each combination, a fresh model is created via ``model_factory()``,
    trained with DARTS bilevel optimisation, then the derived architecture is
    evaluated on ``val_loader``.

    Args:
        trainer:        :class:`~darts.trainer.DARTSTrainer` instance.
        model_factory:  Callable ``() -> model`` (no args) that returns a
                        freshly-initialised model placed on the correct device.
        train_loader:   Training DataLoader.
        val_loader:     Validation DataLoader.
        model_lrs:      Iterable of model learning rates to sweep.
        arch_lrs:       Iterable of architecture learning rates to sweep.
        seeds:          Iterable of random seeds to average across.
        epochs:         DARTS training epochs per configuration.
        save_csv_path:  If provided, write a CSV summary to this path.

    Returns:
        :class:`pandas.DataFrame` with columns:
        ``model_lr``, ``arch_lr``, ``seed``, ``best_val_loss_mixed``,
        ``val_loss_derived``, ``train_time_s``, ``health_score``,
        ``avg_identity_dominance``.
    """
    import random as _random

    import numpy as np
    import pandas as pd

    results = []

    for mlr in model_lrs:
        for alr in arch_lrs:
            for s in seeds:
                torch.manual_seed(s)
                np.random.seed(s)
                _random.seed(s)

                model = model_factory().to(trainer.device)
                out = trainer.train_darts_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    model_learning_rate=mlr,
                    arch_learning_rate=alr,
                    use_bilevel_optimization=True,
                    verbose=False,
                )

                derived = trainer.derive_final_architecture(out["model"])
                derived_val = trainer._evaluate_model(derived, val_loader)

                health_last = None
                if out.get("diversity_scores"):
                    health_last = out["diversity_scores"][-1]

                results.append({
                    "model_lr": mlr,
                    "arch_lr": alr,
                    "seed": s,
                    "best_val_loss_mixed": float(out["best_val_loss"]),
                    "val_loss_derived": float(derived_val),
                    "train_time_s": float(out["training_time"]),
                    "health_score": None
                    if not health_last
                    else float(health_last["health_score"]),
                    "avg_identity_dominance": None
                    if not health_last
                    else float(health_last["avg_identity_dominance"]),
                })

    df = pd.DataFrame(results)
    if save_csv_path:
        df.to_csv(save_csv_path, index=False)
    return df
