"""
Zero-cost neural architecture search metric evaluation.

Thin wrappers around ``ZeroCostNAS`` from ``darts_metrics`` that follow the
same interface as the rest of the search sub-package: free functions that
receive a ``trainer`` as their first argument.
"""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn

from .nas_metrics import Config, ZeroCostNAS
from .weight_schemes import build_weight_schemes

# ---------------------------------------------------------------------------
# Raw metrics (no weighting – single evaluation pass)
# ---------------------------------------------------------------------------


def evaluate_zero_cost_metrics_raw(
    trainer,
    model: nn.Module,
    dataloader,
    *,
    max_samples: int = 32,
    num_batches: int = 1,
    fast_mode: bool = True,
) -> Dict[str, Any]:
    """
    Compute zero-cost proxies once and return the raw per-metric values.

    Unlike :func:`evaluate_zero_cost_metrics`, this function does **not**
    aggregate metrics into a single score; the caller can apply any weighting
    scheme afterwards.

    Args:
        trainer:     :class:`~darts.trainer.DARTSTrainer` instance.
        model:       The DARTS model to evaluate.
        dataloader:  DataLoader providing input batches.
        max_samples: Maximum number of samples to use.
        num_batches: Number of batches to forward through the model.
        fast_mode:   Use cheaper proxy settings (fewer Jacobian probes, etc.).

    Returns:
        Dict with keys:
        ``raw_metrics``, ``success_rates``, ``errors``, ``base_weights``.
    """
    cfg = _make_config(max_samples=max_samples, fast_mode=fast_mode)
    nas_evaluator = ZeroCostNAS(config=cfg)

    out = nas_evaluator.evaluate_model_raw_metrics(
        model=model,
        dataloader=dataloader,
        device=trainer.device,
        num_batches=num_batches,
    )

    return {
        "raw_metrics": out.get("raw_metrics", {}),
        "success_rates": out.get("success_rates", {}),
        "errors": out.get("errors", {}),
        "base_weights": dict(cfg.weights),
    }


# ---------------------------------------------------------------------------
# Aggregate score (single weight scheme or ablation)
# ---------------------------------------------------------------------------


def evaluate_zero_cost_metrics(
    trainer,
    model: nn.Module,
    dataloader,
    *,
    max_samples: int = 32,
    num_batches: int = 1,
    fast_mode: bool = True,
    ablation: bool = False,
    n_random: int = 20,
    random_sigma: float = 0.25,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Evaluate a model using zero-cost proxy metrics.

    When ``ablation=False`` (the default), a single aggregate score is
    produced using the default weights from :class:`~darts_metrics.Config`.
    When ``ablation=True``, the model is evaluated under many weight schemes
    and the per-scheme results are returned alongside the default.

    Args:
        trainer:      :class:`~darts.trainer.DARTSTrainer` instance.
        model:        Model to evaluate.
        dataloader:   Validation DataLoader.
        max_samples:  Max samples for metric computation.
        num_batches:  Batches to forward.
        fast_mode:    Use cheaper proxy settings.
        ablation:     Run weight-scheme ablation if ``True``.
        n_random:     Number of random weight perturbations to include.
        random_sigma: Standard deviation of Gaussian used for random schemes.
        seed:         Random seed for reproducibility.

    Returns:
        On baseline mode: dict from ``ZeroCostNAS.evaluate_model``.
        On ablation mode: dict with keys ``"ablation"``, ``"base_weights"``,
        ``"schemes"``, ``"per_scheme"``.
    """
    cfg = _make_config(max_samples=max_samples, fast_mode=fast_mode)
    nas_evaluator = ZeroCostNAS(config=cfg)

    if not ablation:
        return nas_evaluator.evaluate_model(
            model, dataloader, trainer.device, num_batches=num_batches
        )

    # Ablation: compute metrics once, re-score under many weight schemes
    schemes = build_weight_schemes(
        base_weights=dict(cfg.weights),
        n_random=n_random,
        random_sigma=random_sigma,
        seed=seed,
    )

    per_scheme: Dict[str, Any] = {}
    for scheme_name, weights in schemes.items():
        scheme_cfg = _make_config(max_samples=max_samples, fast_mode=fast_mode)
        scheme_cfg.weights = weights
        ev = ZeroCostNAS(config=scheme_cfg)
        out = ev.evaluate_model(
            model, dataloader, trainer.device, num_batches=num_batches
        )
        per_scheme[scheme_name] = {
            "aggregate_score": out["aggregate_score"],
            "metrics": out["metrics"],
            "success_rates": out["success_rates"],
        }

    return {
        "ablation": True,
        "base_weights": dict(cfg.weights),
        "schemes": list(per_scheme),
        "per_scheme": per_scheme,
    }


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def _make_config(*, max_samples: int, fast_mode: bool) -> Config:
    """Build a :class:`~darts_metrics.Config` with appropriate defaults."""
    cfg = Config(max_samples=max_samples, max_outputs=10)
    if fast_mode:
        cfg.jacobian_probes = 1
        cfg.gradient_max_samples = max(
            1, min(int(max_samples), int(getattr(cfg, "gradient_max_samples", 4)), 2)
        )
        cfg.fisher_per_sample = False
        cfg.snip_mode = "current"
        cfg.heavy_metrics_batches = 1
        cfg.conditioning_every_n_layers = max(2, cfg.conditioning_every_n_layers)
    return cfg
