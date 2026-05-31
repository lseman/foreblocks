"""
Zero-cost neural architecture search metric evaluation.

Thin wrappers around ``ZeroCostNAS`` from ``search.metrics`` that follow the
same interface as the rest of the search sub-package: free functions that
receive a ``trainer`` as their first argument.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from .metrics import Config, ZeroCostNAS
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
    fast_mode: bool | str = True,
    preset: str = "smart_fast",
) -> dict[str, Any]:
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
        fast_mode:   Use cheaper proxy settings.  Accepts ``True`` (fast),
                     ``False`` (full), or a preset name (``"smart_fast"``,
                     ``"ultra_fast"``).
        preset:      Named preset to use when ``fast_mode`` is truthy but
                     not ``True``.  Ignored when ``fast_mode=True``.

    Returns:
        Dict with keys:
        ``raw_metrics``, ``success_rates``, ``errors``, ``base_weights``.
    """
    if fast_mode is True:
        cfg = _make_config(max_samples=max_samples, fast_mode=True)
    elif fast_mode is False:
        cfg = _make_config(max_samples=max_samples, fast_mode=False)
    else:
        # fast_mode is a preset name
        cfg = _make_config_preset(max_samples=max_samples, preset=preset)
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
    fast_mode: bool | str = True,
    preset: str = "smart_fast",
    ablation: bool = False,
    n_random: int = 20,
    random_sigma: float = 0.25,
    seed: int = 0,
) -> dict[str, Any]:
    """
    Evaluate a model using zero-cost proxy metrics.

    When ``ablation=False`` (the default), a single aggregate score is
    produced using the weights from the selected preset.  When
    ``ablation=True``, the model is evaluated under many weight schemes
    and the per-scheme results are returned alongside the default.

    Args:
        trainer:      :class:`~darts.trainer.DARTSTrainer` instance.
        model:        Model to evaluate.
        dataloader:   Validation DataLoader.
        max_samples:  Max samples for metric computation.
        num_batches:  Batches to forward.
        fast_mode:    Use cheaper proxy settings.  Accepts ``True`` (fast),
                     ``False`` (full), or a preset name (``"smart_fast"``,
                     ``"ultra_fast"``).
        preset:       Named preset when ``fast_mode`` is a string.
        ablation:     Run weight-scheme ablation if ``True``.
        n_random:     Number of random weight perturbations to include.
        random_sigma: Standard deviation of Gaussian used for random schemes.
        seed:         Random seed for reproducibility.

    Returns:
        On baseline mode: dict from ``ZeroCostNAS.evaluate_model``.
        On ablation mode: dict with keys ``"ablation"``, ``"base_weights"``,
        ``"schemes"``, ``"per_scheme"``.
    """
    if fast_mode is True:
        cfg = _make_config(max_samples=max_samples, fast_mode=True)
    elif fast_mode is False:
        cfg = _make_config(max_samples=max_samples, fast_mode=False)
    else:
        cfg = _make_config_preset(max_samples=max_samples, preset=preset)
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

    per_scheme: dict[str, Any] = {}
    for scheme_name, weights in schemes.items():
        scheme_cfg = Config(max_samples=max_samples, max_outputs=cfg.max_outputs,
                            weights=weights)
        # Copy enabled flags from the base config
        scheme_cfg.jacobian_probes = cfg.jacobian_probes
        scheme_cfg.gradient_max_samples = cfg.gradient_max_samples
        scheme_cfg.fisher_per_sample = cfg.fisher_per_sample
        scheme_cfg.snip_mode = cfg.snip_mode
        scheme_cfg.heavy_metrics_batches = cfg.heavy_metrics_batches
        scheme_cfg.enable_grasp = cfg.enable_grasp
        scheme_cfg.enable_jacobian = cfg.enable_jacobian
        scheme_cfg.enable_synflow = cfg.enable_synflow
        scheme_cfg.conditioning_every_n_layers = cfg.conditioning_every_n_layers
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
    """Build a :class:`~search.metrics.Config` with appropriate defaults."""
    cfg = Config(max_samples=max_samples, max_outputs=10)
    if fast_mode:
        cfg.jacobian_probes = 1
        cfg.gradient_max_samples = max(
            1, min(int(max_samples), int(getattr(cfg, "gradient_max_samples", 4)), 2)
        )
        cfg.fisher_per_sample = False
        cfg.snip_mode = "current"
        cfg.heavy_metrics_batches = 1
        # Phase-1 search needs a cheap, reliable ranking signal. GRASP,
        # Jacobian and SynFlow are useful diagnostics, but they are also the
        # common culprits for very slow or backend-sensitive candidate screens.
        cfg.enable_grasp = False
        cfg.enable_jacobian = False
        cfg.enable_synflow = False
        cfg.weights = {
            k: v
            for k, v in cfg.weights.items()
            if k not in {"grasp", "jacobian", "synflow"}
        }
        cfg.conditioning_every_n_layers = max(2, cfg.conditioning_every_n_layers)
    return cfg


def _make_config_preset(*, max_samples: int, preset: str) -> Config:
    """Build a config from a named preset (smart_fast, ultra_fast, full)."""
    from .metrics import Config as _Config, _get_presets

    presets = _get_presets()
    if preset not in presets:
        raise ValueError(
            f"Unknown zero-cost preset {preset!r}. "
            f"Available: {list(presets.keys())}"
        )
    base = presets[preset]
    cfg = _Config(
        max_samples=max_samples,
        max_outputs=base.max_outputs,
        jacobian_probes=base.jacobian_probes,
        gradient_max_samples=base.gradient_max_samples,
        fisher_per_sample=base.fisher_per_sample,
        snip_mode=base.snip_mode,
        heavy_metrics_batches=base.heavy_metrics_batches,
        enable_grasp=base.enable_grasp,
        enable_jacobian=base.enable_jacobian,
        enable_synflow=base.enable_synflow,
        conditioning_every_n_layers=base.conditioning_every_n_layers,
        weights=dict(base.weights),
    )
    return cfg
