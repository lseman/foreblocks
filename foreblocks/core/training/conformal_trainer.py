"""Trainer-side conformal prediction wrapper.

Extracted from the monolithic ``trainer.py``.  These methods delegate to
the conformal engine (``self.conformal_engine``) while providing a clean
Trainer API for calibration, online updates, and prediction with
conformal intervals.

The conformal engine itself lives in ``conformal.py`` (~1500 lines).
This module is the thin Trainer-facing API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


if TYPE_CHECKING:
    from foreblocks.core.training.trainer import Trainer


# ---------------------------------------------------------------------------
# Conformal helpers (Trainer-side only)
# ---------------------------------------------------------------------------

def _as_numpy(x: Any) -> np.ndarray:
    """Convert *x* to a NumPy array (handles tensors, lists, scalars)."""
    if isinstance(x, np.ndarray):
        return x
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _ensure_y_shape_like_intervals(y_np: np.ndarray, lower: np.ndarray) -> np.ndarray:
    """Make *y* shape match ``[N, H, D]`` like *lower* / *upper*.

    Accepts y as ``[N, H]``, ``[N, H, 1]``, or ``[N, H, D]``.
    Broadcasts a singleton last-dimension to match *lower*.
    """
    if y_np.ndim == 2:
        y_np = y_np[:, :, None]  # [N,H,1]
    if y_np.ndim != 3:
        raise ValueError(f"y must have shape [N,H] or [N,H,D], got {y_np.shape}")
    if y_np.shape[-1] == 1 and lower.shape[-1] > 1:
        y_np = np.repeat(y_np, lower.shape[-1], axis=-1)
    if y_np.shape != lower.shape:
        raise ValueError(f"Shape mismatch: y {y_np.shape} vs intervals {lower.shape}")
    return y_np


def _collect_xy_from_loader(cal_loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    """Collect (X, y) from a calibration DataLoader."""
    Xc, Yc = [], []
    for batch in cal_loader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError(
                "Calibration loader must yield (X, y) or (X, y, ...) tuples."
            )
        xb, yb = batch[0], batch[1]
        Xc.append(xb)
        Yc.append(yb)
    Xc = torch.cat(Xc, dim=0)
    Yc = torch.cat(Yc, dim=0)
    return Xc, Yc


# ---------------------------------------------------------------------------
# Coverage summary (shared between compute_coverage and streaming)
# ---------------------------------------------------------------------------

def coverage_summary(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_coverage: float,
    *,
    include_breakdowns: bool = False,
) -> dict[str, Any]:
    """Compute shared conformal coverage statistics."""
    covered = (y_true >= lower) & (y_true <= upper)
    joint_covered = covered.all(axis=(1, 2))
    widths = upper - lower

    summary: dict[str, Any] = {
        "coverage": float(covered.mean()),
        "joint_coverage": float(joint_covered.mean()),
        "target_coverage": float(target_coverage),
        "coverage_gap": float(covered.mean() - target_coverage),
        "joint_coverage_gap": float(joint_covered.mean() - target_coverage),
        "mean_interval_width": float(widths.mean()),
        "std_interval_width": float(widths.std()),
        "min_interval_width": float(widths.min()),
        "max_interval_width": float(widths.max()),
    }
    if include_breakdowns:
        summary.update({
            "per_horizon_coverage": covered.mean(axis=(0, 2)),
            "per_feature_coverage": covered.mean(axis=(0, 1)),
            "per_feature_joint_coverage": covered.all(axis=1).mean(axis=0),
            "per_horizon_all_feature_coverage": covered.all(axis=2).mean(axis=0),
        })
    return summary


# ---------------------------------------------------------------------------
# Trainer-side conformal methods
# ---------------------------------------------------------------------------

def calibrate_conformal(
    trainer: "Trainer",
    cal_loader: DataLoader,
    state_model: Any = None,
    feature_extractor: Any = None,
    jackknife_cv_models: Any = None,
    jackknife_cv_indices: Any = None,
    enbpi_member_models: Any = None,
    enbpi_boot_indices: Any = None,
) -> None:
    """Calibrate conformal engine with held-out calibration data.

    Contract:
    - ``self.model`` must support ``forward(X)`` for inference.
    - Output must be ``[N, H, D]`` (or ``[N, H]``, which engine handles).
    """
    engine = trainer.conformal_engine
    if engine is None:
        raise RuntimeError("Conformal prediction not enabled in config.")
    method = engine.method

    # Method-specific hard requirements
    if method == "cptc" and state_model is None:
        raise ValueError("CPTC requires `state_model`.")
    if method == "enbpi" and enbpi_member_models is not None and enbpi_boot_indices is None:
        raise ValueError("EnbPI with member models requires `enbpi_boot_indices`.")

    if method == "afocp" and feature_extractor is None:
        import warnings
        warnings.warn(
            "AFOCP without a feature_extractor will use an internal "
            "(untrained) DefaultFeatureExtractor. This is valid but may be "
            "weaker than a pretrained extractor."
        )

    Xc, Yc = _collect_xy_from_loader(cal_loader)
    batch_size = int(getattr(trainer.config, "batch_size", 256))

    print(f"[Conformal] Calibrating with {len(Xc)} samples using method='{method}'")

    engine.calibrate(
        model=trainer.model,
        X_cal=Xc, y_cal=Yc, device=trainer.device, batch_size=batch_size,
        state_model=state_model, feature_extractor=feature_extractor,
        enbpi_member_models=enbpi_member_models, enbpi_boot_indices=enbpi_boot_indices,
        jackknife_cv_models=jackknife_cv_models, jackknife_cv_indices=jackknife_cv_indices,
    )

    print(f"[Conformal] Calibration completed. Radii shape: {engine.radii.shape}")


def update_conformal(
    trainer: "Trainer",
    X_new: torch.Tensor,
    y_new: torch.Tensor,
    state_model: Any = None,
    feature_extractor: Any = None,
    sequential: bool = True,
) -> None:
    """Online update for adaptive conformal methods.

    Args:
        sequential: If True, update point-by-point within the batch (required
            for ACI/AgACI correctness). If False, batch update (faster but
            may be approximate for adaptive methods).
    """
    engine = trainer.conformal_engine
    if engine is None:
        raise RuntimeError("Conformal prediction not enabled in config.")
    if engine.radii is None:
        raise RuntimeError("Conformal engine not calibrated. Call calibrate_conformal() first.")

    X_np = _as_numpy(X_new)
    y_np = _as_numpy(y_new)
    method = engine.method
    batch_size = int(getattr(trainer.config, "batch_size", 256))

    # For adaptive methods, update sequentially
    if sequential and method in ("aci", "agaci", "rolling"):
        for i in range(len(X_np)):
            engine.update(
                model=trainer.model,
                X_new=X_np[i : i + 1], y_new=y_np[i : i + 1],
                device=trainer.device, batch_size=1,
                state_model=state_model, feature_extractor=feature_extractor,
            )
    else:
        engine.update(
            model=trainer.model,
            X_new=X_np, y_new=y_np, device=trainer.device, batch_size=batch_size,
            state_model=state_model, feature_extractor=feature_extractor,
        )


def predict_with_intervals(
    trainer: "Trainer",
    X: torch.Tensor,
    return_tensors: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Predict with conformal intervals.

    Returns ``(preds, lower, upper)`` with shape ``[N, H, D]`` (or ``[N, H, 1]``
    if single target dim).  If *return_tensors* is True, returns PyTorch tensors.
    """
    engine = trainer.conformal_engine
    if engine is None:
        raise RuntimeError("Conformal prediction not enabled in config.")
    if engine.radii is None:
        raise RuntimeError("Conformal engine not calibrated. Call calibrate_conformal() first.")

    X_np = _as_numpy(X)
    batch_size = int(getattr(trainer.config, "batch_size", 256))

    preds, lower, upper = engine.predict(
        model=trainer.model, X=X_np, device=trainer.device, batch_size=batch_size,
    )

    # Normalize to [N, H, D] at the Trainer boundary
    if preds.ndim == 2:
        preds = preds[:, :, None]
        lower = lower[:, :, None]
        upper = upper[:, :, None]

    if return_tensors:
        return torch.from_numpy(preds), torch.from_numpy(lower), torch.from_numpy(upper)
    return preds, lower, upper


def compute_coverage(
    trainer: "Trainer",
    X: torch.Tensor,
    y: torch.Tensor,
) -> dict[str, float]:
    """Empirical coverage and basic interval stats.

    ``coverage`` is elementwise over all ``[N, H, D]`` entries.  Also exposes
    ``joint_coverage`` (a sample is covered only if *every* horizon/target is
    inside its interval).
    """
    _, lower, upper = predict_with_intervals(trainer, X, return_tensors=False)
    y_np = _as_numpy(y)
    y_np = _ensure_y_shape_like_intervals(y_np, lower)

    return coverage_summary(
        y_np, lower, upper,
        target_coverage=float(trainer.conformal_engine.q),
    )


def predict_with_intervals_streaming(
    trainer: "Trainer",
    dataloader: DataLoader,
    do_update: bool = True,
    return_numpy: bool = True,
    sequential: bool | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Streaming (rolling) prediction over a DataLoader.

    - Assumes dataloader yields ``(X, y)`` or ``(X, y, ...)``
    - Predicts intervals batch-by-batch in chronological order.
    - If *do_update* is True, performs online update AFTER predicting each batch.
    - If *sequential* is None, auto-enables for ACI methods (rolling/agaci).

    Returns ``(preds, lower, upper, y_true)`` all concatenated with shape ``[N, H, D]``.
    """
    engine = trainer.conformal_engine
    if engine is None or engine.radii is None:
        raise RuntimeError("Conformal engine not calibrated. Call calibrate_conformal() first.")

    trainer.model.eval()
    preds_all, low_all, up_all, y_all = [], [], [], []

    for batch in dataloader:
        if not isinstance(batch, (list, tuple)) or len(batch) < 2:
            raise ValueError("Dataloader must yield (X, y) or (X, y, ...) tuples.")
        xb, yb = batch[0], batch[1]

        preds_b, low_b, up_b = predict_with_intervals(trainer, xb, return_tensors=False)
        preds_all.append(preds_b)
        low_all.append(low_b)
        up_all.append(up_b)

        y_np = _as_numpy(yb)
        if y_np.ndim == 2:
            y_np = y_np[:, :, None]
        y_all.append(y_np)

        if do_update:
            update_conformal(trainer, xb, yb, sequential=sequential)

    preds = np.concatenate(preds_all, axis=0)
    low = np.concatenate(low_all, axis=0)
    up = np.concatenate(up_all, axis=0)
    y_true = np.concatenate(y_all, axis=0)

    if return_numpy:
        return preds, low, up, y_true
    return torch.from_numpy(preds), torch.from_numpy(low), torch.from_numpy(up), torch.from_numpy(y_true)


def compute_coverage_streaming(
    trainer: "Trainer",
    dataloader: DataLoader,
    do_update: bool = True,
    sequential: bool | None = None,
) -> dict[str, Any]:
    """Coverage diagnostics for streaming/rolling evaluation.

    Returns elementwise ``coverage`` and ``joint_coverage`` with breakdowns.
    """
    _, lower, upper, y_true = predict_with_intervals_streaming(
        trainer, dataloader, do_update=do_update, return_numpy=True, sequential=sequential,
    )

    return coverage_summary(
        y_true, lower, upper,
        target_coverage=float(trainer.conformal_engine.q),
        include_breakdowns=True,
    )
