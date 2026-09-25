"""Shared per-window preprocessing for the discharge classifiers."""

from __future__ import annotations

import numpy as np

from foreblocks.features._validation import normalize_series


def normalize_context(x: np.ndarray) -> np.ndarray:
    """Per-context Z-score normalization with a numerical floor, as
    specified for the raw waveform in the source paper. Used by every
    classifier that consumes raw amplitude (the CNN and the Rocket
    baseline) so none of them can use absolute scale as an implicit,
    unintended shortcut for classes with overlapping RMS ranges (corona vs.
    surface_discharge, see `model_proposal.md`'s Milestone 1 results).
    `FeatureDischargeClassifier` explicitly does the opposite -- it computes
    log-RMS/log-peak as *named* features on the raw signal instead, so it is
    not normalized before feature extraction.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] < 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite waveforms with at least two samples.")
    return normalize_series(x)
