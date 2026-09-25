"""Input contracts and recording-aware development splits."""

from __future__ import annotations

import numpy as np


def waveform_matrix(x, *, context_size=None, allow_empty=False):
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] < 2:
        raise ValueError("Expected waveform contexts with shape [samples, time>=2].")
    if not allow_empty and len(x) == 0:
        raise ValueError("At least one waveform context is required.")
    if context_size is not None and x.shape[1] != context_size:
        raise ValueError(f"Expected context length {context_size}, got {x.shape[1]}.")
    if not np.isfinite(x).all():
        raise ValueError("Waveform contexts must contain only finite values.")
    return x


def label_vector(y, n):
    y = np.asarray(y)
    if y.ndim != 1 or len(y) != n:
        raise ValueError("Labels must be a one-dimensional array matching the samples.")
    if y.dtype.kind in "fc" and not np.isfinite(y).all():
        raise ValueError("Labels must be finite.")
    if any(label is None for label in y):
        raise ValueError("Labels cannot contain None.")
    return y


def split_training_indices(labels, validation_split=0.1, *, groups=None, seed=42):
    """Stratify rows, or hold out whole groups while retaining every training class.

    Group search is a bounded randomized greedy search. Validation class coverage
    and sample fraction are best-effort when groups are scarce or imbalanced.
    No row-level fallback is permitted for an infeasible group split.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or len(labels) < 2:
        raise ValueError("At least two one-dimensional labels are required.")
    if not np.isfinite(validation_split) or not 0 <= validation_split < 1:
        raise ValueError("validation_split must be in [0, 1).")
    _, encoded = np.unique(labels, return_inverse=True)
    n_classes = encoded.max() + 1
    if groups is not None:
        groups = label_vector(groups, len(labels))
    if validation_split == 0:
        return np.arange(len(labels)), np.empty(0, dtype=int)
    rng = np.random.default_rng(seed)
    if groups is None:
        train, validation = [], []
        for c in range(n_classes):
            indices = rng.permutation(np.flatnonzero(encoded == c))
            count = min(len(indices) - 1, max(1, int(len(indices) * validation_split)))
            validation.extend(indices[:count])
            train.extend(indices[count:])
        if not validation:
            raise ValueError(
                "Not enough samples to retain each training class and validate; use validation_split=0."
            )
        return np.sort(train), np.sort(validation)

    unique_groups, group_index = np.unique(groups, return_inverse=True)
    if len(unique_groups) < 2:
        raise ValueError(
            "Group validation needs at least two distinct recording groups."
        )
    counts = np.zeros((len(unique_groups), n_classes), dtype=int)
    np.add.at(counts, (group_index, encoded), 1)
    total = counts.sum(axis=0)
    target = max(1, round(len(unique_groups) * validation_split))
    best = None
    best_cost = np.inf
    for _ in range(128):
        remaining = total.copy()
        chosen = []
        for g in rng.permutation(len(unique_groups)):
            if np.all(remaining - counts[g] > 0):
                chosen.append(g)
                remaining -= counts[g]
            if len(chosen) == target:
                break
        if not chosen:
            continue
        validation = np.flatnonzero(np.isin(group_index, chosen))
        held_counts = total - remaining
        cost = (
            2 * np.count_nonzero(held_counts == 0)
            + abs(len(validation) / len(labels) - validation_split)
            + np.mean(np.abs(held_counts / total - validation_split))
        )
        if cost < best_cost:
            best, best_cost = validation, cost
    if best is None:
        raise ValueError(
            "Cannot hold out recording groups while retaining every training class; provide more groups or use validation_split=0."
        )
    return np.setdiff1d(np.arange(len(labels)), best), best
