"""Chronological data splitting for bilevel architecture search."""

from __future__ import annotations

from operator import index

import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
    Subset,
    SubsetRandomSampler,
)


def _eligible_indices(loader):
    """Return local sample indices paired with chronological base-dataset indices."""
    sampler = loader.sampler
    dataset = loader.dataset
    if type(sampler) is SubsetRandomSampler:
        eligible = [index(i) for i in sampler.indices]
        if any(i < 0 or i >= len(dataset) for i in eligible):
            raise ValueError("Subset sampler contains an out-of-range index")
    elif type(sampler) is SequentialSampler or (
        type(sampler) is RandomSampler
        and not sampler.replacement
        and sampler.num_samples == len(dataset)
    ):
        eligible = range(len(dataset))
    else:
        raise ValueError(
            "Unsupported bilevel sampler; pass a chronological Subset dataset"
        )

    base_indices = eligible
    while isinstance(dataset, Subset):
        parent = dataset.dataset
        base_indices = [
            range(len(parent))[index(dataset.indices[i])] for i in base_indices
        ]
        dataset = parent
    if not isinstance(base_indices, range) and len(set(base_indices)) != len(
        base_indices
    ):
        raise ValueError("Duplicate samples in bilevel fold")
    return dataset, eligible, base_indices


def create_bilevel_loaders(
    train_loader: DataLoader,
    seed: int = 42,
    *,
    forecast_horizon: int = 1,
    window_stride: int = 1,
):
    """Split eligible windows 70/30 and purge overlapping weight-training targets.

    For regularly spaced windows, base dataset index i starts its target at
    i * window_stride and covers forecast_horizon time steps. For irregular
    windows supply forecast_origins and target_ends on the base dataset. These
    are comparable timestamps for the first and last target, both inclusive.
    Origins can repeat (e.g. panel data); tied origins stay in one partition.
    """
    dataset = train_loader.dataset
    if isinstance(dataset, IterableDataset):
        raise ValueError("Bilevel splitting requires an indexable dataset")
    if (
        train_loader.batch_size is None
        or type(train_loader.batch_sampler) is not BatchSampler
    ):
        raise ValueError(
            "Bilevel splitting requires an explicit batch_size and standard batch sampler"
        )
    horizon, stride = index(forecast_horizon), index(window_stride)
    if min(horizon, stride) < 1:
        raise ValueError("forecast_horizon and window_stride must be positive")
    base, eligible, base_indices = _eligible_indices(train_loader)
    if len(eligible) < 2:
        raise ValueError(
            "Bilevel search requires at least two eligible training samples"
        )
    origins = getattr(base, "forecast_origins", None)
    ends = getattr(base, "target_ends", None)
    split = min(max(int(0.7 * len(eligible)), 1), len(eligible) - 1)
    if origins is None and ends is None and isinstance(base_indices, range):
        # The standard sequential/shuffled full dataset needs no index allocation.
        gap = (horizon - 1) // stride
        weight_indices = range(max(0, split - gap))
        arch_indices = range(split, len(eligible))
    else:
        if origins is None and ends is None:
            windows = [
                (i * stride, i * stride + horizon - 1, local)
                for local, i in zip(eligible, base_indices)
            ]
        else:
            if (
                origins is None
                or ends is None
                or len(origins) != len(base)
                or len(ends) != len(base)
            ):
                raise ValueError(
                    "forecast_origins and target_ends must each cover the base dataset"
                )
            windows = [
                (origins[i], ends[i], local) for local, i in zip(eligible, base_indices)
            ]
            if any(not start <= end for start, end, _ in windows):
                raise ValueError("Invalid forecast target interval")
        windows.sort(key=lambda window: (window[0], window[2]))
        boundary = windows[split][0]
        weight_indices = [local for _, end, local in windows if end < boundary]
        arch_indices = [local for start, _, local in windows if start >= boundary]
    if not weight_indices:
        raise ValueError(
            "No weight-training windows remain after purging overlapping targets"
        )
    common = dict(
        batch_size=train_loader.batch_size,
        collate_fn=train_loader.collate_fn,
        num_workers=train_loader.num_workers,
        pin_memory=train_loader.pin_memory,
        drop_last=train_loader.drop_last,
        timeout=train_loader.timeout,
        worker_init_fn=train_loader.worker_init_fn,
        persistent_workers=train_loader.persistent_workers,
    )
    if train_loader.num_workers > 0:
        common.update(
            prefetch_factor=train_loader.prefetch_factor,
            multiprocessing_context=train_loader.multiprocessing_context,
        )

    def partition(indices, offset):
        loader = DataLoader(
            Subset(dataset, indices),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(seed) + offset),
            **common,
        )
        if len(loader) == 0:
            raise ValueError(
                "Bilevel partition has no batches; disable drop_last or reduce batch_size"
            )
        return loader

    return partition(arch_indices, 1), partition(weight_indices, 0)
