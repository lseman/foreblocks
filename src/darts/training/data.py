"""Chronological data splitting for bilevel architecture search."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, IterableDataset, Subset


def create_bilevel_loaders(train_loader: DataLoader, seed: int = 42):
    """Return architecture and weight loaders, preserving batch collation.

    Dataset order must be chronological. The first 70% fits weights and the
    remaining 30% fits architecture parameters. Callers must remove overlap
    between forecasting windows at the split. Custom sampling is replaced by
    reproducible shuffling within each partition; custom batch samplers are
    unsupported because they can encode indices from the unsplit dataset.
    """
    dataset = train_loader.dataset
    if isinstance(dataset, IterableDataset):
        raise ValueError("Bilevel splitting requires an indexable dataset")
    if train_loader.batch_size is None:
        raise ValueError("Bilevel splitting requires an explicit batch_size")
    if len(dataset) < 2:
        raise ValueError("Bilevel search requires at least two training samples")
    split = min(max(int(0.7 * len(dataset)), 1), len(dataset) - 1)
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

    return partition(range(split, len(dataset)), 1), partition(range(split), 0)
