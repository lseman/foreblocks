"""foreblocks.core.training.batch_io.

Batch I/O utilities for the Trainer: unpacking, device transfer, and dataloader helpers.

Normalizes diverse batch formats (dict, tuple/list, bare tensors) into a
standard ``(X, y, time_feat, graph_kwargs)`` tuple. Handles adjacency/edge attribute
extraction from dict batches and moves all tensors to the target device. Use when
writing custom training loops or dataloaders that need consistent batch handling.

Core API:
- unpack_batch: normalize batch formats to (X, y, time_feat, graph_kwargs)
- to_device: recursively move tensors/nested containers to a device
- move_batch_to_device: full batch device transfer convenience wrapper
- loader_len: safe dataloader length (returns None if unavailable)

"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def loader_len(dataloader: Any) -> int | None:
    """Return the number of batches, or *None* if unavailable."""
    try:
        return len(dataloader)
    except TypeError:
        return None


# ---------------------------------------------------------------------------
# Batch unpacking & device transfer
# ---------------------------------------------------------------------------

_GRAPH_KEYS = {"adj", "edge_index", "edge_weight"}


def unpack_batch(
    batch: Any,
) -> tuple[Any, Any | None, Any | None, dict[str, Any]]:
    """Normalize batch formats to ``(X, y, time_feat, graph_kwargs)``.

    Handles dict batches, tuple/list batches of length 2/3+, and bare
    tensors.  ``graph_kwargs`` collects any adjacency / edge attributes
    found in the batch.
    """
    if isinstance(batch, dict):
        X = batch.get("X", batch.get("x", batch.get("src", batch.get("input"))))
        y = batch.get("y", batch.get("target", batch.get("targets")))
        time_feat = batch.get(
            "time_feat",
            batch.get("time_features", batch.get("time_feature")),
        )
        graph_kwargs = {key: batch[key] for key in _GRAPH_KEYS if key in batch}
        if X is None:
            raise ValueError(
                "Batch dict must include one of: 'X', 'x', 'src', or 'input'."
            )
        return X, y, time_feat, graph_kwargs

    if isinstance(batch, (list, tuple)):
        graph_kwargs: dict[str, Any] = {}
        if batch and isinstance(batch[-1], dict):
            maybe_graph = {
                key: value for key, value in batch[-1].items() if key in _GRAPH_KEYS
            }
            if maybe_graph:
                graph_kwargs = maybe_graph
                batch = batch[:-1]
        if len(batch) == 3:
            third = batch[2]
            if isinstance(third, dict):
                graph_kwargs.update(
                    {key: value for key, value in third.items() if key in _GRAPH_KEYS}
                )
                return batch[0], batch[1], None, graph_kwargs
            return batch[0], batch[1], third, graph_kwargs
        if len(batch) == 2:
            return batch[0], batch[1], None, graph_kwargs
        if len(batch) >= 1:
            y = batch[1] if len(batch) > 1 else None
            time_feat = batch[2] if len(batch) > 2 else None
            return batch[0], y, time_feat, graph_kwargs
        return batch, None, None, graph_kwargs

    return batch, None, None, {}


def to_device(value: Any, device: torch.device) -> Any:
    """Move tensors or simple nested containers to *device*."""
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device(item, device) for item in value)
    return value


def move_batch_to_device(
    X: Any,
    y: Any | None,
    time_feat: Any | None = None,
    graph_kwargs: dict[str, Any] | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[Any, Any | None, Any | None, dict[str, Any]]:
    """Move every tensor in a batch to the trainer device."""
    X = to_device(X, device)
    y = to_device(y, device)
    time_feat = to_device(time_feat, device)
    moved_graph_kwargs: dict[str, Any] = {}
    for key, value in (graph_kwargs or {}).items():
        moved_graph_kwargs[key] = to_device(value, device)
    return X, y, time_feat, moved_graph_kwargs
