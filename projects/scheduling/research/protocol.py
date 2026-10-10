"""Frozen experiment plans, durable result records, and ablation wrappers."""

from __future__ import annotations

import json
import os
import platform
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy
import torch

from ..neighborhood import MilpGraph, NeighborhoodPolicy


class AblatedPolicy(torch.nn.Module):
    def __init__(self, policy: NeighborhoodPolicy, *, no_future=False, no_edges=False):
        super().__init__()
        self.policy = policy
        self.no_future, self.no_edges = no_future, no_edges

    def forward(self, graph: MilpGraph):
        if self.no_future:
            features = graph.variables.clone()
            features[:, 9] = 0
            graph = replace(graph, variables=features)
        if self.no_edges:
            graph = replace(
                graph,
                row_index=graph.row_index[:0],
                col_index=graph.col_index[:0],
                coefficients=graph.coefficients[:0],
            )
        return self.policy(graph)


def environment_metadata():
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch_threads": torch.get_num_threads(),
        "solver_threads": 1,
        "cuda_available": torch.cuda.is_available(),
    }


def read_jsonl(path: Path):
    if not path.exists():
        return []
    data = path.read_bytes()
    # Ignore an interrupted last record; append_record repairs it before writing.
    return [
        json.loads(line)
        for line in data.splitlines(keepends=True)
        if line.endswith(b"\n") and line.strip()
    ]


def append_record(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            end = data.rfind(b"\n") + 1
            with path.open("r+b") as stream:
                stream.truncate(end)
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def objectives_at_budget(result: dict, budget: float):
    available = [
        float(event["objective"])
        for event in result["history"]
        if event["time"] <= budget and event.get("objective") is not None
    ]
    return max(available) if available else None
