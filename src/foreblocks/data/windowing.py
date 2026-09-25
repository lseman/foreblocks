"""Shared NumPy windows, tensor window views, and lazy sliding-window datasets.

These input-only helpers preserve [window, time, feature] ordering. NumPy
windows are copied; tensor windows and dataset slices preserve view semantics.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset


def as_2d_array(series: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f"Expected [T] or [T,D] series, got shape {values.shape}")
    return values


def build_sliding_windows(series: np.ndarray, window_size: int) -> np.ndarray:
    x = as_2d_array(series)
    window_size = int(window_size)
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    n = x.shape[0] - window_size + 1
    if n <= 0:
        raise ValueError(
            f"Series length {x.shape[0]} is shorter than window_size={window_size}"
        )
    return (
        np.lib.stride_tricks.sliding_window_view(x, window_shape=window_size, axis=0)
        .transpose(0, 2, 1)
        .copy()
    )


def build_grouped_frames(
    series_by_group: dict[str, np.ndarray], frame_size: int, stride: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed-length frames built independently within each named series.

    No frame crosses a group boundary. `stride` defaults to `frame_size`
    (non-overlapping frames); a smaller stride overlaps within a group only.
    Returns stacked frames `[N, frame_size, D]` and a parallel `[N]` object
    array of group names, in group-iteration then within-group order. Uses a
    strided view before copying, so a large stride does not materialize every
    stride-1 window in memory.
    """
    frame_size = int(frame_size)
    if frame_size <= 0:
        raise ValueError("frame_size must be positive")
    stride = frame_size if stride is None else int(stride)
    if stride <= 0:
        raise ValueError("stride must be positive")

    frame_chunks = []
    group_chunks = []
    for name, series in series_by_group.items():
        x = as_2d_array(series)
        if x.shape[0] < frame_size:
            raise ValueError(
                f"Group {name!r} length {x.shape[0]} is shorter than "
                f"frame_size={frame_size}"
            )
        view = np.lib.stride_tricks.sliding_window_view(
            x, window_shape=frame_size, axis=0
        ).transpose(0, 2, 1)
        frames = view[::stride].copy()
        frame_chunks.append(frames)
        group_chunks.append(np.full(len(frames), name, dtype=object))
    return np.concatenate(frame_chunks, axis=0), np.concatenate(group_chunks, axis=0)


class SlidingWindowDataset(TensorDataset):
    def __init__(self, data, seq_len):
        self.data = data
        self.seq_len = seq_len
        self.length = data.shape[0] - seq_len + 1

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.data[idx : idx + self.seq_len]


def create_sequences_vectorized(data: np.ndarray, seq_len: int) -> torch.Tensor:
    if data.ndim == 1:
        data = data[:, None]

    n_samples = data.shape[0] - seq_len + 1
    if n_samples <= 0:
        raise ValueError(
            f"Data length {data.shape[0]} is too short for sequence length {seq_len}"
        )

    data_tensor = torch.from_numpy(data.T).float()
    sequences = data_tensor.unfold(1, seq_len, 1).permute(1, 2, 0)
    return sequences
