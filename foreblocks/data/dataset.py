from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


class TimeSeriesDataset(Dataset):
    """Clean dataset for time series data."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        time_feat: np.ndarray | None = None,
    ):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None
        self.t = (
            torch.tensor(time_feat, dtype=torch.long) if time_feat is not None else None
        )

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        if self.y is None and self.t is None:
            return self.X[idx]
        if self.t is None:
            return self.X[idx], self.y[idx]
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[
            idx
        ], self.t[idx]


def create_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    batch_size: int = 32,
    shuffle_train: bool = True,
    time_feat_train: np.ndarray | None = None,
    time_feat_val: np.ndarray | None = None,
) -> tuple[DataLoader, DataLoader | None]:
    train_loader = DataLoader(
        TimeSeriesDataset(X_train, y_train, time_feat_train),
        batch_size=batch_size,
        shuffle=shuffle_train,
    )

    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TimeSeriesDataset(X_val, y_val, time_feat_val),
            batch_size=batch_size,
            shuffle=False,
        )

    return train_loader, val_loader
