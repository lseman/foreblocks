"""Wrap prepared time-series arrays for PyTorch training.

Use TimeSeriesDataset for indexed samples and create_dataloaders for paired
training and validation loaders. Window construction lives in data.windowing.
"""

from foreblocks.data.dataset import TimeSeriesDataset, create_dataloaders

__all__ = ["TimeSeriesDataset", "create_dataloaders"]
