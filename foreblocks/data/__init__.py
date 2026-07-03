"""foreblocks.data.

Dataset loaders and time-series data preprocessing pipelines.

Data provides PyTorch Dataset and DataLoader factories for time-series data,
handling common preprocessing (normalization, windowing, split strategies).
Used by ForecastingModel.fit() to prepare training batches."""

from foreblocks.data.dataset import TimeSeriesDataset, create_dataloaders


__all__ = ["TimeSeriesDataset", "create_dataloaders"]
