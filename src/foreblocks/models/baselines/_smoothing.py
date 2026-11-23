import torch
import torch.nn.functional as F


def moving_average(x: torch.Tensor, k: int) -> torch.Tensor:
    """Smooth each channel of a batch of sequences using reflected edges."""
    if k <= 1:
        return x
    channels = x.size(2)
    padding = (k - 1) // 2
    padded = F.pad(x.transpose(1, 2), (padding, padding), mode="reflect")
    weights = x.new_ones(1, 1, k) / k
    trend = F.conv1d(padded, weights.expand(channels, -1, -1), groups=channels)
    return trend.transpose(1, 2)
