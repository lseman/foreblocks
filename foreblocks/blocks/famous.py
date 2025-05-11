import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union


class N_BEATS(nn.Module):
    """
    N-BEATS (Neural Basis Expansion Analysis for Time Series) block.
    
    A specialized neural architecture for time series forecasting based on
    backward and forward residual links and a very deep stack of fully-connected layers.
    
    This implementation is based on the N-BEATS paper by Oreshkin et al.
    (https://arxiv.org/abs/1905.10437)
    """
    def __init__(
        self,
        input_size: int,
        theta_size: int,
        basis_size: int,
        hidden_size: int = 256,
        stack_layers: int = 4,
        activation: str = "relu",
        share_weights: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Input sequence length
            theta_size: Basis expansion coefficient size
            basis_size: Number of basis functions
            hidden_size: Size of hidden layers
            stack_layers: Number of fully connected layers
            activation: Activation function
            share_weights: Whether to share weights in stack
            dropout: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_size = basis_size
        self.hidden_size = hidden_size
        self.stack_layers = stack_layers
        self.share_weights = share_weights
        
        # Fully connected stack
        if share_weights:
            self.fc_layer = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                self._get_activation(activation),
                nn.Dropout(dropout)
            )
            self.stacks = nn.ModuleList([self.fc_layer for _ in range(stack_layers)])
        else:
            self.stacks = nn.ModuleList()
            for i in range(stack_layers):
                if i == 0:
                    layer = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        self._get_activation(activation),
                        nn.Dropout(dropout)
                    )
                else:
                    layer = nn.Sequential(
                        nn.Linear(hidden_size, hidden_size),
                        self._get_activation(activation),
                        nn.Dropout(dropout)
                    )
                self.stacks.append(layer)
                
        # Basis coefficient generator
        self.theta_layer = nn.Linear(hidden_size, theta_size)
        
        # Basis functions for backward and forward signals
        self.backcast_basis = nn.Linear(theta_size, input_size)
        self.forecast_basis = nn.Linear(theta_size, basis_size)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }.get(activation.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            Tuple of:
                - Backcast (input reconstruction) [batch_size, input_size]
                - Forecast (prediction) [batch_size, basis_size]
        """
        # Stack of fully connected layers
        block_input = x
        for layer in self.stacks:
            block_input = layer(block_input)
            
        # Compute basis expansion coefficients
        theta = self.theta_layer(block_input)
        
        # Compute backcast and forecast
        backcast = self.backcast_basis(theta)
        forecast = self.forecast_basis(theta)
        
        return backcast, forecast

import torch
import torch.nn as nn
import torch.fft

class TimesBlock(nn.Module):
    def __init__(self, d_model: int, k_periods: int = 3, conv_channels: int = 64):
        super().__init__()
        self.k = k_periods
        self.d_model = d_model

        self.conv_bank = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(d_model, conv_channels, kernel_size=(3, 3), padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((None, None)),
                nn.Conv2d(conv_channels, d_model, kernel_size=1)
            ) for _ in range(k_periods)
        ])

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [B, T, C]
        Returns:
            Tensor of shape [B, T, C]
        """
        B, T, C = x.shape
        assert C == self.d_model, f"Expected input dim {self.d_model}, got {C}"

        # FFT over time
        fft_vals = torch.fft.fft(x, dim=1)
        amp = fft_vals.abs().mean(dim=2)  # [B, T]

        # Get dominant frequencies (indices)
        topk_freqs = torch.topk(amp[:, 1:T // 2], self.k, dim=1).indices + 1  # Avoid DC at index 0

        outputs = []
        periods = (T // topk_freqs).int()  # Shape [B, k]

        for i in range(self.k):
            freq = topk_freqs[:, i]
            period = periods[:, i].max().item()  # Max period across batch
            T_pad = (period - T % period) % period
            padded = F.pad(x, (0, 0, 0, T_pad))  # [B, T+pad, C]
            T_new = T + T_pad

            # Reshape: [B, T_new // period, period, C] -> [B, C, #periods, period]
            reshaped = padded.reshape(B, T_new // period, period, C).permute(0, 3, 1, 2)
            conv_out = self.conv_bank[i](reshaped)

            # Restore original shape [B, T_new, C] -> [B, T, C]
            out = conv_out.permute(0, 2, 3, 1).reshape(B, T_new, C)[:, :T, :]
            outputs.append(out)

        # Fuse using softmax weights based on amplitudes
        weights = torch.softmax(amp.gather(1, topk_freqs), dim=1)  # [B, k]
        fused = sum(w.unsqueeze(-1).unsqueeze(-1) * o for w, o in zip(weights.permute(1, 0), outputs))  # [B, T, C]
        return fused
    
class TimesBlockPreprocessor(nn.Module):
    def __init__(self, d_model=64, k_periods=3, conv_channels=64):
        super().__init__()
        self.times_block = TimesBlock(d_model=d_model, k_periods=k_periods, conv_channels=conv_channels)

    def forward(self, x):
        # Input: [B, T, C]
        return self.times_block(x)
