import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union

class MultiScaleTemporalConv(nn.Module):
    """
    Multi-scale temporal convolution block for time series.
    
    Uses parallel dilated convolutions with different dilation rates to capture
    patterns at multiple time scales efficiently. Similar to the TCN architecture
    but with parallel rather than sequential dilations.
    
    This approach has shown strong performance in various time series models.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        kernel_size: int = 3,
        dilation_rates: List[int] = [1, 2, 4, 8],
        dropout: float = 0.1,
        activation: str = "gelu",
        causal: bool = True
    ):
        """
        Args:
            input_size: Number of input channels
            output_size: Number of output channels
            kernel_size: Kernel size for all convolutions
            dilation_rates: List of dilation rates for parallel branches
            dropout: Dropout rate
            activation: Activation function
            causal: Whether to use causal (padding on left only) convolutions
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.causal = causal
        
        # Create parallel dilated convolution branches
        self.branches = nn.ModuleList()
        for dilation in dilation_rates:
            padding = ((kernel_size - 1) * dilation) if causal else ((kernel_size - 1) * dilation // 2)
            branch = nn.Sequential(
                nn.Conv1d(
                    input_size,
                    output_size,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    padding_mode='zeros'
                ),
                self._get_activation(activation),
                nn.Dropout(dropout)
            )
            self.branches.append(branch)
            
        # Weight averaging layer for combining branch outputs
        self.combiner = nn.Conv1d(
            output_size * len(dilation_rates),
            output_size,
            kernel_size=1
        )
        
        # Residual connection for stability
        self.residual = nn.Conv1d(input_size, output_size, kernel_size=1) if input_size != output_size else nn.Identity()
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
    def _get_activation(self, activation: str) -> nn.Module:
        """Get activation function by name"""
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh()
        }.get(activation.lower(), nn.GELU())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, output_size]
        """
        # Transpose for convolution: [batch, input_size, seq_len]
        x_trans = x.transpose(1, 2)
        
        # Apply each branch
        branch_outputs = []
        for branch in self.branches:
            branch_output = branch(x_trans)
            if self.causal:
                # Remove extra padding on the right for causal convolution
                branch_output = branch_output[:, :, :x_trans.size(2)]
            branch_outputs.append(branch_output)
            
        # Concatenate branch outputs along channel dimension
        multi_scale_features = torch.cat(branch_outputs, dim=1)
        
        # Combine multi-scale features
        combined = self.combiner(multi_scale_features)
        
        # Apply residual connection
        res = self.residual(x_trans)
        output = combined + res
        
        # Transpose back: [batch, seq_len, output_size]
        output = output.transpose(1, 2)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        return output

