import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, List, Tuple, Union

class MambaBlock(nn.Module):
    """
    Simplified implementation of Mamba block for time series modeling.
    
    Mamba is a state-space model that combines the benefits of RNNs and Transformers,
    with linear scaling in sequence length and selective scanning.
    
    This implementation is a simplified version of the original Mamba paper
    (https://arxiv.org/abs/2312.00752) focused on time series applications.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        state_size: int = 16,
        expand_factor: int = 2,
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dropout: float = 0.1
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden feature dimension
            state_size: Size of the internal SSM state
            expand_factor: Expansion factor for feature projection
            dt_min: Minimum value for delta parameter
            dt_max: Maximum value for delta parameter
            dt_init: Initialization method for delta ("random" or "uniform")
            dropout: Dropout probability
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.state_size = state_size
        self.expanded_size = int(hidden_size * expand_factor)
        
        # Input projection
        self.in_proj = nn.Linear(input_size, self.expanded_size)
        
        # SSM parameters
        # A is a diagonal matrix, so we just store the diagonal
        self.A_log = nn.Parameter(torch.randn(self.expanded_size, state_size))
        
        # B and C are learned for each feature
        self.B = nn.Parameter(torch.randn(self.expanded_size, state_size))
        self.C = nn.Parameter(torch.randn(self.expanded_size, state_size))
        
        # Time-step parameter (dt) - controls dynamics discretization
        if dt_init == "random":
            dt = torch.exp(torch.rand(self.expanded_size) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min))
        else:  # uniform
            dt = torch.ones(self.expanded_size) * dt_min
            
        self.dt = nn.Parameter(dt)
        
        # Output projection
        self.out_proj = nn.Linear(self.expanded_size, hidden_size)
        
        # Gating mechanism similar to GLU
        self.gate_proj = nn.Linear(input_size, self.expanded_size)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def _discretize(self, A_log: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """Discretize continuous parameters for the discrete SSM update"""
        # A_log contains log-space diagonal entries of A
        # Calculate discrete state matrix: A_discrete = exp(dt * A)
        return torch.exp(A_log.unsqueeze(-1) * dt.unsqueeze(1).unsqueeze(-1))
        
    def _scan_SSM(
        self, 
        u: torch.Tensor,
        A_discrete: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Scan the SSM over the sequence.
        
        Args:
            u: Input sequence after projection [batch, seq_len, expanded_size]
            A_discrete: Discretized state matrix [expanded_size, state_size]
            B: Input matrix [expanded_size, state_size]
            C: Output matrix [expanded_size, state_size]
            
        Returns:
            Output sequence [batch, seq_len, expanded_size]
        """
        batch, seq_len, _ = u.shape
        
        # Initialize state
        x = torch.zeros(
            batch, self.expanded_size, self.state_size, 
            device=u.device, dtype=u.dtype
        )
        
        # Output sequence
        outputs = []
        
        # Scan through sequence
        for t in range(seq_len):
            # Update state: x_t = A_discrete * x_{t-1} + B * u_t
            x = A_discrete.unsqueeze(0) * x + B.unsqueeze(0) * u[:, t, :].unsqueeze(-1)
            
            # Compute output: y_t = C * x_t
            y = torch.sum(C.unsqueeze(0) * x, dim=-1)  # [batch, expanded_size]
            
            outputs.append(y)
            
        # Stack outputs over sequence dimension
        return torch.stack(outputs, dim=1)  # [batch, seq_len, expanded_size]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor [batch_size, seq_len, hidden_size]
        """
        # Apply input projection and gating
        u = self.in_proj(x)  # [batch, seq_len, expanded_size]
        gate = torch.sigmoid(self.gate_proj(x))  # [batch, seq_len, expanded_size]
        
        # Discretize SSM parameters
        A_discrete = self._discretize(self.A_log, self.dt)
        
        # Scan SSM over sequence
        y = self._scan_SSM(u, A_discrete, self.B, self.C)
        
        # Apply gating
        y = y * gate
        
        # Project to output dimension
        y = self.out_proj(y)  # [batch, seq_len, hidden_size]
        
        # Apply dropout and layer norm
        y = self.dropout(y)
        
        # Apply residual connection
        out = self.norm(x + y)
        
        return out

