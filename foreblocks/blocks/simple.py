import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Gated Residual Network (GRN) from Temporal Fusion Transformer
    
    The GRN applies non-linear processing with gating and residual connections:
    1. Linear transformation
    2. ELU activation  
    3. Linear transformation
    4. GLU (Gated Linear Unit) for gating
    5. Dropout
    6. Residual connection with optional skip connection
    """
    
    def __init__(self, input_size, hidden_size=None, output_size=None, dropout=0.0, 
                 context_size=None, use_time_distributed=False):
        super().__init__()
        
        if hidden_size is None:
            hidden_size = input_size
        if output_size is None:
            output_size = input_size
            
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.context_size = context_size
        self.use_time_distributed = use_time_distributed
        
        # First linear layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Context projection (if context is provided)
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        
        # Second linear layer (output size is doubled for GLU)
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        
        # Output projection
        self.output_projection = nn.Linear(hidden_size, output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Skip connection projection (if input and output sizes differ)
        if input_size != output_size:
            self.skip_projection = nn.Linear(input_size, output_size)
        else:
            self.skip_projection = None

    def forward(self, x, context=None):
        """
        Forward pass of GRN
        
        Args:
            x: Input tensor of shape [..., input_size]
            context: Optional context tensor of shape [..., context_size]
            
        Returns:
            Output tensor of shape [..., output_size]
        """
        # Store original input for residual connection
        residual = x
        
        # First linear transformation
        x = self.fc1(x)
        
        # Add context if provided
        if context is not None and self.context_projection is not None:
            x = x + self.context_projection(context)
        
        # ELU activation
        x = F.elu(x)
        
        # Second linear transformation (doubled output for GLU)
        x = self.fc2(x)
        
        # Split for Gated Linear Unit (GLU)
        # GLU(x) = x_1 ⊙ σ(x_2) where x = [x_1, x_2]
        x, gate = x.chunk(2, dim=-1)
        x = x * torch.sigmoid(gate)
        
        # Dropout
        x = self.dropout(x)
        
        # Output projection
        x = self.output_projection(x)
        
        # Skip connection
        if self.skip_projection is not None:
            residual = self.skip_projection(residual)
        
        # Add residual connection
        x = x + residual
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x