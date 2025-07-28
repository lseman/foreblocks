
import torch.nn as nn
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    """Wrapper around Mamba with residual connection and normalization"""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand, **kwargs
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Pre-norm residual connection
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return residual + x


class MambaEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=2,
        d_state=16,
        d_conv=8,
        expand=2,
        dropout=0.1,
        **mamba_kwargs
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    **mamba_kwargs
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x, time_features=None):
        # Project input to hidden dimension
        x = self.input_proj(x)
        x = self.dropout(x)

        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization
        return self.norm(x)


class MambaDecoder(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=4,
        output_size=None,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
        **mamba_kwargs
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=hidden_size,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dropout=dropout,
                    **mamba_kwargs
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, output_size or input_size)

    def forward(self, x, z):
        # Project input to hidden dimension
        x = self.input_proj(x)
        x = self.dropout(x)

        # Apply Mamba layers
        for layer in self.layers:
            x = layer(x)

        # Final normalization and output projection
        x = self.norm(x)
        return self.output_proj(x)
