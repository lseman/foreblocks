from typing import Optional, Tuple

import torch
import torch.nn as nn

from foreblocks.ui.node_spec import node


@node(
    type_id="nbeats_block",
    name="N-BEATS Block",
    category="Popular",
)
class NBEATSBlock(nn.Module):
    """
    Canonical N-BEATS generic block (single block).

    - Fully-connected MLP stack
    - Produces backcast (reconstruction of input) and forecast
    - Optional weight sharing across hidden layers

    Shapes (univariate case):
        x:        [B, input_size]
        backcast: [B, input_size]
        forecast: [B, basis_size]   # usually = forecast horizon
    """

    def __init__(
        self,
        input_size: int,
        basis_size: int,  # usually = forecast horizon
        theta_size: Optional[int] = None,  # if None → input_size + basis_size
        hidden_size: int = 512,
        stack_layers: int = 4,  # total number of layers (>= 2 recommended)
        activation: str = "relu",
        share_weights_across_layers: bool = False,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        assert stack_layers >= 2, "stack_layers should be >= 2 for meaningful blocks"

        self.input_size = input_size
        self.basis_size = basis_size
        self.theta_size = (
            theta_size if theta_size is not None else (input_size + basis_size)
        )
        self.hidden_size = hidden_size
        self.stack_layers = stack_layers
        self.share_weights_across_layers = share_weights_across_layers

        act_fn = self._get_activation(activation)

        # Input projection
        self.fc_in = nn.Linear(input_size, hidden_size)

        # Hidden layers (either shared or independent)
        def make_hidden_layer():
            layers = [nn.Linear(hidden_size, hidden_size), act_fn]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_size))
            return nn.Sequential(*layers)

        if share_weights_across_layers:
            self.hidden_layer = make_hidden_layer()
            self.hidden_blocks = None
        else:
            self.hidden_blocks = nn.ModuleList(
                [make_hidden_layer() for _ in range(stack_layers - 1)]
            )
            self.hidden_layer = None

        # Theta coefficients
        self.theta_layer = nn.Linear(hidden_size, self.theta_size)

        # Basis expansions (generic version = simple linear projections)
        self.backcast_basis = nn.Linear(self.theta_size, input_size)
        self.forecast_basis = nn.Linear(self.theta_size, basis_size)

        self.act_fn = act_fn
        self.reset_parameters()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        name = name.lower()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "silu":
            return nn.SiLU()
        if name == "tanh":
            return nn.Tanh()
        return nn.ReLU()  # fallback

    def reset_parameters(self):
        # He/Kaiming init — standard for ReLU-family activations
        def init_linear(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    a=0 if self.act_fn.__class__.__name__ == "ReLU" else math.sqrt(5),
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init_linear)

        # Slightly smaller init on output layers (common practice)
        nn.init.xavier_uniform_(self.theta_layer.weight, gain=1.0)
        nn.init.xavier_uniform_(self.backcast_basis.weight, gain=1.0)
        nn.init.xavier_uniform_(self.forecast_basis.weight, gain=1.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_size]
        Returns:
            backcast:  [B, input_size]
            forecast:  [B, basis_size]
        """
        h = self.act_fn(self.fc_in(x))

        # Hidden stack
        if self.share_weights_across_layers:
            for _ in range(self.stack_layers - 1):
                h = self.hidden_layer(h)
        else:
            for layer in self.hidden_blocks:
                h = layer(h)

        # Project to interpretable coefficients
        theta = self.theta_layer(h)  # no activation here — canonical

        backcast = self.backcast_basis(theta)
        forecast = self.forecast_basis(theta)

        return backcast, forecast


@node(
    type_id="nbeats",
    name="N-BEATS",
    category="Backbone",
    color="bg-gradient-to-br from-orange-600 to-orange-800",
)
class NBEATS(nn.Module):
    """
    Minimal canonical N-BEATS model:
    - Stacks multiple generic blocks
    - Residual connection: subtract backcast from running input
    - Sum all block forecasts
    """

    def __init__(
        self,
        input_size: int,  # lookback / context length
        horizon: int,  # forecast length
        hidden_size: int = 512,
        stack_layers: int = 4,
        n_blocks: int = 4,  # total number of blocks (usually 4–30)
        theta_size: Optional[int] = None,
        activation: str = "relu",
        share_weights_in_block: bool = False,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                NBEATSBlock(
                    input_size=input_size,
                    basis_size=horizon,
                    theta_size=theta_size,
                    hidden_size=hidden_size,
                    stack_layers=stack_layers,
                    activation=activation,
                    share_weights_across_layers=share_weights_in_block,
                    dropout=dropout,
                    use_layernorm=use_layernorm,
                )
                for _ in range(n_blocks)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_size] or [B, input_size, 1] etc.
        Returns:
            forecast: [B, horizon]
        """
        residual = x
        forecast = 0.0

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast
