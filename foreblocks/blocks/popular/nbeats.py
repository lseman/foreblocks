from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NBEATSBlock(nn.Module):
    """
    N-BEATS block (single block):
      - Fully-connected stack
      - Produces backcast (input reconstruction) and forecast (horizon)
      - Optional weight sharing across the repeated hidden→hidden layer

    Shapes (univariate):
      x: [B, input_size]
      backcast:  [B, input_size]
      forecast:  [B, basis_size]

    Notes:
    - This block is the atomic unit from the N-BEATS paper.
      A full model typically stacks many blocks and subtracts the
      backcast from a running residual between blocks.
    """

    def __init__(
        self,
        input_size: int,
        theta_size: int,
        basis_size: int,
        hidden_size: int = 256,
        stack_layers: int = 4,          # total depth in this block (>=1)
        activation: str = "relu",
        share_weights: bool = False,    # share the hidden→hidden layer across depths (after the first layer)
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        assert stack_layers >= 1, "stack_layers must be >= 1"
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_size = basis_size
        self.hidden_size = hidden_size
        self.stack_layers = stack_layers
        self.share_weights = share_weights
        self.use_layernorm = use_layernorm

        act = self._get_activation(activation)

        # First layer: input → hidden
        self.fc_in = nn.Linear(input_size, hidden_size)

        # Hidden→hidden building block (optionally shared)
        def make_hidden_block():
            layers = [nn.Linear(hidden_size, hidden_size), act]
            if dropout and dropout > 0:
                layers.append(nn.Dropout(dropout))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_size))
            return nn.Sequential(*layers)

        if share_weights:
            # One shared module applied repeatedly for (stack_layers-1) times
            self.shared_hidden = make_hidden_block()
            self.hidden_blocks = None  # not used
        else:
            # Independent modules per depth (after the first layer)
            self.hidden_blocks = nn.ModuleList([make_hidden_block() for _ in range(stack_layers - 1)])
            self.shared_hidden = None

        # Theta (basis coefficients) head
        self.theta_layer = nn.Linear(hidden_size, theta_size)

        # Linear “basis” projections for backcast and forecast
        # (For the canonical N-BEATS “generic” block these are linear;
        #  for “trend/seasonal” blocks you’d map theta onto fixed polynomial/Fourier bases.)
        self.backcast_basis = nn.Linear(theta_size, input_size)
        self.forecast_basis = nn.Linear(theta_size, basis_size)

        # Optional post-activation/dropout/ln after fc_in
        self.act = act
        self.post_in = nn.Sequential(*((
            [nn.Dropout(dropout)] if (dropout and dropout > 0) else []
        ) + (
            [nn.LayerNorm(hidden_size)] if use_layernorm else []
        )))

        self.reset_parameters()

    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        return {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "tanh": nn.Tanh(),
        }.get(name.lower(), nn.ReLU())

    def reset_parameters(self) -> None:
        # Kaiming initialization is common for MLPs with ReLU-family activations
        nn.init.kaiming_uniform_(self.fc_in.weight, a=0.01)
        if self.fc_in.bias is not None:
            nn.init.zeros_(self.fc_in.bias)

        if self.shared_hidden is not None:
            self._init_mlp(self.shared_hidden)
        if self.hidden_blocks is not None:
            for blk in self.hidden_blocks:
                self._init_mlp(blk)

        nn.init.xavier_uniform_(self.theta_layer.weight)
        if self.theta_layer.bias is not None:
            nn.init.zeros_(self.theta_layer.bias)

        nn.init.xavier_uniform_(self.backcast_basis.weight)
        if self.backcast_basis.bias is not None:
            nn.init.zeros_(self.backcast_basis.bias)

        nn.init.xavier_uniform_(self.forecast_basis.weight)
        if self.forecast_basis.bias is not None:
            nn.init.zeros_(self.forecast_basis.bias)

    @staticmethod
    def _init_mlp(mlp: nn.Sequential) -> None:
        for m in mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, input_size]
        Returns:
            backcast: [B, input_size]
            forecast: [B, basis_size]
        """
        # First layer
        h = self.fc_in(x)
        h = self.act(h)
        if len(self.post_in) > 0:
            h = self.post_in(h)

        # Hidden depths
        if self.stack_layers > 1:
            if self.share_weights:
                # Apply the same block repeatedly
                for _ in range(self.stack_layers - 1):
                    h = self.shared_hidden(h)
            else:
                # Apply each independent block once
                for blk in self.hidden_blocks:
                    h = blk(h)

        # Coefficients and bases
        theta = self.theta_layer(h)               # [B, theta_size]
        backcast = self.backcast_basis(theta)     # [B, input_size]
        forecast = self.forecast_basis(theta)     # [B, basis_size]
        return backcast, forecast


# (Optional) Minimal full N-BEATS wrapper that stacks blocks with residual refinement
class NBEATS(nn.Module):
    """
    Minimal N-BEATS model wrapper:
      - Stacks multiple NBEATSBlock instances
      - On each block, subtracts backcast from running residual
      - Sums all forecasts

    This mirrors the canonical “residual removal” training in N-BEATS.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        theta_size: int,
        hidden_size: int = 256,
        stack_layers: int = 4,
        n_blocks: int = 4,                 # number of blocks in the whole model
        activation: str = "relu",
        share_weights_in_block: bool = False,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([
            NBEATSBlock(
                input_size=input_size,
                theta_size=theta_size,
                basis_size=horizon,                # forecast size per block
                hidden_size=hidden_size,
                stack_layers=stack_layers,
                activation=activation,
                share_weights=share_weights_in_block,
                dropout=dropout,
                use_layernorm=use_layernorm,
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_size]
        returns: forecast [B, horizon]
        """
        residual = x
        forecast_sum = None
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual = residual - backcast
            forecast_sum = forecast if forecast_sum is None else (forecast_sum + forecast)
        return forecast_sum
