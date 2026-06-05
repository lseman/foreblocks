"""N-BEATS forecasting blocks.

This module provides N-BEATS blocks for interpretable time series forecasting
and related basis/block components.

Based on:

    Oreshkin et al., "N-BEATS: Neural basis expansion analysis for
    interpretable time series forecasting", ICLR 2020.
    Paper: https://arxiv.org/abs/1905.10437
"""

import math

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
        theta_size: int | None = None,  # if None → input_size + basis_size
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
            self.hidden_blocks = nn.ModuleList([
                make_hidden_layer() for _ in range(stack_layers - 1)
            ])
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

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        theta_size: int | None = None,
        activation: str = "relu",
        share_weights_in_block: bool = False,
        dropout: float = 0.0,
        use_layernorm: bool = False,
        naive_level: bool = True,
    ):
        super().__init__()
        # Naive1 level init: forecast starts at the last observed value and blocks
        # learn residual corrections (matches Nixtla/neuralforecast).
        self.naive_level = naive_level
        self.blocks = nn.ModuleList([
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
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_size]  (univariate lookback)
        Returns:
            forecast: [B, horizon]
        """
        residual = x
        # Naive1 level: start from the last observed value (else from zero).
        forecast = x[:, -1:] if self.naive_level else x.new_zeros(x.shape[0], 1)

        for block in self.blocks:
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast

        return forecast


# ---------------------------------------------------------------------------
# Interpretable N-BEATS — trend (polynomial) + seasonality (Fourier) bases
# ---------------------------------------------------------------------------


def _mlp_stack(input_size: int, hidden_size: int, n_layers: int, act_fn: nn.Module,
               dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_size, hidden_size), act_fn]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_size, hidden_size), act_fn]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _InterpretableBlock(nn.Module):
    """N-BEATS interpretable block with a fixed (non-learned) basis.

    FC stack → separate θ^b / θ^f heads → fixed basis matrices give backcast and
    forecast.  ``backcast_basis`` is [input_size, theta_b] and ``forecast_basis``
    is [horizon, theta_f]; both are registered buffers (not trained).
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        backcast_basis: torch.Tensor,  # [input_size, theta_b]
        forecast_basis: torch.Tensor,  # [horizon, theta_f]
        hidden_size: int = 512,
        n_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_fn = NBEATSBlock._get_activation(activation)
        self.mlp = _mlp_stack(input_size, hidden_size, n_layers, act_fn, dropout)
        self.theta_b = nn.Linear(hidden_size, backcast_basis.shape[1], bias=False)
        self.theta_f = nn.Linear(hidden_size, forecast_basis.shape[1], bias=False)
        self.register_buffer("backcast_basis", backcast_basis)
        self.register_buffer("forecast_basis", forecast_basis)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.mlp(x)
        # basis: [out, theta] @ theta[B, theta]^T → [B, out]
        backcast = torch.einsum("ot,bt->bo", self.backcast_basis, self.theta_b(h))
        forecast = torch.einsum("ot,bt->bo", self.forecast_basis, self.theta_f(h))
        return backcast, forecast


def _trend_basis(length: int, degree: int) -> torch.Tensor:
    """Polynomial trend basis: ``[length, degree+1]`` with column i = (t)^i,
    t = [0, 1, ..., length-1] / length."""
    t = torch.arange(length, dtype=torch.float32) / max(length, 1)
    return torch.stack([t ** i for i in range(degree + 1)], dim=1)  # [length, degree+1]


def _seasonality_basis(length: int, n_harmonics: int) -> torch.Tensor:
    """Fourier seasonality basis: ``[length, 2*H]`` of cos/sin harmonics,
    H = min(n_harmonics, length//2)."""
    t = torch.arange(length, dtype=torch.float32) / max(length, 1)
    H = max(1, min(n_harmonics, length // 2))
    freqs = torch.arange(1, H + 1, dtype=torch.float32)  # harmonics 1..H
    arg = 2.0 * math.pi * freqs[None, :] * t[:, None]  # [length, H]
    return torch.cat([torch.cos(arg), torch.sin(arg)], dim=1)  # [length, 2H]


@node(
    type_id="nbeats_interpretable",
    name="N-BEATS (Interpretable)",
    category="Backbone",
    color="bg-gradient-to-br from-orange-600 to-amber-700",
)
class NBEATSInterpretable(nn.Module):
    """Interpretable N-BEATS: a trend stack followed by a seasonality stack.

    - Trend stack: ``trend_blocks`` blocks with a shared polynomial basis
      (degree ``trend_degree``).
    - Seasonality stack: ``season_blocks`` blocks with a shared Fourier basis
      (``n_harmonics`` harmonics).
    Doubly-residual across all blocks; forecasts summed.

    Input : x [B, input_size]
    Output: forecast [B, horizon]

    ``trend_forecast`` / ``seasonality_forecast`` are exposed via
    :meth:`decompose` for interpretability.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        trend_blocks: int = 3,
        season_blocks: int = 3,
        trend_degree: int = 2,
        n_harmonics: int = 5,
        hidden_size: int = 512,
        n_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
        naive_level: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.naive_level = naive_level

        trend_bc = _trend_basis(input_size, trend_degree)
        trend_fc = _trend_basis(horizon, trend_degree)
        season_bc = _seasonality_basis(input_size, n_harmonics)
        season_fc = _seasonality_basis(horizon, n_harmonics)

        def block(bc_basis, fc_basis):
            return _InterpretableBlock(
                input_size, horizon, bc_basis.clone(), fc_basis.clone(),
                hidden_size=hidden_size, n_layers=n_layers,
                activation=activation, dropout=dropout,
            )

        self.trend_stack = nn.ModuleList(
            [block(trend_bc, trend_fc) for _ in range(trend_blocks)]
        )
        self.season_stack = nn.ModuleList(
            [block(season_bc, season_fc) for _ in range(season_blocks)]
        )

    def _run(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = x
        # Naive1 level init (Nixtla): folded into the trend component.
        level = x[:, -1:] if self.naive_level else x.new_zeros(x.shape[0], 1)
        trend_fc = level.expand(-1, self.horizon).clone()
        season_fc = x.new_zeros(x.shape[0], self.horizon)
        for blk in self.trend_stack:
            backcast, fc = blk(residual)
            residual = residual - backcast
            trend_fc = trend_fc + fc
        for blk in self.season_stack:
            backcast, fc = blk(residual)
            residual = residual - backcast
            season_fc = season_fc + fc
        return trend_fc + season_fc, trend_fc, season_fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x)[0]

    @torch.no_grad()
    def decompose(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the trend and seasonality forecast components."""
        total, trend, season = self._run(x)
        return {"forecast": total, "trend": trend, "seasonality": season}


# ---------------------------------------------------------------------------
# N-BEATSx — N-BEATS with exogenous variables
#   Olivares, Challu, Marcjasz, Weron, Dubrawski,
#   "Neural basis expansion analysis with exogenous variables: Forecasting
#    electricity prices with NBEATSx", IJF 2023. https://arxiv.org/abs/2104.05522
# ---------------------------------------------------------------------------


class _ExogTCN(nn.Module):
    """Small causal-TCN encoder over exogenous covariates → ``n_filters`` channels.

    Maps exog ``[B, n_exog, T]`` → basis ``[B, n_filters, T]`` (NBEATSx-W/wavelet
    flavour). Stacked dilated causal convs preserve the time length T.
    """

    def __init__(self, n_exog: int, n_filters: int, kernel_size: int = 3,
                 n_layers: int = 2, activation: str = "relu"):
        super().__init__()
        self.act = NBEATSBlock._get_activation(activation)
        self.convs = nn.ModuleList()
        self.trims: list[int] = []
        in_ch = n_exog
        for i in range(n_layers):
            dil = 2 ** i
            pad = (kernel_size - 1) * dil  # causal: pad both ends, trim the right
            self.convs.append(
                nn.Conv1d(in_ch, n_filters, kernel_size, dilation=dil, padding=pad)
            )
            self.trims.append(pad)
            in_ch = n_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_exog, T] → [B, n_filters, T]
        out = x
        for conv, trim in zip(self.convs, self.trims):
            out = conv(out)
            if trim > 0:
                out = out[..., :-trim]  # drop right padding → causal, length T
            out = self.act(out)
        return out


class _ExogBlock(nn.Module):
    """N-BEATSx exogenous block.

    The MLP consumes ``[y_lookback, exog_insample(flat), exog_outsample(flat)]``
    and emits coefficients ``θ`` that are combined with an exogenous *basis*:

      * ``mode="linear"`` (NBEATSx-G): basis = the exog covariates themselves;
        ``forecast = sum_e θ_e * exog_out[:, e, :]``  (θ size = n_exog).
      * ``mode="tcn"`` (NBEATSx-W): a shared causal-TCN encodes the exog into
        ``n_filters`` channels first; θ size = n_filters.

    Backcast uses the insample exog with the same θ-basis contraction so the
    block stays doubly-residual on ``y``.
    """

    def __init__(
        self,
        input_size: int,       # L
        horizon: int,          # H
        n_exog: int,
        mode: str = "linear",  # "linear" | "tcn"
        n_filters: int = 32,
        hidden_size: int = 512,
        n_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
        tcn_kernel: int = 3,
        tcn_layers: int = 2,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_exog = n_exog
        self.mode = mode

        if mode == "tcn":
            self.encoder = _ExogTCN(n_exog, n_filters, tcn_kernel, tcn_layers, activation)
            self.basis_dim = n_filters
        elif mode == "linear":
            self.encoder = None
            self.basis_dim = n_exog
        else:
            raise ValueError("mode must be 'linear' or 'tcn'")

        act_fn = NBEATSBlock._get_activation(activation)
        mlp_in = input_size + n_exog * (input_size + horizon)
        self.mlp = _mlp_stack(mlp_in, hidden_size, n_layers, act_fn, dropout)
        self.theta_b = nn.Linear(hidden_size, self.basis_dim, bias=False)
        self.theta_f = nn.Linear(hidden_size, self.basis_dim, bias=False)

    def forward(
        self,
        y: torch.Tensor,          # [B, L]
        exog_in: torch.Tensor,    # [B, n_exog, L]
        exog_out: torch.Tensor,   # [B, n_exog, H]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = y.shape[0]
        feats = torch.cat(
            [y, exog_in.reshape(B, -1), exog_out.reshape(B, -1)], dim=1
        )
        h = self.mlp(feats)

        if self.encoder is not None:
            basis_in = self.encoder(exog_in)    # [B, F, L]
            basis_out = self.encoder(exog_out)  # [B, F, H]
        else:
            basis_in, basis_out = exog_in, exog_out  # [B, n_exog, *]

        # backcast/forecast = sum_e θ_e * basis[:, e, :]
        backcast = torch.einsum("be,bet->bt", self.theta_b(h), basis_in)
        forecast = torch.einsum("be,bet->bt", self.theta_f(h), basis_out)
        return backcast, forecast


@node(
    type_id="nbeatsx",
    name="N-BEATSx",
    category="Backbone",
    color="bg-gradient-to-br from-amber-600 to-yellow-700",
)
class NBEATSx(nn.Module):
    """N-BEATS with exogenous variables.

    Stacks (each doubly-residual on the target ``y``):
      * trend stack       — polynomial basis (target only)
      * seasonality stack — Fourier basis (target only)
      * exogenous stack   — ``exog_blocks`` :class:`_ExogBlock` (linear or TCN)

    Input
    -----
    y         : [B, input_size]                 lookback target
    exog_in   : [B, input_size, n_exog]         insample exogenous (or None)
    exog_out  : [B, horizon, n_exog]            future exogenous (or None)

    Output: forecast [B, horizon].  With ``n_exog=0`` this reduces to interpretable
    N-BEATS. :meth:`decompose` returns the per-stack forecast components.
    """

    def __init__(
        self,
        input_size: int,
        horizon: int,
        n_exog: int = 0,
        exog_mode: str = "linear",   # "linear" | "tcn"
        trend_blocks: int = 1,
        season_blocks: int = 1,
        exog_blocks: int = 1,
        trend_degree: int = 2,
        n_harmonics: int = 5,
        n_filters: int = 32,
        hidden_size: int = 512,
        n_layers: int = 4,
        activation: str = "relu",
        dropout: float = 0.0,
        naive_level: bool = True,
    ):
        super().__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.n_exog = n_exog
        self.naive_level = naive_level

        trend_bc = _trend_basis(input_size, trend_degree)
        trend_fc = _trend_basis(horizon, trend_degree)
        season_bc = _seasonality_basis(input_size, n_harmonics)
        season_fc = _seasonality_basis(horizon, n_harmonics)

        def interp(bc, fc):
            return _InterpretableBlock(
                input_size, horizon, bc.clone(), fc.clone(),
                hidden_size=hidden_size, n_layers=n_layers,
                activation=activation, dropout=dropout,
            )

        self.trend_stack = nn.ModuleList([interp(trend_bc, trend_fc) for _ in range(trend_blocks)])
        self.season_stack = nn.ModuleList([interp(season_bc, season_fc) for _ in range(season_blocks)])

        if n_exog > 0:
            self.exog_stack = nn.ModuleList([
                _ExogBlock(
                    input_size, horizon, n_exog, mode=exog_mode, n_filters=n_filters,
                    hidden_size=hidden_size, n_layers=n_layers,
                    activation=activation, dropout=dropout,
                )
                for _ in range(exog_blocks)
            ])
        else:
            self.exog_stack = nn.ModuleList()

    def _run(self, y, exog_in, exog_out):
        residual = y
        # Naive1 level init (Nixtla): forecast starts at the last observed value.
        level = y[:, -1:] if self.naive_level else y.new_zeros(y.shape[0], 1)
        total = level.expand(-1, self.horizon).clone()
        comps: dict[str, torch.Tensor] = {}

        ein = exog_in.transpose(1, 2) if len(self.exog_stack) > 0 else None
        eout = exog_out.transpose(1, 2) if len(self.exog_stack) > 0 else None

        # flat ordered schedule of (block, name, is_exog)
        schedule = (
            [(b, "trend", False) for b in self.trend_stack]
            + [(b, "seasonality", False) for b in self.season_stack]
            + [(b, "exogenous", True) for b in self.exog_stack]
        )
        if self.naive_level:
            comps["level"] = level.expand(-1, self.horizon)
        for blk, name, is_exog in schedule:
            bc, fc = blk(residual, ein, eout) if is_exog else blk(residual)
            residual = residual - bc
            comps[name] = comps.get(name, 0.0) + fc
            total = total + fc
        return total, comps

    def forward(
        self,
        y: torch.Tensor,
        exog_in: torch.Tensor | None = None,
        exog_out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if len(self.exog_stack) > 0 and (exog_in is None or exog_out is None):
            raise ValueError("NBEATSx with n_exog>0 requires exog_in and exog_out")
        return self._run(y, exog_in, exog_out)[0]

    @torch.no_grad()
    def decompose(
        self,
        y: torch.Tensor,
        exog_in: torch.Tensor | None = None,
        exog_out: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        total, comps = self._run(y, exog_in, exog_out)
        return {"forecast": total, **comps}
