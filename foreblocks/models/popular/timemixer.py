"""TimeMixer-style decomposable multiscale mixing for forecasting.

Multi-granularity forecasting with PDM (Past-Decomposable-Mixing) that mixes
seasonal bottom-up and trend top-down across scales, and FMM (Future-Multipredictor-Mixing)
that ensembles predictions from all extracted scales. Creates n_levels observations
through average downsampling, processes each with MLP mixers.

Based on: Wang et al., "TimeMixer: Decomposable Multiscale Mixing for Time
Series Forecasting", ICLR 2024.
Paper: https://openreview.net/pdf?id=7oLshfEIC2

Core API:
- TimeMixer: multi-granularity forecasting with PDM and FMM
- PastDecomposableMixing: seasonal/trend mixing across scales
- FutureMultipredictorMixing: ensemble prediction from all scales
- SeriesDecomp1D: moving-average trend + seasonal decomposition

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from foreblocks.layers.norms import create_norm_layer


# =========================================================
# 1) Series decomposition (MA-based trend + seasonal)
# =========================================================
class SeriesDecomp1D(nn.Module):
    """Moving-average trend + seasonal decomposition."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        self.register_buffer("weight", None)

    def _init_weight(self, device, dtype):
        if self.weight is None:
            w = (
                torch.ones(1, 1, self.kernel_size, device=device, dtype=dtype)
                / self.kernel_size
            )
            self.register_buffer("weight", w, persistent=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._init_weight(x.device, x.dtype)
        B, L, C = x.shape
        x_n = x.permute(0, 2, 1).contiguous()  # [B, C, L]
        pad = (self.kernel_size - 1) // 2
        pad_mode = "reflect" if L > pad else "replicate"
        x_pad = F.pad(x_n, (pad, pad), mode=pad_mode)
        trend = F.conv1d(x_pad, self.weight.expand(C, -1, -1), groups=C)  # [B,C,L]
        seasonal = x_n - trend
        return (
            trend.permute(0, 2, 1).contiguous(),
            seasonal.permute(0, 2, 1).contiguous(),
        )


# =========================================================
# 2) Downsampling / Upsampling (stride-2 with 1×1 conv)
# =========================================================
class DownSample(nn.Module):
    """Stride-2 downsampling with local temporal mixing + norm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        pad = (kernel_size - stride) // 2
        self.body = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                bias=False,
            ),
            create_norm_layer(norm_type, out_channels, eps=eps),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, C = x.shape
        out = self.body(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        return out


class UpSample(nn.Module):
    """Stride-2 upsampling via repeat-interpolate + norm (matches DownSample)."""

    def __init__(
        self,
        d_model: int,
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.norm = create_norm_layer(norm_type, d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample by 2× via repeat, then normalize."""
        B, L, C = x.shape
        out = x.repeat_interleave(2, dim=1)  # [B, 2L, C]
        out = self.norm(out)
        return out


# =========================================================
# 3) Temporal Mixing (MLP across time within a granularity)
# =========================================================
class TemporalMix(nn.Module):
    """
    Mixing across time dimension within one granularity.

    Architecture:
      x → linear_in → split → [x · σ(x)] → linear_out
      with shortcut: res + out
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        expansion: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        d_hidden = max(int(d_model * expansion), 1)

        self.norm = create_norm_layer(norm_type, d_model, eps=eps)

        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_hidden
            layers.append(nn.Linear(in_dim, d_hidden if i < n_layers - 1 else d_model))
            if i < n_layers - 1:
                layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
                if i > 0 or n_layers > 1:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
                if n_layers > 1:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.norm(x)
        out = self.mlp(x)
        return res + out


# =========================================================
# 4) Channel Mixing (MLP across channels)
# =========================================================
class ChannelMix(nn.Module):
    """
    Mixing across the channel dimension within one granularity.

    Same structure as TemporalMix but applied across channels.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        expansion: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        d_hidden = max(int(d_model * expansion), 1)

        self.norm = create_norm_layer(norm_type, d_model, eps=eps)

        layers: list[nn.Module] = []
        for i in range(n_layers):
            in_dim = d_model if i == 0 else d_hidden
            layers.append(nn.Linear(in_dim, d_hidden if i < n_layers - 1 else d_model))
            if i < n_layers - 1:
                layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
                if i > 0 or n_layers > 1:
                    layers.append(nn.Dropout(dropout))
            else:
                layers.append(nn.GELU() if activation == "gelu" else nn.ReLU())
                if n_layers > 1:
                    layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, C]
        """
        res = x
        x = self.norm(x)
        out = self.mlp(x)
        return res + out


# =========================================================
# 5) PMS: Past-Mixer Stage (one granularity's processing block)
# =========================================================
class PMS(nn.Module):
    """
    Past-Mixer Stage — processes one granularity.

    Sequence:
      Input → [TemporalMix + ChannelMix] (×n_layers) → output
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        expansion: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        TemporalMix(
                            d_model=d_model,
                            n_layers=1,
                            dropout=dropout,
                            expansion=expansion,
                            activation=activation,
                            norm_type=norm_type,
                            eps=eps,
                        ),
                        ChannelMix(
                            d_model=d_model,
                            n_layers=1,
                            dropout=dropout,
                            expansion=expansion,
                            activation=activation,
                            norm_type=norm_type,
                            eps=eps,
                        ),
                    ]
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for tmix, cmix in self.layers:
            x = tmix(x)
            x = cmix(x)
        return x


# =========================================================
# 6) PDM/FMM: multiscale decomposition, mixing, and prediction
# =========================================================
class PastDecomposableMixing(nn.Module):
    """
    Mix seasonal and trend components across scales in opposite directions.

    Seasonal details move bottom-up from fine to coarse scales, while trend
    information moves top-down from coarse to fine scales.
    """

    def __init__(
        self,
        d_model: int,
        n_levels: int = 3,
        dropout: float = 0.0,
        expansion: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "rms",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.seasonal_mixers = nn.ModuleList(
            [
                TemporalMix(
                    d_model=d_model,
                    n_layers=1,
                    dropout=dropout,
                    expansion=expansion,
                    activation=activation,
                    norm_type=norm_type,
                    eps=eps,
                )
                for _ in range(n_levels - 1)
            ]
        )
        self.trend_mixers = nn.ModuleList(
            [
                TemporalMix(
                    d_model=d_model,
                    n_layers=1,
                    dropout=dropout,
                    expansion=expansion,
                    activation=activation,
                    norm_type=norm_type,
                    eps=eps,
                )
                for _ in range(n_levels - 1)
            ]
        )
        hidden = max(int(d_model * (1.0 + expansion)), d_model)
        self.feed_forward = nn.ModuleList(
            [
                nn.Sequential(
                    create_norm_layer(norm_type, d_model, eps=eps),
                    nn.Linear(d_model, hidden),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, d_model),
                )
                for _ in range(n_levels)
            ]
        )

    @staticmethod
    def _resize(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        seasonal: list[torch.Tensor],
        trend: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        if len(seasonal) != self.n_levels or len(trend) != self.n_levels:
            raise ValueError("seasonal and trend lists must match n_levels")

        s_mix = list(seasonal)
        for level in range(1, self.n_levels):
            fine = self._resize(s_mix[level - 1], s_mix[level].size(1))
            s_mix[level] = s_mix[level] + self.seasonal_mixers[level - 1](fine)

        t_mix = list(trend)
        for level in range(self.n_levels - 2, -1, -1):
            coarse = self._resize(t_mix[level + 1], t_mix[level].size(1))
            t_mix[level] = t_mix[level] + self.trend_mixers[level](coarse)

        return [
            residual + self.feed_forward[level](s_mix[level] + t_mix[level])
            for level, residual in enumerate(seasonal)
        ]


class FutureMultipredictorMixing(nn.Module):
    """Predict from every scale and learn an ensemble over scale forecasts."""

    def __init__(
        self,
        d_model: int,
        d_out: int,
        n_levels: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU() if activation == "gelu" else nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_out),
                )
                for _ in range(n_levels)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(n_levels))

    @staticmethod
    def _resize(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(1) == target_len:
            return x
        return F.interpolate(
            x.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, scales: list[torch.Tensor], pred_len: int) -> torch.Tensor:
        weights = torch.softmax(self.scale_logits[: len(scales)], dim=0)
        preds = []
        for level, x in enumerate(scales):
            h = self._resize(x, pred_len)
            preds.append(self.predictors[level](h) * weights[level])
        return torch.stack(preds, dim=0).sum(dim=0)


# =========================================================
# 7) Full TimeMixer
# =========================================================
@torch.no_grad()
def _downsample_lengths(L: int, n_levels: int) -> list[int]:
    """Compute the temporal lengths at each granularity level."""
    lengths = [L]
    for _ in range(n_levels - 1):
        lengths.append(max((lengths[-1] + 1) // 2, 1))
    return lengths


class TimeMixer(nn.Module):
    """
    TimeMixer forecasting head.

    Multi-granularity architecture:
      1. Create n_levels observations through average downsampling.
      2. Process each scale with MLP mixers.
      3. Decompose each scale and mix seasonal/trend components with PDM.
      4. Predict from all scales and ensemble them with FMM.

    Inputs:
      x: [B, L_in, C_in]

    Outputs:
      y: [B, pred_len, C_out] (or C_out × Q if quantiles)

    Args:
      pred_len: forecast horizon
      in_channels: number of input channels
      out_channels: number of output channels
      d_model: hidden dimension
      n_levels: number of temporal granularities (downsample levels)
      n_layers_pms: PMS mixing layers per granularity
      dropout: dropout rate
      expansion: MLP expansion ratio
      activation: activation function ("gelu", "relu", "swish", "mish")
      norm_type: normalization type (see layers.norms.create_norm_layer)
      eps: epsilon for normalization
      use_seasonal: whether to process seasonal component separately
      use_trend: whether to process trend component separately
      quantiles: optional quantile values for probabilistic output
    """

    def __init__(
        self,
        pred_len: int,
        in_channels: int = 1,
        out_channels: int = 1,
        d_model: int = 512,
        n_levels: int = 3,
        n_layers_pms: int = 2,
        dropout: float = 0.1,
        expansion: float = 0.5,
        activation: str = "gelu",
        norm_type: str = "rms",
        eps: float = 1e-5,
        use_seasonal: bool = True,
        use_trend: bool = True,
        quantiles: tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_levels = n_levels
        self.use_seasonal = use_seasonal
        self.use_trend = use_trend
        self.quantiles = quantiles

        d_out = out_channels if quantiles is None else out_channels * len(quantiles)

        # Input projection
        self.enc_in = nn.Linear(in_channels, d_model)

        # Decomposition
        self.decomp = SeriesDecomp1D(kernel_size=25)

        # Channel mixing for input (multi-variate C_in→C_out projection)
        self.channel_mixer = (
            nn.Linear(in_channels, out_channels)
            if in_channels != out_channels
            else nn.Identity()
        )

        # Multi-granularity processing
        # Each granularity: input_proj → PMS → output_proj
        self.granularities = nn.ModuleList()
        lengths = _downsample_lengths(256, n_levels)  # compute at reasonable max L

        for i in range(n_levels):
            self.granularities.append(
                nn.Sequential(
                    nn.Linear(d_model, d_model),
                    PMS(
                        d_model=d_model,
                        n_layers=n_layers_pms,
                        dropout=dropout,
                        expansion=expansion,
                        activation=activation,
                        norm_type=norm_type,
                        eps=eps,
                    ),
                    nn.Linear(d_model, d_model),
                )
            )

        self.pdm = PastDecomposableMixing(
            d_model=d_model,
            n_levels=n_levels,
            dropout=dropout,
            expansion=expansion,
            activation=activation,
            norm_type=norm_type,
            eps=eps,
        )

        self.fmm = FutureMultipredictorMixing(
            d_model=d_model,
            d_out=d_out,
            n_levels=n_levels,
            dropout=dropout,
            activation=activation,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _downsample(self, x: torch.Tensor, level: int) -> torch.Tensor:
        """Downsample x by stride-2, applied `level` times."""
        out = x
        for _ in range(level):
            B, L, C = out.shape
            # Pad to even
            if L % 2 != 0:
                out = F.pad(out, (0, 0, 0, 1))
            B2, L2, C = out.shape
            out = out.view(B2, L2 // 2, 2, C).mean(dim=2)  # avg pool stride 2
        return out

    def _upsample_to(self, x: torch.Tensor, target_L: int) -> torch.Tensor:
        """Upsample x to target_L via nearest-neighbor repeat."""
        B, L, C = x.shape
        if L >= target_L:
            return x[:, :target_L, :]
        # Repeat
        repeats = (target_L + L - 1) // L
        x_rep = x.repeat_interleave(torch.tensor(repeats, device=x.device), dim=1)[
            :, :target_L, :
        ]
        return x_rep

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L_in, C_in] → y: [B, pred_len, C_out]
        """
        if x.dim() != 3 or x.size(-1) != self.in_channels:
            raise ValueError(
                f"Expected x [B, L, C_in={self.in_channels}], got {tuple(x.shape)}"
            )
        B, L_in, Cin = x.shape

        z = self.enc_in(x)  # [B, L_in, D]

        seasonal_parts = []
        trend_parts = []
        for level in range(self.n_levels):
            lvl_z = self._downsample(z, level)
            lvl_out = self.granularities[level](lvl_z)
            trend, seasonal = self.decomp(lvl_out)
            seasonal_parts.append(
                seasonal if self.use_seasonal else torch.zeros_like(seasonal)
            )
            trend_parts.append(trend if self.use_trend else torch.zeros_like(trend))

        mixed_scales = self.pdm(seasonal_parts, trend_parts)
        return self.fmm(mixed_scales, self.pred_len)

    def split_quantiles(self, y: torch.Tensor) -> dict[float, torch.Tensor]:
        """Split quantile outputs into a dict keyed by quantile value."""
        if self.quantiles is None:
            raise ValueError("No quantiles configured")
        Q = len(self.quantiles)
        B, H, _ = y.shape
        yq = y.view(B, H, self.out_channels, Q)
        return {q: yq[..., i] for i, q in enumerate(self.quantiles)}
